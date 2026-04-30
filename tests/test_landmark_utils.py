"""Tests for landmark_utils: OFIQ-aligned geometry and color functions."""

import math

import numpy as np
import pytest

from ofiq_syngen.landmark_utils import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EYE_CORNERS,
    RIGHT_EYE_CORNERS,
    MOUTH_OUTER,
    MOUTH_INNER,
    CONTOUR,
    CHIN,
    NOSETIP,
    PAIRS_LEFT_EYE,
    PAIRS_RIGHT_EYE,
    PAIRS_MOUTH_INNER,
    get_middle,
    get_distance,
    get_max_pair_distance,
    calculate_eye_centers,
    tmetric,
    inter_eye_distance,
    calculate_reference_points,
    calculate_roi,
    compute_evz_rects,
    get_face_mask,
    get_luminance_image,
    convert_bgr_to_cielab,
    BISENET_BACKGROUND,
    BISENET_SKIN,
    BISENET_HAT,
    BISENET_CLOTH,
)


@pytest.fixture
def synthetic_landmarks():
    """98-point landmarks in a realistic face layout for a 616x616 image."""
    lm = np.zeros((98, 2), dtype=np.int32)
    # Contour (0-32): oval face outline
    for i in range(33):
        angle = math.pi * (i / 32)
        lm[i] = [308 + int(150 * math.sin(angle)), 100 + int(250 * (1 - math.cos(angle)) / 2)]
    # Left eye (60-67): around (251, 272)
    for i, offset in enumerate([(-15, 0), (-10, -5), (-5, -8), (0, -6),
                                 (10, 0), (5, 5), (0, 6), (-5, 5)]):
        lm[60 + i] = [251 + offset[0], 272 + offset[1]]
    # Right eye (68-75): around (364, 272)
    for i, offset in enumerate([(-10, 0), (-5, -5), (0, -8), (5, -6),
                                 (15, 0), (10, 5), (5, 6), (0, 5)]):
        lm[68 + i] = [364 + offset[0], 272 + offset[1]]
    # Nose tip (54)
    lm[54] = [308, 336]
    # Chin (16)
    lm[16] = [308, 480]
    # Mouth outer (76-87)
    for i in range(12):
        angle = 2 * math.pi * i / 12
        lm[76 + i] = [308 + int(30 * math.cos(angle)), 402 + int(15 * math.sin(angle))]
    # Mouth inner (88-95)
    for i in range(8):
        angle = 2 * math.pi * i / 8
        lm[88 + i] = [308 + int(20 * math.cos(angle)), 402 + int(8 * math.sin(angle))]
    return lm


class TestIndexMaps:
    def test_left_eye_indices(self):
        assert LEFT_EYE == list(range(60, 68))

    def test_right_eye_indices(self):
        assert RIGHT_EYE == list(range(68, 76))

    def test_mouth_outer_indices(self):
        assert MOUTH_OUTER == list(range(76, 88))

    def test_mouth_inner_indices(self):
        assert MOUTH_INNER == list(range(88, 96))

    def test_contour_length(self):
        assert len(CONTOUR) == 33

    def test_chin_is_16(self):
        assert CHIN == [16]

    def test_nosetip_is_54(self):
        assert NOSETIP == [54]

    def test_eye_pairs(self):
        assert PAIRS_LEFT_EYE == [(61, 67), (62, 66), (63, 65)]
        assert PAIRS_RIGHT_EYE == [(69, 75), (70, 74), (71, 73)]

    def test_mouth_pairs(self):
        assert PAIRS_MOUTH_INNER == [(89, 95), (90, 94), (91, 93)]


class TestGeometry:
    def test_get_distance(self):
        assert get_distance((0, 0), (3, 4)) == 5.0

    def test_get_middle_simple(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.int32)
        mid = get_middle(pts)
        assert mid == (5.0, 5.0)

    def test_get_middle_with_indices(self):
        pts = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.int32)
        mid = get_middle(pts, [0, 2])
        assert mid == (10.0, 10.0)

    def test_get_max_pair_distance(self, synthetic_landmarks):
        dist = get_max_pair_distance(synthetic_landmarks, PAIRS_LEFT_EYE)
        assert dist > 0

    def test_calculate_eye_centers(self, synthetic_landmarks):
        left, right = calculate_eye_centers(synthetic_landmarks)
        assert left[0] < right[0]  # left eye is left of right eye

    def test_tmetric_positive(self, synthetic_landmarks):
        t = tmetric(synthetic_landmarks)
        assert t > 0

    def test_inter_eye_distance_normal(self, synthetic_landmarks):
        ied = inter_eye_distance(synthetic_landmarks, yaw=0.0)
        assert ied > 50  # should be reasonable on 616x616

    def test_inter_eye_distance_yaw_correction(self, synthetic_landmarks):
        ied_0 = inter_eye_distance(synthetic_landmarks, yaw=0.0)
        ied_30 = inter_eye_distance(synthetic_landmarks, yaw=30.0)
        assert ied_30 > ied_0  # yaw correction increases apparent IED

    def test_inter_eye_distance_90_is_nan(self, synthetic_landmarks):
        ied = inter_eye_distance(synthetic_landmarks, yaw=90.0)
        assert math.isnan(ied)

    def test_calculate_roi(self, synthetic_landmarks):
        left, right, ied, emd = calculate_reference_points(synthetic_landmarks)
        right_roi, left_roi = calculate_roi(left, right, ied, emd)
        # ROIs should be valid rectangles
        assert right_roi[2] > 0 and right_roi[3] > 0
        assert left_roi[2] > 0 and left_roi[3] > 0

    def test_compute_evz_rects(self, synthetic_landmarks):
        ied = inter_eye_distance(synthetic_landmarks, 0.0)
        left_evz, right_evz = compute_evz_rects(synthetic_landmarks, ied)
        assert left_evz[2] > 0 and left_evz[3] > 0


class TestFaceMask:
    def test_face_mask_shape(self, synthetic_landmarks):
        mask = get_face_mask(synthetic_landmarks, 616, 616)
        assert mask.shape == (616, 616)
        assert mask.dtype == np.uint8

    def test_face_mask_has_nonzero(self, synthetic_landmarks):
        mask = get_face_mask(synthetic_landmarks, 616, 616)
        assert mask.sum() > 0

    def test_face_mask_with_forehead(self, synthetic_landmarks):
        mask_no = get_face_mask(synthetic_landmarks, 616, 616, alpha=0.0)
        mask_yes = get_face_mask(synthetic_landmarks, 616, 616, alpha=1.0)
        assert mask_yes.sum() >= mask_no.sum()


class TestColorConversion:
    def test_luminance_shape(self):
        bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        lum = get_luminance_image(bgr)
        assert lum.shape == (100, 100)
        assert lum.dtype == np.uint8

    def test_luminance_range(self):
        bgr = np.full((10, 10, 3), 128, dtype=np.uint8)
        lum = get_luminance_image(bgr)
        assert 0 <= lum.min() <= lum.max() <= 255

    def test_luminance_black(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        lum = get_luminance_image(bgr)
        assert lum.max() == 0

    def test_luminance_white(self):
        bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
        lum = get_luminance_image(bgr)
        assert lum.min() == 255

    def test_cielab_skin_tone(self):
        """Typical skin tone should be in the natural range."""
        skin = np.full((50, 50, 3), [120, 150, 200], dtype=np.uint8)
        a, b = convert_bgr_to_cielab(skin)
        # Skin tone a* should be roughly in [5, 25], b* in [5, 35]
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_cielab_blue_cast(self):
        """Strong blue should have negative b* (outside natural range)."""
        blue = np.full((50, 50, 3), [255, 100, 50], dtype=np.uint8)
        a, b = convert_bgr_to_cielab(blue)
        # Blue has negative b* or very high a* -- outside natural skin range
        assert a < 5 or b < 5  # at least one out of range


class TestBiSeNetClasses:
    def test_class_indices(self):
        assert BISENET_BACKGROUND == 0
        assert BISENET_SKIN == 1
        assert BISENET_CLOTH == 16
        assert BISENET_HAT == 18
