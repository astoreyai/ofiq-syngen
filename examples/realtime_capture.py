"""Real-time OFIQ-aligned face capture with live quality feedback.

Streams from a webcam (or a video file with --video), scores every frame
against OFIQ-aligned components in a background thread, renders a live
overlay grouping the 28 components into 5 user-facing panels, and triggers
auto-capture when all panels stay green for a configurable number of
consecutive scored frames.

Usage:
    python examples/realtime_capture.py                    # webcam id 0
    python examples/realtime_capture.py --camera 1
    python examples/realtime_capture.py --video clip.mp4   # for testing
    python examples/realtime_capture.py --threshold 80     # stricter
    python examples/realtime_capture.py --no-autocapture   # manual only
    python examples/realtime_capture.py --output /tmp/captures

Keys:
    SPACE  manual capture
    R      reset capture countdown
    Q/ESC  quit

Output:
    <output-dir>/capture_YYYYMMDD_HHMMSS.png
    <output-dir>/capture_YYYYMMDD_HHMMSS.json   (per-component scores)

This is an example, not a production application. The capture criteria,
guidance copy, and panel groupings are illustrative.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# -- Component groupings ------------------------------------------------------
# 28 OFIQ components map to 5 panels for human-readable feedback. Each
# panel's worst component drives its color and guidance line.

PANELS: dict[str, dict] = {
    "LIGHTING": {
        "components": [
            "LuminanceMean.scalar",
            "LuminanceVariance.scalar",
            "UnderExposurePrevention.scalar",
            "OverExposurePrevention.scalar",
            "DynamicRange.scalar",
            "IlluminationUniformity.scalar",
            "BackgroundUniformity.scalar",
            "NaturalColour.scalar",
        ],
        "guidance": {
            "LuminanceMean.scalar": "Adjust brightness; the scene is too dark or too bright.",
            "UnderExposurePrevention.scalar": "Add light; underexposed.",
            "OverExposurePrevention.scalar": "Reduce light; overexposed.",
            "DynamicRange.scalar": "Improve scene contrast.",
            "IlluminationUniformity.scalar": "Light both sides of the face evenly.",
            "BackgroundUniformity.scalar": "Use a plainer background.",
            "NaturalColour.scalar": "White-balance the camera; skin tone is off.",
        },
    },
    "FOCUS": {
        "components": [
            "Sharpness.scalar",
            "CompressionArtifacts.scalar",
            "RadialDistortion.scalar",
        ],
        "guidance": {
            "Sharpness.scalar": "Hold still; the image is blurry.",
            "CompressionArtifacts.scalar": "Reduce compression; visible artifacts.",
            "RadialDistortion.scalar": "Use a different lens; barrel distortion present.",
        },
    },
    "POSE": {
        "components": [
            "HeadPoseYaw.scalar",
            "HeadPosePitch.scalar",
            "HeadPoseRoll.scalar",
            "LeftwardCropOfTheFaceImage.scalar",
            "RightwardCropOfTheFaceImage.scalar",
            "MarginAboveOfTheFaceImage.scalar",
            "MarginBelowOfTheFaceImage.scalar",
            "InterEyeDistance.scalar",
            "HeadSize.scalar",
        ],
        "guidance": {
            "HeadPoseYaw.scalar": "Look straight at the camera.",
            "HeadPosePitch.scalar": "Level your chin; do not tilt up or down.",
            "HeadPoseRoll.scalar": "Hold your head upright.",
            "LeftwardCropOfTheFaceImage.scalar": "Center your face left-to-right.",
            "RightwardCropOfTheFaceImage.scalar": "Center your face left-to-right.",
            "MarginAboveOfTheFaceImage.scalar": "Lower the camera or move down.",
            "MarginBelowOfTheFaceImage.scalar": "Raise the camera or move up.",
            "InterEyeDistance.scalar": "Move closer to the camera.",
            "HeadSize.scalar": "Move closer; your face is too small in the frame.",
        },
    },
    "EXPRESSION": {
        "components": [
            "ExpressionNeutrality.scalar",
            "EyesOpen.scalar",
            "MouthClosed.scalar",
        ],
        "guidance": {
            "ExpressionNeutrality.scalar": "Relax your face; neutral expression.",
            "EyesOpen.scalar": "Open your eyes wider.",
            "MouthClosed.scalar": "Close your mouth.",
        },
    },
    "OCCLUSION": {
        "components": [
            "EyesVisible.scalar",
            "MouthOcclusionPrevention.scalar",
            "FaceOcclusionPrevention.scalar",
            "NoHeadCoverings.scalar",
            "SingleFacePresent.scalar",
        ],
        "guidance": {
            "EyesVisible.scalar": "Remove sunglasses or anything covering your eyes.",
            "MouthOcclusionPrevention.scalar": "Move hands or masks away from your mouth.",
            "FaceOcclusionPrevention.scalar": "Move hair or objects off your face.",
            "NoHeadCoverings.scalar": "Remove hats or non-religious head coverings.",
            "SingleFacePresent.scalar": "Only one person should be in frame.",
        },
    },
}

PANEL_ORDER = ["LIGHTING", "FOCUS", "POSE", "EXPRESSION", "OCCLUSION"]


@dataclass
class ScoreSnapshot:
    """Latest scoring result published from the background thread."""

    scores: dict[str, float]
    timestamp: float
    inference_ms: float
    error: str | None = None


# -- Background scorer --------------------------------------------------------


class BackgroundScorer:
    """Runs OFIQ scoring off the main thread to keep the UI responsive."""

    def __init__(self, score_fn, dropping_queue_size: int = 1):
        self._score_fn = score_fn
        self._inbox: queue.Queue = queue.Queue(maxsize=dropping_queue_size)
        self._lock = threading.Lock()
        self._latest: ScoreSnapshot | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray) -> None:
        """Submit a frame for scoring. Drops if the inbox is full."""
        try:
            self._inbox.put_nowait(frame)
        except queue.Full:
            pass

    def latest(self) -> ScoreSnapshot | None:
        """Return the most recent scored snapshot, or None if none yet."""
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            t0 = time.perf_counter()
            try:
                scores = self._score_fn(frame)
                err = None
            except Exception as exc:
                scores = {}
                err = str(exc)
            t1 = time.perf_counter()
            snap = ScoreSnapshot(
                scores=scores,
                timestamp=t1,
                inference_ms=(t1 - t0) * 1000.0,
                error=err,
            )
            with self._lock:
                self._latest = snap


def _build_score_fn(model_dir: str | None):
    """Resolve the scoring function. Falls back to a mock when models absent."""
    try:
        from ofiq_syngen.gpu_ofiq_scorer import GPUOFIQScorer
        scorer = GPUOFIQScorer(model_dir=model_dir)
        return scorer.score_image
    except Exception as exc:
        print(
            f"[warn] GPUOFIQScorer unavailable ({exc}); using mock scorer.",
            file=sys.stderr,
        )
        return _mock_score


def _mock_score(image: np.ndarray) -> dict[str, float]:
    """Mock scorer for environments without OFIQ models."""
    from ofiq_syngen.standards import STANDARDS_REFS
    h, w = image.shape[:2]
    base = float(np.clip((np.mean(image) / 255.0) * 100, 30, 95))
    return {comp: base + np.random.normal(0, 5) for comp in STANDARDS_REFS}


# -- Panel evaluation ---------------------------------------------------------


def evaluate_panel(panel_name: str, scores: dict[str, float], threshold: float):
    """Return (color, status, guidance) for a panel given current scores."""
    panel = PANELS[panel_name]
    components = panel["components"]
    available = [(c, scores[c]) for c in components if c in scores]

    if not available:
        return ((128, 128, 128), "NO DATA", "Waiting for scores...")

    worst_comp, worst_score = min(available, key=lambda kv: kv[1])

    if worst_score >= threshold:
        return ((40, 200, 40), "OK", "")
    if worst_score >= threshold * 0.6:
        guidance = panel["guidance"].get(worst_comp, f"Adjust {worst_comp}")
        return ((0, 200, 230), "WARN", guidance)
    guidance = panel["guidance"].get(worst_comp, f"Fix {worst_comp}")
    return ((40, 40, 220), "FAIL", guidance)


# -- Overlay rendering --------------------------------------------------------


def render_overlay(
    frame: np.ndarray,
    snapshot: ScoreSnapshot | None,
    threshold: float,
    countdown: int,
    autocapture_target: int,
    fps: float,
) -> np.ndarray:
    """Compose the live overlay on top of the camera frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    panel_h = 70
    bar_h = 30
    overlay_h = panel_h * len(PANEL_ORDER) + bar_h
    overlay = np.zeros((overlay_h, 360, 3), dtype=np.uint8)
    overlay[:] = (24, 24, 24)

    cv2.putText(
        overlay, f"FPS: {fps:5.1f}", (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    if snapshot is not None:
        cv2.putText(
            overlay,
            f"score age: {(time.perf_counter() - snapshot.timestamp) * 1000:5.0f}ms",
            (110, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )
    cv2.putText(
        overlay, f"thresh: {threshold:.0f}", (260, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )

    if snapshot is not None and snapshot.error:
        cv2.putText(
            overlay, f"err: {snapshot.error[:36]}", (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 240), 1, cv2.LINE_AA,
        )

    scores = snapshot.scores if snapshot else {}
    for i, panel_name in enumerate(PANEL_ORDER):
        color, status, guidance = evaluate_panel(panel_name, scores, threshold)
        y0 = bar_h + i * panel_h
        cv2.rectangle(overlay, (5, y0 + 5), (15, y0 + panel_h - 5), color, -1)
        cv2.putText(
            overlay, f"{panel_name:11s} {status}", (22, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
        if guidance:
            cv2.putText(
                overlay, guidance[:46], (22, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA,
            )

    out[:overlay_h, :360] = overlay

    if countdown > 0 and autocapture_target > 0:
        ratio = countdown / autocapture_target
        bar_w = int(w * 0.3)
        bx = (w - bar_w) // 2
        by = h - 40
        cv2.rectangle(out, (bx, by), (bx + bar_w, by + 18), (50, 50, 50), -1)
        cv2.rectangle(
            out, (bx, by), (bx + int(bar_w * ratio), by + 18),
            (40, 200, 40), -1,
        )
        cv2.putText(
            out, f"capturing in {autocapture_target - countdown}",
            (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (200, 255, 200), 1, cv2.LINE_AA,
        )

    cv2.putText(
        out, "SPACE: capture  R: reset  Q: quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (180, 180, 180), 1, cv2.LINE_AA,
    )

    return out


# -- Capture handling ---------------------------------------------------------


def all_panels_green(scores: dict[str, float], threshold: float) -> bool:
    if not scores:
        return False
    for panel_name in PANEL_ORDER:
        color, _, _ = evaluate_panel(panel_name, scores, threshold)
        if color != (40, 200, 40):
            return False
    return True


def save_capture(
    frame: np.ndarray, scores: dict[str, float], output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_path = output_dir / f"capture_{stamp}.png"
    json_path = output_dir / f"capture_{stamp}.json"
    cv2.imwrite(str(img_path), frame)
    with json_path.open("w") as f:
        json.dump(
            {
                "timestamp": stamp,
                "scores": scores,
                "ofiq_syngen_version": _package_version(),
            },
            f, indent=2, sort_keys=True,
        )
    return img_path


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("ofiq-syngen")
    except Exception:
        return "unknown"


# -- Main loop ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None,
                        help="Read from a video file instead of a webcam.")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="Per-component pass threshold (0-100).")
    parser.add_argument("--autocapture-frames", type=int, default=30,
                        help="Consecutive scored frames all-green to auto-capture.")
    parser.add_argument("--no-autocapture", action="store_true")
    parser.add_argument("--output", type=str, default="./captures")
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open video source.", file=sys.stderr)
        return 1

    score_fn = _build_score_fn(args.model_dir)
    bg = BackgroundScorer(score_fn)
    bg.start()

    fps_window: deque = deque(maxlen=30)
    autocapture_target = 0 if args.no_autocapture else args.autocapture_frames
    countdown = 0
    last_t = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            fps_window.append(now - last_t)
            last_t = now
            fps = (1.0 / np.mean(fps_window)) if fps_window else 0.0

            bg.submit(frame)
            snap = bg.latest()

            display = render_overlay(
                frame, snap, args.threshold, countdown,
                autocapture_target, fps,
            )

            if snap is not None and all_panels_green(snap.scores, args.threshold):
                countdown += 1
                if autocapture_target > 0 and countdown >= autocapture_target:
                    img_path = save_capture(frame, snap.scores, output_dir)
                    print(f"[auto-capture] {img_path}")
                    countdown = 0
            else:
                countdown = 0

            cv2.imshow("ofiq-syngen capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if snap is not None:
                    img_path = save_capture(frame, snap.scores, output_dir)
                    print(f"[manual capture] {img_path}")
            if key == ord("r"):
                countdown = 0

    finally:
        bg.stop()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
