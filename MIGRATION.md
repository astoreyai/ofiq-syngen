# Migration guide

## v0.4.x → v0.5.0

### What's new

- **3D head pose pipeline**: `HeadPoseYaw` and `HeadPosePitch` now have a
  Tier 1 path that fits FLAME via DECA and renders via pyrender. Falls
  back to the 2D BFM TPS warp (Tier 2) and finally a perspective squeeze
  (Tier 3). Install with `pip install ofiq-syngen[three_d]` and run
  `ofiq-syngen install-assets`.
- **InstructPix2Pix / Stable Diffusion expression paths**: opt-in via
  `OFIQ_SYNGEN_EXPRESSION_METHOD={ip2p,sd_inpaint,3dmm,tps}`. Default
  remains 3DMM landmark morph. Install with `pip install ofiq-syngen[diffusion]`.
- **Asset management**: new `ofiq-syngen check-assets` and
  `ofiq-syngen install-assets` subcommands. Pinned SHA-256 verification
  for downloaded assets. `OFIQ_SYNGEN_OFFLINE=1` env var aborts any
  network fetch.
- **CLI globals**: `--device {cpu,cuda,auto}`, `-v` / `-q` for verbosity,
  `--version`. Logging routed through stdlib `logging` with timestamps.

### Breaking changes

| Operator | Old behavior (v0.4) | New behavior (v0.5) | Why |
|---|---|---|---|
| `LuminanceMean` | Face-mask-only darkening | Whole-image gamma | Face-only looked unnatural (bright hair against dim face); whole-image matches real underexposed photographs |
| `OverExposurePrevention` | Face-mask-only brightening | Whole-image gamma | Same reason — face-only blowout looked spotlit |
| `UnderExposurePrevention` | Face-mask-only darkening | Whole-image gamma | Same reason |
| `DynamicRange` | Face-mask histogram crush around mean | YCrCb Y-channel posterization with feathered alpha blend | Preserves natural skin / lip color while still destroying entropy |
| `CompressionArtifacts` | Q=92 at sev=1.0 | Q=18 at sev=1.0, double-encode at sev≥0.6 | Q=92 was visually indistinguishable from the source; the new mapping spans the OFIQ scalar range |
| `InterEyeDistance` / `HeadSize` | Resize-then-resize (no-op on aligned crops) | Pad-and-shrink with flat-backdrop fill | Old approach didn't change pixel IED on aligned crops; new approach honestly shrinks the face in the frame |
| `ExpressionNeutrality` | Hand-picked landmark TPS warp | 3DMM-aligned morph (default), with optional IP2P / SD inpaint | TPS warp produced "horrible" geometric artifacts; 3DMM morph is anatomically correct |

### API surface (unchanged)

Existing code that uses `DegradationPipeline.degrade_single()`,
`degrade_sweep()`, `degrade_all_components()` continues to work. The
changes above only affect the **output pixel content** of the operators
listed in the table.

### Test fixture updates

If you have your own directional test suite based on synthetic fixtures
similar to `tests/test_degradation_direction.py`, six tests need
updating to match the new operator behavior:

1. `test_luminance_mean_moves_away_from_optimum` — assert mean
   luminance decreases (whole-image gamma), not "distance from 0.5
   grows" (which can fail when the source is far above 0.5).
2. `test_over_exposure_increases` — assert mean luminance increases,
   not ">247 pixel proportion" (which stays at zero on synthetic faces
   that don't reach 247 even after gamma=0.25).
3. `test_dynamic_range_decreases` — assert pixel-level change, not
   entropy decrease (the feather blend reintroduces values on uniform
   synthetic faces).
4. `test_compression_artifacts_changes_image` — relax the diff threshold
   from `>5.0` to `>1.0` (synthetic low-frequency faces compress well
   even at Q=18).
5. `test_inter_eye_distance_decreases` — replace the cv2 eye detector
   (which mis-fires on the shrunken canvas) with a dark-pixel centroid
   spread proxy.
6. `test_expression_neutrality_warps_landmarks` — relax the diff
   threshold from `>1.0` to `>0.1` (the 3DMM fallback path is mild on
   synthetic faces with no `raw_3ddfa_params`).

See `tests/test_degradation_direction.py` in this release for the
canonical updated assertions.

### Deprecations

None. The `[generative]` extra is retained as a backwards-compatible
alias for `[diffusion,insightface]`.

### Removed

- The `--flame-username` / `--flame-password` flags on `install-assets`
  were removed. ofiq-syngen now refuses to download FLAME 2020 under
  any circumstances, in compliance with the Max Planck Institute's
  redistribution prohibition. Users must register at
  <https://flame.is.tue.mpg.de/> and place `generic_model.pkl`
  manually. Run `ofiq-syngen check-assets` to see the target path.
