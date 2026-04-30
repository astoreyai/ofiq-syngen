# Real-time capture example

`realtime_capture.py` demonstrates a live OFIQ-aligned face-capture loop with
panel-grouped feedback and auto-trigger when quality holds.

## What it shows

1. Streaming from a webcam or video file via OpenCV.
2. Background-thread OFIQ scoring through `gpu_ofiq_scorer.GPUOFIQScorer` so
   the UI stays responsive while inference runs at 100-300 ms per frame.
3. Component aggregation: 28 OFIQ components are grouped into 5 user-facing
   panels (LIGHTING, FOCUS, POSE, EXPRESSION, OCCLUSION). Each panel reports
   the worst-scoring component as actionable guidance.
4. Auto-capture: triggers after N consecutive scored frames where every panel
   is green (configurable via `--autocapture-frames`).
5. Manual capture: SPACE saves the current frame and per-component scores.

## Run

```bash
# Webcam (default)
python examples/realtime_capture.py

# Video file (for testing without a camera)
python examples/realtime_capture.py --video clip.mp4

# Stricter pass threshold
python examples/realtime_capture.py --threshold 80

# Manual-only mode (no auto-capture)
python examples/realtime_capture.py --no-autocapture

# Custom output directory
python examples/realtime_capture.py --output /tmp/captures

# Custom OFIQ model directory
python examples/realtime_capture.py --model-dir /opt/ofiq/models
```

## Output

Every capture writes two files to `--output`:

- `capture_YYYYMMDD_HHMMSS.png` -- the frame
- `capture_YYYYMMDD_HHMMSS.json` -- per-component OFIQ scores plus
  `ofiq_syngen` package version

## Keys

| Key | Action |
|---|---|
| SPACE | Manual capture |
| R | Reset auto-capture countdown |
| Q or ESC | Quit |

## Without OFIQ models

If `GPUOFIQScorer` cannot load (models missing, GPU unavailable), the example
falls back to a brightness-based mock scorer so the rest of the pipeline still
runs. A warning prints once on startup. Use `pip install ofiq-syngen[gpu]` and
download the OFIQ ONNX models to enable real scoring.

## Architecture (one screen)

```
+-----------------+   submit(frame)    +--------------------+
| main thread     | -----------------> | BackgroundScorer   |
|  cv2 read       |                    |  inbox: maxsize=1  |
|  render overlay |                    |  worker thread     |
|  cv2 imshow     | <----------------- |  publishes latest  |
|  key handling   |   latest()         |  ScoreSnapshot     |
+-----------------+                    +--------------------+
        |
        v
+-----------------+
| capture trigger |
|  all 5 panels   |
|  green for N    |
|  scored frames  |
+-----------------+
```

The inbox is bounded at 1 with drop-newest semantics so the scorer never
queues stale frames. The main thread always renders against the latest
published snapshot, which may be a few frames stale.

## Customizing

- **Panel groupings** live in the `PANELS` dict at the top of the file.
  Edit the component lists or guidance strings to fit your operational
  context (passport, DMV, mobile enrollment, etc.).
- **Pass threshold** is per-component. To enforce ICAO-strict capture, raise
  the threshold and prune the panels to only `ICAO_STRICT_COMPONENTS` from
  `ofiq_syngen.standards`.
- **Capture criteria** (`all_panels_green`) is the only branch that decides
  when to fire. Replace it with a weighted-score function or a
  per-component-threshold rule for stricter regimes.

## Caveats

- This is a demo. Production capture stations need pose-stability filtering,
  liveness checks, multi-frame fusion, and audit logging. None of that is
  here.
- OFIQ scoring throughput on CPU is roughly 0.3-1.0 fps. GPU inference via
  `onnxruntime-gpu` and the OFIQ models reaches 5-15 fps depending on
  hardware. The UI runs at full camera fps regardless; the score panel
  simply lags.
- The mock scorer is for plumbing tests only. Do not use its output for
  anything operational.
