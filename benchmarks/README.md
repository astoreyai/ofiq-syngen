# Benchmarks

Standalone benchmark harnesses for `ofiq-syngen`. None of these are
run by CI; invoke manually when you want to characterize the package
or compare it to alternatives.

## Throughput

`bench_throughput.py` measures wall-clock per degradation, per
component, on a synthetic input.

```bash
python benchmarks/bench_throughput.py --iterations 100 --output benchmarks/results/throughput.csv
```

The synthetic input is uniform-noise; real face images on real OFIQ
context add the model-inference cost (~100-300 ms/image) and dwarf
the per-degrader cost. Use this benchmark to characterize the bare
degrader cost; use `examples/realtime_capture.py` to characterize the
end-to-end OFIQ-loop cost.

## Best-in-class grid

`run_grid.py` runs `ofiq-syngen` plus every adapter in `adapters/`
across (component, image, metric) cells. Outputs a parquet
(or JSON if pandas is unavailable).

```bash
python benchmarks/run_grid.py --image-dir /path/to/face_images --max-images 10
```

### Adapters

`adapters/` holds thin wrappers around competitor methods. Each
adapter exposes:

```python
name: str
def degrade(image, component, severity, seed) -> image
```

Adapters lazy-load their dependencies. A missing dependency just skips
that adapter; the grid still runs over the rest.

Currently shipped:

- `_template.py` -- copy this to start a new adapter
- `opencv_baseline.py` -- OpenCV-only minimal perturbations (sanity
  baseline: any method losing to this is doing something wrong)

To add a serious competitor (GFPGAN, Real-ESRGAN, BSRGAN, NatOcc,
DSL-FIQA, MR-FIQA, etc.):

1. Copy `_template.py` to `adapters/<your_method>.py`.
2. Implement `degrade` with lazy imports of the heavy deps.
3. Document in `COMPETITORS.md` under the relevant component.
4. Re-run the grid; results land in `results/grid.parquet`.

## Component coverage / status

`COMPETITORS.md` -- per-component catalog of alternatives, license,
cost, determinism, with placeholder for the head-to-head verdict.
