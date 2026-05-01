# Gallery

Per-component visual demonstration of every degrader at five severity
levels (0.0, 0.25, 0.5, 0.75, 1.0) on a canonical face image.

The gallery images themselves are not bundled with the package because
canonical face images carry license restrictions. To populate the
gallery locally:

## Choose a canonical face

License-clean options for **redistribution**:

- **FFHQ samples** (Creative Commons BY 2.0): pick a 1024x1024 face from
  https://github.com/NVlabs/ffhq-dataset.
- **StyleGAN-generated synthetic face** (no real subject):
  https://thispersondoesnotexist.com produces a fresh sample per refresh.
- **Generated/placeholder**: any face you control the rights to.

For **local development only** (do NOT commit the image):

- **CelebA** (research-only): if you have it downloaded, use any image
  from `img_align_celeba/` as your canonical face. The package's parity
  fixtures (`tests/fixtures/ofiq_parity/`) reference CelebA images
  `000001`, `000010`, `000100` by sha256 — the manifest with expected
  OFIQ scores ships with the package; the images themselves are
  gitignored and must be downloaded separately.

Avoid committing CelebA, CASIA, MS-Celeb-1M, and VGGFace2 images to
public repositories — those datasets restrict redistribution.

## Regenerate

```bash
python scripts/regenerate_gallery.py --face docs/gallery/canonical.png
```

Optional: limit to a single component for quick iteration.

```bash
python scripts/regenerate_gallery.py \
    --face docs/gallery/canonical.png \
    --components Sharpness,CompressionArtifacts
```

Outputs:

- `docs/gallery/images/<Component>_strip.png` -- 1x5 severity strip per component
- `docs/gallery/<Component>.md` -- per-component page with strip embedded plus standards block
- `docs/gallery/INDEX.md` -- navigation index

## Per-component caption convention

Each `<Component>.md` page is auto-generated from `STANDARDS_REFS` plus
the registered degrader descriptions. The "Notes" section is a placeholder
to be filled in by hand with:

1. **Where the perturbation works**: severity range over which the output
   is visually convincing.
2. **Where it fails**: severity at which artifacts become apparent.
3. **Better-if-you-can alternative**: any higher-fidelity method the
   project chose not to use, with the trade-off (e.g., GPU dependency,
   non-determinism, model size).

These notes feed directly into the JOSS paper's limitations section and
the per-component theory pages in `docs/theory/`.

## Why the gallery is not auto-published with the package

The gallery is documentation, not code. Running it requires a license-
clean face image you (the user) supply. The `scripts/regenerate_gallery.py`
script ensures consistent output formatting; the inputs and the rendered
outputs are local to each fork.

The CI workflow does not run the gallery regeneration. Add it manually
to your release process if you want gallery refreshes per version.
