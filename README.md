# jax-geneval

JAX/TPU implementation of the [GenEval](https://github.com/djghosh13/geneval)
image evaluator.

The goal is to keep GenEval's official scoring semantics while moving the
expensive model forwards to JAX:

- Mask2Former Swin-S detector in fixed-shape JAX, suitable for `jit` and
  data-parallel `pjit`.
- JAX CLIP color classification through a compatible `jclip` implementation.
- Host Python for the naturally dynamic pieces: image discovery, thresholding,
  class filtering, crop construction, NMS-style filtering, and JSONL output.
- Parity tests against the original PyTorch/mmdetection implementation.

## Status

The current port matches the official PyTorch/mmdetection path closely on a
553-image GenEval run:

```text
JAX/TPU score:      476/553 = 0.860759
reference score:    475/553 = 0.858951
task average:       0.861467
```

Only one image differed from the reference JSONL in that run. The detector
logit/mask sanity check against the real checkpoint passes with small numerical
error:

```text
class logits max diff: 5.7e-6
mask logits max diff:  1.3e-4
```

On a v4-8, after the first compile, the measured detector plus device-side
instance postprocess throughput was:

```text
batch size 4: ~4.24 images/s
batch size 8: ~6.30 images/s detector-only smoke
```

## Repository Layout

```text
src/jax_geneval/
  swin.py                 SwinTransformer / WindowMSA JAX port
  pixel_decoder.py        MSDeformAttn pixel decoder
  mask2former_head.py     Mask2Former transformer decoder/head
  detector.py             End-to-end detector forward
  instances.py            JAX top-k, mask scores, bboxes, binary masks
  evaluation.py           GenEval host-side scoring rules
  color.py                JAX CLIP color classifier adapter

scripts/
  run_jax_geneval.py              Main evaluator CLI
  summary_scores.py               Score summarizer/comparator
  check_mmdet_oracle.py           Random-weight mmdet parity checks
  check_full_detector_checkpoint.py Checkpoint parity check
  run_unit_tests.py               Lightweight tests
```

## Requirements

Core runtime:

- Python 3.10+
- JAX with CPU/GPU/TPU backend as appropriate
- NumPy
- Pillow
- PyTorch, used for loading `.pth` checkpoints and by the optional CLIP adapter

Optional but recommended:

- OpenCV (`cv2`) for preprocessing that better matches mmdetection resize
  behavior. The code falls back to Pillow if OpenCV is unavailable.
- A JAX CLIP package exposing `jclip.clip` for color tasks. You can either put
  it on `PYTHONPATH`, pass `--clip-repo`, or set `JAX_CLIP_REPO`.
- mmdetection 2.x and mmcv-full for parity scripts only. The JAX evaluator does
  not need mmdetection at runtime.

## Checkpoints

The detector checkpoint is the same one used by official GenEval:

```text
mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

Pass it explicitly:

```bash
python scripts/run_jax_geneval.py /path/to/geneval/images \
  --checkpoint /path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

Or set:

```bash
export JAX_GENEVAL_DETECTOR_CKPT=/path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

For CLIP color classification:

```bash
export JAX_CLIP_REPO=/path/to/legacy/jax-clip
```

See `.env.example` for all supported environment variables.

You may skip color classification for detector-only smoke tests with
`--skip-clip`.

## Quick Start

Run a small CPU smoke test:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_unit_tests.py
```

Evaluate a GenEval image directory:

```bash
python scripts/run_jax_geneval.py /path/to/geneval/images \
  --checkpoint /path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
  --outfile outputs/geneval_results.jsonl \
  --benchmark-out outputs/geneval_benchmark.json \
  --batch-size 4 \
  --input-height 800 \
  --input-width 800 \
  --output-height 512 \
  --output-width 512 \
  --compile pjit
```

Why `output-height/width=512`? Official GenEval images are usually 512x512.
The detector still receives the fixed 800x800 mmdetection input, but returning
512x512 masks avoids moving large query-mask tensors back to the host.

Summarize results:

```bash
python scripts/summary_scores.py outputs/geneval_results.jsonl
```

Compare two JSONL files:

```bash
python scripts/summary_scores.py outputs/geneval_results.jsonl \
  --compare /path/to/reference/geneval_results.jsonl
```

## TPU Notes

Use `--compile pjit` for explicit data-parallel execution. The batch size must
be divisible by the number of local JAX devices:

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

If you want JAX to choose automatically, use `--compile auto`; it selects `pjit`
when more than one local device is visible, otherwise `jit`.

The first batch includes XLA compilation. For speed reporting, use
`steady_images_per_s` in the benchmark JSON, which subtracts the first compiled
step from the inference total.

## Parity Checks

The lightweight tests do not require mmdetection:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_unit_tests.py
```

For PyTorch/mmdetection parity, provide a mmdetection 2.x checkout and the
checkpoint:

```bash
export MMDET_ROOT=/path/to/mmdetection
export JAX_GENEVAL_DETECTOR_CKPT=/path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth

JAX_PLATFORMS=cpu python scripts/check_mmdet_oracle.py
JAX_PLATFORMS=cpu python scripts/check_full_detector_checkpoint.py
```

If mmcv/mmdet live in a separate virtualenv site-packages directory, add:

```bash
export GENEVAL_VENV_SITE=/path/to/venv/lib/python3.10/site-packages
```

## Design Choices

- Model code is written as pure functions over pytrees rather than Flax modules.
  That keeps checkpoint conversion direct and makes pjit sharding explicit.
- Compiled model inputs and outputs are fixed shape.
- Instance top-k, mask scoring, bboxes, and binary masks run on device.
- GenEval's dynamic bookkeeping stays on the host, where it is simpler and much
  easier to audit.

## Current Limitations

- The detector port targets the GenEval Mask2Former Swin-S checkpoint.
- Color classification expects a `jclip`-compatible JAX CLIP implementation.
- The main CLI does not yet support resume-from-partial-output.
