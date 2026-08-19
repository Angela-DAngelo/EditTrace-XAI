# EditTrace-XAI

This package implements the frozen v1.0 action schema, deterministic replay engine, benchmark generators, and RQ1--RQ4 evaluation pipeline for EditTrace-XAI.

## Implemented

- Versioned Pydantic schemas for images, actions, traces, manifests, and generator outcomes.
- Deterministic brightness, contrast, saturation, Gaussian blur, crop-resize, and JPEG operations.
- Exact replay, action removal, parameter attenuation, typed invalid-replay reasons, lineage hashes, and deterministic trace construction.
- `B-Natural`, `B-Single`, `B-Interaction`, and `B-Oracle` development generators.
- ImageFolder-style manifest construction and a lazy optional torchvision predictor.
- Standard-library unit tests and a synthetic smoke pipeline.
- Preflight, flat/ImageFolder ImageNet manifests, and cross-model split-isolation audit.
- RQ1 action diagnosis with controlled trigger labels, proxy baselines, optional Integrated Gradients, and clustered bootstrap intervals.
- RQ2 pairwise difference-in-differences with an independent-action ablation.
- RQ3 hierarchical constrained repair, naive rollback baselines, LPIPS/SSIM, and a deterministic shifted B-Oracle subset.
- RQ4 latency/model-call aggregation and paper-ready CSV/LaTeX tables.

Benchmark generation is documented in `DEVELOPMENT_RUNBOOK.md`; evaluation is documented in `EVALUATION_RUNBOOK.md`.

The synthetic fixture validates software behavior only. It is not paper evidence and must not be mixed with the locked ImageNet evaluation.

## Install

```bash
python -m pip install -e .
```

In a network-restricted environment with the build tools already installed, use `python -m pip install -e . --no-build-isolation`.

For ImageNet eligibility and model-dependent generators:

```bash
python -m pip install -e '.[torch]'
```

For the paper evaluation, including LPIPS and SSIM:

```bash
python -m pip install -e '.[torch,evaluation]'
```

## Verify

```bash
python -m unittest discover -s tests -v
python -m edittrace_xai.cli smoke --output-dir artifacts/smoke
```

## Development data

Create a manifest from an ImageFolder-style tree (`root/class_name/image.JPEG`):

```bash
python -m edittrace_xai.cli build-manifest \
  --root /path/to/imagenet/val \
  --output artifacts/manifests/imagenet_val.jsonl
```

Generate model-independent natural traces:

```bash
python -m edittrace_xai.cli generate-natural \
  --manifest artifacts/manifests/resnet50_eligible.jsonl \
  --config configs/development.yaml \
  --output artifacts/traces/b_natural.jsonl \
  --split development \
  --count 500
```

Select the frozen 500-image development and 2,000-image test manifests for each model (requires the optional torch dependencies):

```bash
python -m edittrace_xai.cli select-eligible \
  --manifest artifacts/manifests/imagenet_val.jsonl \
  --config configs/development.yaml \
  --model resnet50 \
  --device cuda \
  --output artifacts/manifests/resnet50_eligible.jsonl
```

The first 5,000 records after the global seeded shuffle form the development candidate pool. The remaining records form the test candidate pool, so pool membership is identical across models and cannot leak between development and test.

Generate a controlled benchmark while retaining accepted and rejected outcomes:

```bash
python -m edittrace_xai.cli generate-controlled \
  --manifest artifacts/manifests/resnet50_eligible.jsonl \
  --config configs/development.yaml \
  --benchmark B-Single \
  --model resnet50 \
  --device cuda \
  --split development \
  --output artifacts/traces/resnet50_b_single_outcomes.jsonl
```

Create the stable oracle subset from natural traces:

```bash
python -m edittrace_xai.cli generate-oracle \
  --input artifacts/traces/b_natural.jsonl \
  --output artifacts/traces/b_oracle.jsonl \
  --count 500
```

`B-Single` and `B-Interaction` require a predictor because their acceptance rules use the frozen target logit margin. Their Python APIs return accepted traces plus complete attempt/rejection accounting.

## Test unlock

Do not run the test manifest until operation versions, development amendments, model weights/preprocessing, all baseline configurations, and analysis code have been frozen and hashed.
