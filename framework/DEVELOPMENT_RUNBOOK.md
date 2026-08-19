# Development execution runbook

This runbook executes only the development phase. Do not inspect or generate scientific results from the test split yet.

## 1. Prepare the environment

Use Python 3.11 or 3.12 and, preferably, an NVIDIA GPU with enough memory for torchvision ResNet-50 and ViT-B/16 inference.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[torch]'
```

Freeze the resolved environment after it works:

```bash
python -m pip freeze > artifacts/environment.txt
```

## 2. Prepare ImageNet validation

Obtain ILSVRC2012 validation images and ground-truth labels through an authorized ImageNet source. Do not redistribute the dataset.

Supported layouts:

- `imagefolder`: 1,000 class directories, each containing its validation images.
- `flat`: 50,000 validation images plus the official ground-truth text file, one 1-indexed label per image in sorted filename order.

Set paths:

```bash
export EDITTRACE_ROOT="$PWD"
export IMAGENET_VAL="/absolute/path/to/ILSVRC2012_img_val"
export IMAGENET_LABELS="/absolute/path/to/ILSVRC2012_validation_ground_truth.txt"
mkdir -p artifacts/manifests artifacts/traces artifacts/logs
```

## 3. Run preflight

For a flat validation directory:

```bash
edittrace preflight \
  --config configs/development.yaml \
  --data-root "$IMAGENET_VAL" \
  --layout flat \
  --labels "$IMAGENET_LABELS" \
  | tee artifacts/logs/preflight.json
```

The command must finish with `"passed": true`. Resolve every blocker before continuing.

## 4. Build the raw manifest

Flat layout:

```bash
edittrace build-manifest \
  --root "$IMAGENET_VAL" \
  --layout flat \
  --labels "$IMAGENET_LABELS" \
  --output artifacts/manifests/imagenet_val.jsonl
```

ImageFolder layout:

```bash
edittrace build-manifest \
  --root "$IMAGENET_VAL" \
  --layout imagefolder \
  --output artifacts/manifests/imagenet_val.jsonl
```

Expected result: exactly 50,000 records with SHA-256 hashes and labels 0–999.

## 5. Select model-specific eligible images

Ensure the torchvision weights are already cached or that the environment can download them once. Record the exact weight identifiers printed in the `.meta.json` sidecars.

```bash
edittrace select-eligible \
  --manifest artifacts/manifests/imagenet_val.jsonl \
  --config configs/development.yaml \
  --model resnet50 \
  --device cuda \
  --output artifacts/manifests/resnet50_eligible.jsonl \
  | tee artifacts/logs/resnet50_eligibility.json

edittrace select-eligible \
  --manifest artifacts/manifests/imagenet_val.jsonl \
  --config configs/development.yaml \
  --model vit_b_16 \
  --device cuda \
  --output artifacts/manifests/vit_b_16_eligible.jsonl \
  | tee artifacts/logs/vit_b_16_eligibility.json
```

Each file must contain 500 development and 2,000 test records.

## 6. Audit split isolation

```bash
edittrace audit-eligible \
  --config configs/development.yaml \
  --manifests \
    artifacts/manifests/resnet50_eligible.jsonl \
    artifacts/manifests/vit_b_16_eligible.jsonl \
  | tee artifacts/logs/eligible_audit.json
```

The command must report `"passed": true` and an empty `cross_split_leakage` list.

## 7. Generate B-Natural on development only

```bash
for MODEL in resnet50 vit_b_16; do
  edittrace generate-natural \
    --manifest "artifacts/manifests/${MODEL}_eligible.jsonl" \
    --config configs/development.yaml \
    --split development \
    --count 500 \
    --output "artifacts/traces/${MODEL}_b_natural_dev.jsonl"
done
```

Check that history lengths are balanced and that every operation appears across positions.

## 8. Generate controlled benchmarks

Run one model at a time because these generators can require many model evaluations.

```bash
for MODEL in resnet50 vit_b_16; do
  edittrace generate-controlled \
    --manifest "artifacts/manifests/${MODEL}_eligible.jsonl" \
    --config configs/development.yaml \
    --benchmark B-Single \
    --model "$MODEL" \
    --device cuda \
    --split development \
    --count 500 \
    --output "artifacts/traces/${MODEL}_b_single_dev_outcomes.jsonl"

  edittrace generate-controlled \
    --manifest "artifacts/manifests/${MODEL}_eligible.jsonl" \
    --config configs/development.yaml \
    --benchmark B-Interaction \
    --model "$MODEL" \
    --device cuda \
    --split development \
    --count 500 \
    --output "artifacts/traces/${MODEL}_b_interaction_dev_outcomes.jsonl"
done
```

Do not discard rejected outcomes. Record acceptance yield and rejection reasons separately for each model.

## 9. Create B-Oracle

```bash
for MODEL in resnet50 vit_b_16; do
  edittrace generate-oracle \
    --input "artifacts/traces/${MODEL}_b_natural_dev.jsonl" \
    --count 500 \
    --output "artifacts/traces/${MODEL}_b_oracle_dev.jsonl"
done
```

This step selects the stable subset. Exhaustive repair evaluation is a later component.

## 10. Development-only review

Before changing the protocol, report:

- eligible-image counts and evaluated-image counts;
- B-Single and B-Interaction acceptance yield;
- rejection reasons and bounded-attempt exhaustion;
- replay-invalidity rate by operation, action position, and history length;
- operation/parameter/position coverage;
- visible saturation or codec artifacts;
- median and p95 generation time by model and benchmark;
- GPU/CPU memory use and exact package versions.

Any parameter-grid or generator amendment must be based only on these development outputs and entered in the protocol amendment log. Regenerate invalidated development artifacts after the amendment. Keep the test split locked.

