# EditTrace-XAI Evaluation Runbook

This runbook evaluates the frozen development benchmarks for RQ1--RQ4. Development outputs are diagnostic only. Do not place them in the final paper as held-out evidence.

## 1. Environment

Run from the project root:

```bash
python -m pip install -e '.[torch,evaluation]'
python -m unittest discover -s tests -v
export IMAGE_ROOT=/home/dange/EditTrace/data/imagenet
export DEVICE=cuda
export EVAL_DIR=artifacts/evaluation/development
mkdir -p "$EVAL_DIR"
```

Use `DEVICE=cpu` when CUDA is unavailable. `--image-root` remaps the absolute paths stored during generation by appending each trace `source_id` to `IMAGE_ROOT`.

LPIPS may download the frozen AlexNet weights on first use. Complete that download before starting a timed run. Model construction, weight loading, and warm-up are excluded from RQ4 timings.

## 2. Smoke checks

Use small limits only to validate execution. These outputs are not scientific results.

```bash
python scripts/evaluate_rq1.py \
  --input artifacts/traces/resnet50_B-Single_dev_outcomes.jsonl \
  --config configs/development.yaml --model resnet50 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR/smoke" --limit 5 --include-ig --ig-steps 8 \
  --bootstrap-resamples 100

python scripts/evaluate_rq2.py \
  --input artifacts/traces/resnet50_B-Interaction_dev_outcomes.jsonl \
  --model resnet50 --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR/smoke" --limit 5 --bootstrap-resamples 100
```

Inspect every error record before a full run. A replay error is a blocking failure, not a removable observation.

## 3. RQ1: action diagnosis

The final development run includes the controlled replay method, last-action, adjacent-score, operation-normalized magnitude, seeded random, and Integrated Gradients support baselines.

```bash
python scripts/evaluate_rq1.py \
  --input artifacts/traces/resnet50_B-Single_dev_outcomes.jsonl \
  --config configs/development.yaml --model resnet50 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR" --include-ig --ig-steps 32

python scripts/evaluate_rq1.py \
  --input artifacts/traces/vit_b_16_B-Single_dev_outcomes.jsonl \
  --config configs/development.yaml --model vit_b_16 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR" --include-ig --ig-steps 32
```

Primary outputs are `top1` and `reciprocal_rank`; the summary contains 10,000-resample paired clustered-bootstrap differences. IG is aggregated over the registered edit support. Global operations share global support, so tied scores use deterministic action-index ordering.

## 4. RQ2: pair interactions

```bash
python scripts/evaluate_rq2.py \
  --input artifacts/traces/resnet50_B-Interaction_dev_outcomes.jsonl \
  --model resnet50 --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR"

python scripts/evaluate_rq2.py \
  --input artifacts/traces/vit_b_16_B-Interaction_dev_outcomes.jsonl \
  --model vit_b_16 --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --output-dir "$EVAL_DIR"
```

The proposed score is the frozen difference-in-differences. The ablation ranks pairs by the sum of their two single-action effects. Report designated-pair top-1, MRR/AP, and selected pair-only recovery.

## 5. Freeze the DEV-001 B-Oracle subsets

Run this before any repair evaluation and retain both the JSONL and its manifest/hash.

```bash
python scripts/select_shifted_oracle_subset.py \
  --input artifacts/traces/resnet50_b_oracle_dev.jsonl \
  --model resnet50 --image-root "$IMAGE_ROOT" --device "$DEVICE" --count 50 \
  --output artifacts/traces/resnet50_b_oracle_shifted50_dev.jsonl

python scripts/select_shifted_oracle_subset.py \
  --input artifacts/traces/vit_b_16_b_oracle_dev.jsonl \
  --model vit_b_16 --image-root "$IMAGE_ROOT" --device "$DEVICE" --count 50 \
  --output artifacts/traces/vit_b_16_b_oracle_shifted50_dev.jsonl
```

## 6. RQ3: constrained repair

Evaluate all shifted B-Natural traces:

```bash
python scripts/evaluate_rq3.py \
  --input artifacts/traces/resnet50_b_natural_dev.jsonl \
  --config configs/development.yaml --model resnet50 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --perceptual-metric lpips --output-dir "$EVAL_DIR"

python scripts/evaluate_rq3.py \
  --input artifacts/traces/vit_b_16_b_natural_dev.jsonl \
  --config configs/development.yaml --model vit_b_16 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --perceptual-metric lpips --output-dir "$EVAL_DIR"
```

Then verify the enumerated repair objective on the frozen subsets:

```bash
python scripts/evaluate_rq3.py \
  --input artifacts/traces/resnet50_b_oracle_shifted50_dev.jsonl \
  --config configs/development.yaml --model resnet50 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --perceptual-metric lpips --exhaustive-verification --oracle-limit 50 \
  --output-dir "$EVAL_DIR"

python scripts/evaluate_rq3.py \
  --input artifacts/traces/vit_b_16_b_oracle_shifted50_dev.jsonl \
  --config configs/development.yaml --model vit_b_16 \
  --image-root "$IMAGE_ROOT" --device "$DEVICE" \
  --perceptual-metric lpips --exhaustive-verification --oracle-limit 50 \
  --output-dir "$EVAL_DIR"
```

`--perceptual-metric l1` is allowed only for smoke tests. Paper results use LPIPS; SSIM is emitted when scikit-image is installed.

## 7. RQ4: latency

RQ4 aggregates timings already recorded during RQ1--RQ3, avoiding a second measurement pass:

```bash
python scripts/evaluate_rq4.py \
  --inputs \
    "$EVAL_DIR"/rq1_*_per_trace.jsonl \
    "$EVAL_DIR"/rq2_*_per_trace.jsonl \
    "$EVAL_DIR"/rq3_*_natural_per_trace.jsonl \
  --output-dir "$EVAL_DIR"
```

Report component-level median and p95 by model and history length, model-call counts, hardware, warm-up, caching policy, and synchronization behavior. Do not describe the measured latency as human usability evidence.

## 8. Paper tables

```bash
python scripts/build_paper_tables.py \
  --summaries \
    "$EVAL_DIR"/rq1_*_summary.json \
    "$EVAL_DIR"/rq2_*_summary.json \
    "$EVAL_DIR"/rq3_*_natural_summary.json \
  --rq4 "$EVAL_DIR"/rq4_latency.json \
  --output-dir "$EVAL_DIR/paper_tables"
```

The generated LaTeX explicitly labels development results. Regenerate the tables from the immutable held-out snapshot for the manuscript.

## 9. Test unlock gate

Do not unlock the held-out split until all of the following hold:

- no unexplained replay errors;
- RQ1--RQ4 commands finish end to end;
- baseline definitions and tie-breaking are frozen;
- IG steps, LPIPS implementation, warm-up, and device are frozen;
- bootstrap seed/resamples are unchanged;
- DEV-001 oracle manifests and hashes are archived;
- development failure analysis is complete;
- the evaluation code and configuration are saved as an immutable snapshot.
