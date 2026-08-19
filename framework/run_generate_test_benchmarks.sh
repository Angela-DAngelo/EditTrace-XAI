#!/usr/bin/env bash

set -Eeuo pipefail

###############################################################################
# Configuration
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export IMAGE_ROOT="$HOME/EditTrace/data/imagenet"
export DEVICE="cpu"

mkdir -p artifacts/logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="artifacts/logs/run_generate_test_benchmarks_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE")
exec 2>&1

echo "=========================================================="
echo "EditTrace-XAI - TEST BENCHMARK GENERATION"
echo "Started: $(date)"
echo "=========================================================="

###############################################################################
# ResNet B-Single
###############################################################################

echo
echo "===== ResNet50 B-Single ====="
python -m edittrace_xai.cli generate-controlled --manifest artifacts/manifests/resnet50_eligible.jsonl --config configs/development.yaml --benchmark B-Single --model resnet50 --device cpu --split test --count 2000 --output artifacts/traces/resnet50_B-Single_test_outcomes.jsonl

###############################################################################
# ViT B-Single
###############################################################################

echo
echo "===== ViT-B16 B-Single ====="
python -m edittrace_xai.cli generate-controlled --manifest artifacts/manifests/vit_b_16_eligible.jsonl --config configs/development.yaml --benchmark B-Single --model vit_b_16 --device cpu --split test --count 2000 --output artifacts/traces/vit_b_16_B-Single_test_outcomes.jsonl

###############################################################################
# ResNet B-Interaction
###############################################################################

echo
echo "===== ResNet50 B-Interaction ====="
python -m edittrace_xai.cli generate-controlled --manifest artifacts/manifests/resnet50_eligible.jsonl --config configs/development.yaml --benchmark B-Interaction --model resnet50 --device cpu --split test --count 2000 --output artifacts/traces/resnet50_B-Interaction_test_outcomes.jsonl

###############################################################################
# ViT B-Interaction
###############################################################################

echo
echo "===== ViT-B16 B-Interaction ====="
python -m edittrace_xai.cli generate-controlled --manifest artifacts/manifests/vit_b_16_eligible.jsonl --config configs/development.yaml --benchmark B-Interaction --model vit_b_16 --device cpu --split test --count 2000 --output artifacts/traces/vit_b_16_B-Interaction_test_outcomes.jsonl

###############################################################################
# ResNet Oracle
###############################################################################

echo
echo "===== ResNet50 B-Oracle ====="
python -m edittrace_xai.cli generate-oracle --input artifacts/traces/resnet50_b_natural_test.jsonl --count 2000 --output artifacts/traces/resnet50_b_oracle_test.jsonl

###############################################################################
# ViT Oracle
###############################################################################

echo
echo "===== ViT-B16 B-Oracle ====="
python -m edittrace_xai.cli generate-oracle --input artifacts/traces/vit_b_16_b_natural_test.jsonl --count 2000 --output artifacts/traces/vit_b_16_b_oracle_test.jsonl

###############################################################################
# ResNet Oracle shifted50
###############################################################################

echo
echo "===== ResNet50 Oracle shifted50 ====="
python scripts/select_shifted_oracle_subset.py --input artifacts/traces/resnet50_b_oracle_test.jsonl --model resnet50 --image-root "$IMAGE_ROOT" --device cpu --count 50 --output artifacts/traces/resnet50_b_oracle_shifted50_test.jsonl

###############################################################################
# ViT Oracle shifted50
###############################################################################

echo
echo "===== ViT-B16 Oracle shifted50 ====="
python scripts/select_shifted_oracle_subset.py --input artifacts/traces/vit_b_16_b_oracle_test.jsonl --model vit_b_16 --image-root "$IMAGE_ROOT" --device cpu --count 50 --output artifacts/traces/vit_b_16_b_oracle_shifted50_test.jsonl

###############################################################################
# END
###############################################################################

echo
echo "=========================================================="
echo "TEST BENCHMARK GENERATION COMPLETED"
echo "Finished: $(date)"
echo "Log: $LOGFILE"
echo "=========================================================="
