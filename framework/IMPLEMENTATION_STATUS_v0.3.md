# EditTrace-XAI implementation status v0.3

Date: 2026-08-02.

## Added

- Shared evaluation utilities for benchmark loading, ImageNet path remapping, deterministic ranking, paired clustered bootstrap, timing, and perceptual metrics.
- RQ1 evaluator with counterfactual replay, last-action, adjacent-score, operation-normalized magnitude, seeded-random, and optional Integrated Gradients support baselines.
- RQ2 evaluator with pairwise difference-in-differences, independent-action ablation, designated-pair accuracy, AP/MRR, and pair-only recovery.
- RQ3 evaluator with the frozen hierarchical repair space, naive rollback baselines, LPIPS/SSIM, and bounded exhaustive verification.
- Separate deterministic selection and hashing of the DEV-001 50-shifted-trace B-Oracle subset.
- RQ4 aggregation of median/p95 component latency and model-call counts by model and history length.
- CSV, JSONL, JSON, Markdown, and LaTeX paper-table outputs.
- Exact frozen torchvision weight identifiers enforced during evaluation.
- Evaluation runbook and six focused evaluation tests.

## Verification

- Editable package installation succeeds.
- All 22 unit tests pass.
- Every evaluation entry point compiles and its CLI help path executes.
- Synthetic tests verify RQ1 trigger ranking, RQ2 designated pair ranking/pair-only recovery, RQ3 single-rollback recovery, bootstrap pairing, magnitude bounds, and RQ4 primary-method aggregation.

## Not executed in this workspace

- Real ImageNet evaluation, because the image files are not mounted.
- Torchvision inference, because torch and torchvision are not installed here.
- LPIPS and SSIM, because their optional dependencies are not installed here.

The development benchmarks remain included in the repository. Run the commands in `EVALUATION_RUNBOOK.md` on the original ImageNet environment before unlocking the held-out test split.
