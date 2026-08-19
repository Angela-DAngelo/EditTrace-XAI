# Frozen Evaluation Definitions

These definitions are implemented by the v0.3 evaluators and should be copied consistently into the experimental protocol and manuscript.

## Shared target scalar

For ground-truth target class `y`, the scalar is the logit margin

`m(x, y) = z_y(x) - max_{c != y} z_c(x)`.

The target is recovered exactly when `m(x, y) > 0`. ResNet-50 uses torchvision `IMAGENET1K_V2`; ViT-B/16 uses `IMAGENET1K_V1`.

## RQ1

For final history output `x_T` and the valid replay `x_T^{-i}` obtained by removing action `i`, the proposed diagnostic score is

`e_i = m(x_T^{-i}, y) - m(x_T, y)`.

Actions are ranked by decreasing `e_i`. The B-Single designated trigger is fixed by the generator and is independent of this score.

Baselines:

- `last_action`: the final history action ranks first.
- `adjacent_score_change`: `m(x_{i-1}, y) - m(x_i, y)`.
- `edit_magnitude`: normalized parameter distance from the operation identity.
- `random`: seeded independently per trace with seed 20260802.
- `integrated_gradients_support`: absolute target-margin IG aggregated over the registered edit support, using 32 integration steps for the final run.

Parameter magnitude normalization uses the frozen development grid: factor distance from 1 divided by the largest available factor distance; blur sigma divided by maximum sigma; removed crop area divided by the largest available removed area; and JPEG distance from quality 100 divided by the largest available distance.

Primary metrics are designated-trigger top-1 accuracy and MRR. All methods use deterministic action-index tie-breaking.

## RQ2

For a pair `(i,j)`, the interaction score is

`I_ij = m(x_T^{-ij}, y) - m(x_T^{-i}, y) - m(x_T^{-j}, y) + m(x_T, y)`.

The independent-action ablation uses

`A_ij = [m(x_T^{-i}, y) - m(x_T, y)] + [m(x_T^{-j}, y) - m(x_T, y)]`.

All pairs are enumerated. A selected pair achieves pair-only recovery when both single-removal margins are non-positive and the joint-removal margin is positive. Primary metrics are designated-pair top-1 accuracy and selected pair-only recovery; MRR and AP are secondary.

## RQ3

Only B-Natural traces satisfying positive source margin and non-positive final margin enter repair evaluation. The frozen hierarchy is:

1. Tier 1A: every valid single rollback.
2. Tier 1B: 20 non-zero attenuation steps per action toward its explicit identity, considered when Tier 1A has no feasible repair.
3. Tier 2: every valid pair rollback, considered when Tier 1 has no feasible repair.

Within the lowest feasible tier, selection minimizes LPIPS, then normalized parameter movement, then maximizes restored margin, and finally uses the deterministic candidate identifier. LPIPS is measured between the original final edited image and the repaired image. SSIM, action count, and parameter movement are secondary.

Baselines are rollback of the last action and rollback of the action with maximum normalized edit magnitude. DEV-001 exhaustive verification uses a separately frozen deterministic set of 50 shifted B-Oracle traces per model.

## RQ4

The evaluators record synchronized wall-clock latency and model-call counts. Model construction, weight loading, and warm-up are excluded. Diagnosis and interaction include replay plus inference. Repair latency assumes the source, final state, and shift status are already cached by the editor and measures candidate search, inference, and perceptual selection.

Report median and p95 separately by component, model, and history length. Record hardware, torch/torchvision versions, device, CUDA/cuDNN versions, warm-up count, and caching policy.

## Inference

Confirmatory estimates use 10,000-resample paired clustered bootstrap intervals over source images with seed 20260802. Models are reported separately. A cross-architecture claim requires the direction of the paired effect to agree and its interval to exclude zero for both architectures.
