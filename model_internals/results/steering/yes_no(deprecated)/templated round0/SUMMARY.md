# RepE Steering Investigation: Economic Uncertainty in Llama 3.1-8B-Instruct

## Objective

This investigation tests whether Llama 3.1-8B-Instruct internally represents **economic uncertainty** as a linear direction in its hidden states, and whether we can **steer** the model's predictions by intervening on that direction at inference time. We follow the Representation Engineering (RepE) methodology (Zou et al., arXiv 2310.01405).

## Background

Prior activation patching experiments (`activation_patching.py`) identified **layer 12** as the layer where patching the residual stream at the final period token produces the largest logit difference between uncertain and certain economic statements. This made layer 12 a natural intervention target for steering.

## Method

### Dataset
- 200 contrastive pairs of synthetic economic statements from `econ_uncertainty_contrastive_flat.json`.
- Each pair shares the same economic topic but has opposite uncertainty labels (high vs. no uncertainty).
- Pair 1 was reserved as the fixed 2-shot demonstration; the remaining 199 pairs were split 80/20 into 160 train and 39 test pairs.

### Stage 1: Direction Extraction (Mass Mean Difference)
For each training pair, we extracted layer-12 residual stream activations at the last period token using `nnsight`, then computed:

```
d_i = h(uncertain_i) - h(certain_i)
direction = mean(d_i) / ||mean(d_i)||
```

This yields a unit-length 4096-dimensional steering direction that separates uncertain from certain representations.

### Stage 2: Validation on Held-Out Data
We projected test-set activations onto the direction and classified by sign (positive = uncertain, negative = certain).

### Stage 3: Steering During Inference
For each alpha in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], we added `alpha * direction` to the layer-12 residual stream at the period token position during the forward pass, then compared logits for YES vs. NO to determine the model's classification.

### Stage 4: Visualization
Generated plots of accuracy and logit difference as a function of steering strength.

## Key Results

### Direction Validation (Stage 2)
| Metric | Value |
|--------|-------|
| **Overall accuracy** | **100%** (78/78 statements) |
| Uncertain accuracy | 100% |
| Certain accuracy | 100% |
| Uncertain mean projection | +2.57 |
| Certain mean projection | -1.53 |

The projection histogram shows **complete separation** between certain (all negative) and uncertain (all positive) activations, with a wide margin around the decision boundary at zero. This confirms that layer 12 encodes economic uncertainty as a clean linear direction.

### Steering Results (Stage 3)

| Alpha | Overall Acc | Uncertain Acc | Certain Acc | Mean Logit Diff (YES-NO) |
|------:|:-----------:|:-------------:|:-----------:|:------------------------:|
| -5.0  | 1.000 | 1.000 | 1.000 | 0.665 |
| -4.0  | 1.000 | 1.000 | 1.000 | 0.901 |
| -3.0  | 1.000 | 1.000 | 1.000 | 1.724 |
| -2.0  | 1.000 | 1.000 | 1.000 | 2.718 |
| -1.0  | 0.987 | 1.000 | 0.974 | 3.397 |
|  0.0  | 0.923 | 1.000 | 0.846 | 3.894 |
|  1.0  | 0.897 | 1.000 | 0.795 | 4.439 |
|  2.0  | 0.718 | 1.000 | 0.436 | 5.200 |
|  3.0  | 0.603 | 1.000 | 0.205 | 6.296 |
|  4.0  | 0.513 | 1.000 | 0.026 | 7.333 |
|  5.0  | 0.500 | 1.000 | 0.000 | 7.889 |

### Interpretation

1. **The steering direction is causally meaningful.** Adding the uncertainty direction (positive alpha) monotonically increases the logit difference toward YES (uncertain), while subtracting it (negative alpha) decreases it. The mean logit diff curve is monotonically increasing across the full alpha range.

2. **Negative alpha (toward certainty) improves accuracy.** The unsteered model (alpha=0) already has a YES bias — it classifies all uncertain statements correctly but misclassifies ~15% of certain statements as uncertain. Steering toward certainty (alpha=-2 to -5) corrects this bias, achieving perfect 100% accuracy.

3. **Positive alpha (toward uncertainty) degrades certain-class accuracy.** As alpha increases, the model increasingly predicts YES for everything. At alpha=5, certain accuracy drops to 0% — every statement is classified as uncertain.

4. **The model has an inherent uncertainty bias.** Even at alpha=0, the mean logit diff is +3.89, meaning the model leans toward "YES" (uncertain) by default. This explains why negative steering helps: it counteracts the model's existing bias.

5. **Asymmetric steering effect.** Uncertain accuracy stays at 100% across all alphas, while certain accuracy is the only thing that moves. This suggests the uncertainty direction primarily modulates the model's confidence about *certain* statements rather than affecting uncertain ones.

## Output Files

| File | Description |
|------|-------------|
| `steering.py` | Main pipeline script (Stages 1-4) |
| `steering_direction.npy` | Unit-length direction vector (4096-d) |
| `steering_split.json` | Train/test pair ID split |
| `train_activations.npz` | Cached train-set activations |
| `test_activations.npz` | Cached test-set activations |
| `steering_validation.json` | Held-out validation metrics |
| `steering_projection_histogram.png` | Projection histogram (certain vs uncertain) |
| `steering_results.json` | Per-alpha, per-sample evaluation results |
| `steering_accuracy_vs_alpha.png` | Overall accuracy vs. alpha |
| `steering_logitdiff_vs_alpha.png` | Mean logit diff vs. alpha |
| `steering_perclass_vs_alpha.png` | Per-class accuracy vs. alpha |
| `run.log` | Full pipeline execution log |
| `SUMMARY.md` | This file |

## Configuration

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Layer**: 12
- **Method**: Mass mean difference (paired diffs averaged)
- **Steer mode**: Period-token only
- **Prompt**: 2-shot YES/NO with uncertainty definition (V2, causal openness framing)
- **Train/test split**: 160/39 pairs (seed=42)

## Reproducing

```bash
conda activate ./econ_env
CUDA_VISIBLE_DEVICES=1 python investigate_uncertainty/steering.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 12 \
    --method mean \
    --steer_mode period \
    --def \
    --out_dir investigate_uncertainty
```
