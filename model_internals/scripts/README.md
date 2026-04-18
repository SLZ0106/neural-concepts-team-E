# model_internals/scripts

Mechanistic interpretability scripts for studying how transformer models represent economic uncertainty. All scripts operate on contrastive pair datasets (paired uncertain/certain economic statements) and use `nnsight` to access internal model activations.

## Shared utilities

### `activation_patching_highlow.py`
Activation patching on HIGH/LOW uncertainty classification, and the source of shared prompt helpers (`build_messages`, `build_messages_synthetic`, `apply_template`, `find_last_period_pos`, `get_label_token_ids`, `load_pairs`) used by most other scripts. Patches the residual stream at each layer at the final period token (and optionally 20 tokens before it), producing per-layer logit-diff curves (HIGH − LOW).

### `activation_patching_yesno.py`
YES/NO variant of activation patching. Patches only the final period token; outputs per-layer logit diff (YES − NO).

## Direction extraction and intervention (main pipeline)

These two scripts replace the older `steering_highlow.py`, `steering_cross_dataset_highlow.py`, `massmean_highlow.py`, and `massmean_cross_dataset.py`. The pipeline runs in two steps.

### `extract_direction.py`
**Step 1 — extract an uncertainty direction and validate it.**

Extracts train activations from a source dataset, computes a direction (mean-diff or PCA of uncertain − certain activations), then validates on the test pairs of a target dataset.

- `--source {templated,synthetic}` and `--target {templated,synthetic}` — dataset names resolve to fixed paths in `DATASET_PATHS`. Same-dataset when source == target; cross-dataset otherwise.
- `--probe {projection,massmean,both}` — probe technique for validation. `projection` uses zero-boundary sign classification on the direction; `massmean` uses Mahalanobis-distance classification with Ledoit-Wolf shrinkage.
- `--method {mean,pca}` — how to compute the direction from paired activation diffs.
- Outputs: `direction.npy`, `train_{uncertain,certain}.npy`, `test_{uncertain,certain}.npy`, `massmean_probe.npz` (if massmean), per-probe validation JSON and histograms, `split_info.json`.

Train/test split: for `templated`, pair 0 is the demo pair (consumed by the 2-shot prompt), train = pairs 1..n_train, test = pairs n_train+1..end. For `synthetic`, demos are fixed in the prompt, train = pairs 0..n_train-1, test = rest.

### `run_intervention.py`
**Step 2 — run steering intervention with a saved direction.**

Loads a `.npy` direction produced by `extract_direction.py`, sweeps alpha values, and adds `alpha * direction` to the residual stream at a chosen layer during inference. Classifies model output as HIGH / LOW / OTHER and plots accuracy, logit diff, and per-class accuracy vs. alpha.

- `--direction` — path to `.npy` direction file.
- `--dataset {templated,synthetic}` — target dataset; resolves to a fixed path.
- `--steer_mode {period,last,all}` — which token positions receive the perturbation.
- `--alphas` — comma-separated list of steering strengths.
- Outputs: `intervention_results_{train,test}.json`, `intervention_{train,test}_{accuracy,logitdiff,perclass}_vs_alpha.png`, `intervention_config.json`.

Evaluates both train and test splits so you can spot overfitting of the direction.

## Other analyses

### `multi_position_patching.py`
Patches K consecutive token positions ending at the period token, for K in a configurable list. Produces scaling curves showing how the patching effect grows with window size.

### `pre_period_patching.py`
Targeted patching comparison at layer 12 across three windows: period-only, 19 pre-period tokens (excluding period), and the combined window. Isolates which part of the statement drives the uncertainty signal.

### `direction_similarity.py`
Computes the uncertainty direction (mean activation diff) at each position in a 20-token window around the period, then produces cosine-similarity matrices between positions. Used to check whether the direction is consistent across the statement or localized to specific tokens.

### `run_linear_probe.py`
Trains logistic-regression probes per layer (optionally last-token or mean-pooled) on labeled activations, across multiple models. Outputs per-layer accuracy/F1/AUC to identify which layers encode uncertainty.

### `run_pca.py`
PCA visualization of per-layer activations. Produces multi-panel scatter plots (8 layers per panel) showing cluster separation between uncertain and no-uncertainty classes.

## YES/NO variants

### `steering_yesno.py`
YES/NO counterpart to the HIGH/LOW `extract_direction` + `run_intervention` pair. Still a monolithic 4-stage pipeline (extract direction → validate → steer → plot). Has not been refactored; if you need the same two-step decomposition for YES/NO experiments, the refactor would mirror the HIGH/LOW split.

## Typical workflow

```bash
# Same-dataset: synthetic train → synthetic test, both probes
python extract_direction.py --source synthetic --target synthetic \
    --probe both --out_dir results/synthetic/

# Intervention using that direction
python run_intervention.py --direction results/synthetic/direction.npy \
    --dataset synthetic --steer_mode period \
    --alphas "-5,-4,-3,-2,-1,0,1,2,3,4,5" \
    --out_dir results/synthetic/

# Cross-dataset: train on synthetic, test on templated
python extract_direction.py --source synthetic --target templated \
    --probe massmean --out_dir results/cross/

# Cross-dataset intervention (reuse the synthetic-trained direction)
python run_intervention.py --direction results/synthetic/direction.npy \
    --dataset templated --out_dir results/cross/
```
