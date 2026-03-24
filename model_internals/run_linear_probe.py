"""
Linear probe analysis for uncertainty detection.
Trains logistic regression classifiers on layer activations to predict
uncertainty labels. Reports accuracy, F1, and AUC per layer for each model.

Supports two pooling modes:
- "last": Use last token activation
- "mean": Mean pooling across answer tokens
"""

import gc
import json
import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange
from nnsight import LanguageModel
import pandas as pd

# Configuration - econ uncertainty (statement + uncertainty label)
DATA_PATH = "/home/shuyilin/7180/synthetic_data/econ_uncertainty_400_robust.json"
# Output folder includes dataset name (basename without .json)
_DATASET_NAME = os.path.splitext(os.path.basename(DATA_PATH))[0]
OUTPUT_DIR = f"./outputs/{_DATASET_NAME}"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Classification: 2 = binary, 3 = Low/Medium/High, 4 = no/low/medium/high (econ)
NUM_CLASSES = 4

# Pooling mode: "last" for last token, "mean" for mean pooling across answer tokens
POOLING_MODE = "mean"  # Change to "last" for last token

# Models to analyze: (display_name, huggingface_id, num_layers)
MODELS = [
    ("llama3.1-8b", "meta-llama/Llama-3.1-8B", 32),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B", 28),
    ("gemma2-9b", "google/gemma-2-9b", 42),
]

# Set MODEL_INDEX (0,1,2,...) to run only that model (avoids OOM when running in separate processes)
MODEL_INDEX = os.environ.get("MODEL_INDEX", None)

# Econ: uncertainty string -> numeric label (by NUM_CLASSES)
def _econ_uncertainty_to_label(uncertainty_str, num_classes):
    u = (uncertainty_str or "").strip().lower()
    if num_classes == 2:
        return 0 if u == "no" else 1
    if num_classes == 3:
        if u == "no":
            return 0
        if u == "low":
            return 1
        return 2  # medium, high
    # 4-class
    return {"no": 0, "low": 1, "medium": 2, "high": 3}.get(u, 0)


def load_labeled_data(path):
    """Load data. 直接标签: statement+uncertainty (no/low/medium/high)；或 Q&A 格式."""
    with open(path, 'r') as f:
        data = json.load(f)

    labeled_data = []
    first = data[0] if data else {}

    if 'statement' in first and 'uncertainty' in first:
        # 直接标签: statement + uncertainty ("no"|"low"|"medium"|"high")，不计算 variance
        for item in data:
            raw = (item.get('uncertainty') or '').strip().lower()
            if raw not in ('no', 'low', 'medium', 'high'):
                continue
            label = _econ_uncertainty_to_label(raw, NUM_CLASSES)
            labeled_data.append({
                'id': item.get('id'),
                'statement': item['statement'],
                'label': label,
                '_format': 'synthetic'
            })
    else:
        # Original Q&A format
        for item in data:
            label = item.get('label', '')
            if label == '':
                continue
            if label in ['HIGH_UNCERTAINTY', 'INTERMEDIATE_UNCERTAINTY']:
                binary_label = 1
            elif label == 'NO_UNCERTAINTY':
                binary_label = 0
            else:
                continue
            labeled_data.append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'label': binary_label,
                'original_label': label,
                '_format': 'qa'
            })

    return labeled_data

def format_prompt(item):
    """Format item into model input string."""
    if item.get('_format') == 'synthetic':
        return f"Statement: {item['statement']}"
    return f"Question: {item['question']} Answer: {item['answer']}"

def get_pooling_start_idx(model, item):
    """Find token index where content starts for mean pooling."""
    if item.get('_format') == 'synthetic':
        prefix = "Statement: "
        prefix_tokens = model.tokenizer(prefix, return_tensors="pt").input_ids
        return prefix_tokens.shape[1]
    question_prefix = f"Question: {item['question']} Answer:"
    prefix_tokens = model.tokenizer(question_prefix, return_tensors="pt").input_ids
    return prefix_tokens.shape[1]

def extract_activations_all_layers(model, data, num_layers, pooling_mode):
    """Extract activations from all layers.

    Args:
        pooling_mode: "last" for last token, "mean" for mean pooling across answer tokens
    """
    layer_activations = {layer: [] for layer in range(num_layers)}
    labels = []

    for i in trange(len(data), desc="Extracting activations"):
        item = data[i]
        prompt = format_prompt(item)

        if pooling_mode == "mean":
            content_start_idx = get_pooling_start_idx(model, item)

        if i == 0:
            print(f"Sample prompt: {prompt[:200]}...")
            if pooling_mode == "mean":
                print(f"Content starts at token index: {content_start_idx}")

        saved_activations = []
        with torch.no_grad():
            with model.trace(prompt) as trace:
                for layer in range(num_layers):
                    if pooling_mode == "last":
                        act = model.model.layers[layer].output[0][-1, :].save()
                    else:  # mean
                        act = model.model.layers[layer].output[0].save()
                    saved_activations.append(act)

        for layer in range(num_layers):
            if pooling_mode == "last":
                activation = saved_activations[layer].detach().cpu().float().numpy()
            else:  # mean
                content_activations = saved_activations[layer][content_start_idx:, :]
                activation = content_activations.mean(dim=0).detach().cpu().float().numpy()
            layer_activations[layer].append(activation)

        labels.append(item['label'])

    # Convert to numpy arrays
    for layer in range(num_layers):
        layer_activations[layer] = np.stack(layer_activations[layer])  # (N, hidden_dim)

    labels = np.array(labels)
    return layer_activations, labels

def train_and_evaluate_probe(X_train, X_test, y_train, y_test):
    """Train a logistic regression probe and return metrics for both train and test."""
    n_classes_train = len(np.unique(y_train))
    if n_classes_train < 2:
        return {
            'train_accuracy': np.nan, 'train_f1': np.nan, 'train_auc': np.nan,
            'test_accuracy': np.nan, 'test_f1': np.nan, 'test_auc': np.nan
        }
    multiclass = NUM_CLASSES >= 3
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)
    y_pred_train = clf.predict(X_train)
    y_prob_train = clf.predict_proba(X_train)

    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

    if multiclass:
        try:
            test_auc = roc_auc_score(y_test, y_prob_test, multi_class='ovr', average='weighted')
        except ValueError:
            test_auc = np.nan
        try:
            train_auc = roc_auc_score(y_train, y_prob_train, multi_class='ovr', average='weighted')
        except ValueError:
            train_auc = np.nan
    else:
        y_prob_test_bin = y_prob_test[:, 1]
        y_prob_train_bin = y_prob_train[:, 1]
        try:
            test_auc = roc_auc_score(y_test, y_prob_test_bin)
        except ValueError:
            test_auc = np.nan
        try:
            train_auc = roc_auc_score(y_train, y_prob_train_bin)
        except ValueError:
            train_auc = np.nan

    return {
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_auc': test_auc
    }

def process_model(model_display_name, model_id, num_layers, data):
    """Process a single model: extract activations, train probes, report results."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_display_name} ({model_id})")
    print(f"Pooling mode: {POOLING_MODE}")
    print(f"{'='*60}")

    print("Loading model...")
    model = LanguageModel(model_id, device_map="auto")

    print(f"\nExtracting activations from all {num_layers} layers...")
    layer_activations, labels = extract_activations_all_layers(model, data, num_layers, POOLING_MODE)

    # Free model memory (aggressive cleanup to avoid OOM when loading next model)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Split data
    print(f"\nSplitting data: {100*(1-TEST_SIZE):.0f}% train, {100*TEST_SIZE:.0f}% test")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )

    y_train, y_test = labels[train_idx], labels[test_idx]
    if NUM_CLASSES == 2:
        print(f"Train: {len(train_idx)} samples ({int(sum(y_train))} uncertain, {len(y_train)-int(sum(y_train))} no uncertainty)")
        print(f"Test: {len(test_idx)} samples ({int(sum(y_test))} uncertain, {len(y_test)-int(sum(y_test))} no uncertainty)")
    elif NUM_CLASSES == 4:
        n0_tr, n1_tr, n2_tr, n3_tr = np.sum(y_train == 0), np.sum(y_train == 1), np.sum(y_train == 2), np.sum(y_train == 3)
        n0_te, n1_te, n2_te, n3_te = np.sum(y_test == 0), np.sum(y_test == 1), np.sum(y_test == 2), np.sum(y_test == 3)
        print(f"Train: {len(train_idx)} samples (no={n0_tr}, low={n1_tr}, medium={n2_tr}, high={n3_tr})")
        print(f"Test: {len(test_idx)} samples (no={n0_te}, low={n1_te}, medium={n2_te}, high={n3_te})")
    else:
        n0_tr, n1_tr, n2_tr = np.sum(y_train == 0), np.sum(y_train == 1), np.sum(y_train == 2)
        n0_te, n1_te, n2_te = np.sum(y_test == 0), np.sum(y_test == 1), np.sum(y_test == 2)
        print(f"Train: {len(train_idx)} samples (low={n0_tr}, medium={n1_tr}, high={n2_tr})")
        print(f"Test: {len(test_idx)} samples (low={n0_te}, medium={n1_te}, high={n2_te})")

    # Train probes for each layer
    print("\nTraining linear probes...")
    results = []
    for layer in trange(num_layers, desc="Training probes"):
        X_train = layer_activations[layer][train_idx]
        X_test = layer_activations[layer][test_idx]

        metrics = train_and_evaluate_probe(X_train, X_test, y_train, y_test)
        metrics['layer'] = layer
        results.append(metrics)

    # Create results table
    df = pd.DataFrame(results)
    df = df[['layer', 'train_accuracy', 'train_f1', 'train_auc', 'test_accuracy', 'test_f1', 'test_auc']]

    return df

def print_results_table(df, model_name):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"Results for {model_name}")
    print(f"{'='*80}")

    # Format percentages
    df_display = df.copy()
    for col in ['train_accuracy', 'train_f1', 'test_accuracy', 'test_f1']:
        df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%")
    for col in ['train_auc', 'test_auc']:
        df_display[col] = df_display[col].apply(lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A")

    print(df_display.to_string(index=False))

    # Summary stats
    print(f"\nBest layer by test accuracy: {df.loc[df['test_accuracy'].idxmax(), 'layer']} ({df['test_accuracy'].max()*100:.1f}%)")
    print(f"Best layer by test F1: {df.loc[df['test_f1'].idxmax(), 'layer']} ({df['test_f1'].max()*100:.1f}%)")
    if not df['test_auc'].isna().all():
        print(f"Best layer by test AUC: {df.loc[df['test_auc'].idxmax(), 'layer']} ({df['test_auc'].max()*100:.1f}%)")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading labeled data...")
    data = load_labeled_data(DATA_PATH)
    if NUM_CLASSES == 2:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        print(f"Loaded {len(data)} labeled samples: {n1} uncertain, {n0} no uncertainty (2-class)")
    elif NUM_CLASSES == 4:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        n2 = sum(1 for d in data if d['label'] == 2)
        n3 = sum(1 for d in data if d['label'] == 3)
        print(f"Loaded {len(data)} labeled samples: no={n0}, low={n1}, medium={n2}, high={n3} (4-class)")
    else:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        n2 = sum(1 for d in data if d['label'] == 2)
        print(f"Loaded {len(data)} labeled samples: low={n0}, medium={n1}, high={n2} (3-class)")
    print(f"Pooling mode: {POOLING_MODE}")

    all_results = {}
    models_to_run = MODELS
    if MODEL_INDEX is not None:
        idx = int(MODEL_INDEX)
        models_to_run = [MODELS[idx]]
        print(f"Running single model: {models_to_run[0][0]} (MODEL_INDEX={idx})")

    for model_display_name, model_id, num_layers in models_to_run:
        df = process_model(model_display_name, model_id, num_layers, data)
        all_results[model_display_name] = df

        # Save individual model results
        output_path = f"{OUTPUT_DIR}/probe_results_{model_display_name}_{POOLING_MODE}_{NUM_CLASSES}class.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        # Print table
        print_results_table(df, model_display_name)

    # Summary comparison across models
    print("\n" + "="*60)
    print("SUMMARY: Best performing layers across models")
    print("="*60)

    summary_data = []
    for model_name, df in all_results.items():
        best_acc_layer = df.loc[df['test_accuracy'].idxmax(), 'layer']
        best_acc = df['test_accuracy'].max()
        best_f1_layer = df.loc[df['test_f1'].idxmax(), 'layer']
        best_f1 = df['test_f1'].max()
        best_auc = df['test_auc'].max() if not df['test_auc'].isna().all() else np.nan

        summary_data.append({
            'model': model_name,
            'best_acc_layer': best_acc_layer,
            'best_test_accuracy': f"{best_acc*100:.1f}%",
            'best_f1_layer': best_f1_layer,
            'best_test_f1': f"{best_f1*100:.1f}%",
            'best_test_auc': f"{best_auc*100:.1f}%" if not np.isnan(best_auc) else "N/A"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\nDone!")

if __name__ == "__main__":
    main()
