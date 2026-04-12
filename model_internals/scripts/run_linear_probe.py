"""
Linear probe analysis for uncertainty detection.
Trains logistic regression classifiers on layer activations to predict
uncertainty labels. Reports accuracy, F1, and AUC per layer for each model.

Supports two pooling modes:
- "last": Use last token activation
- "mean": Mean pooling across answer tokens
"""

import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange
from nnsight import LanguageModel
import pandas as pd

# Configuration
DATA_PATH = "/projects/frink/wang.xil/concepts_E/data/uncertainty_labeling_sample.json"
OUTPUT_DIR = "/projects/frink/wang.xil/concepts_E"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Pooling mode: "last" for last token, "mean" for mean pooling across answer tokens
POOLING_MODE = "mean"  # Change to "last" for last token

# Models to analyze: (display_name, huggingface_id, num_layers)
MODELS = [
    ("llama3.1-8b", "meta-llama/Llama-3.1-8B", 32),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B", 28),
    ("gemma2-9b", "google/gemma-2-9b", 42),
]

def load_labeled_data(path):
    """Load data and filter to only labeled samples. Merge high/intermediate uncertainty."""
    with open(path, 'r') as f:
        data = json.load(f)

    labeled_data = []
    for item in data:
        label = item.get('label', '')
        if label == '':
            continue  # Skip unlabeled

        # Binary label: 1 = uncertain (high or intermediate), 0 = no uncertainty
        if label in ['HIGH_UNCERTAINTY', 'INTERMEDIATE_UNCERTAINTY']:
            binary_label = 1
        elif label == 'NO_UNCERTAINTY':
            binary_label = 0
        else:
            continue  # Skip unknown labels

        labeled_data.append({
            'id': item['id'],
            'question': item['question'],
            'answer': item['answer'],
            'label': binary_label,
            'original_label': label
        })

    return labeled_data

def format_prompt(question, answer):
    """Format question and answer into a single input."""
    return f"Question: {question} Answer: {answer}"

def get_answer_start_idx(model, question):
    """Find the token index where the answer starts."""
    question_prefix = f"Question: {question} Answer:"
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
        prompt = format_prompt(item['question'], item['answer'])

        # For mean pooling, find where answer tokens start
        if pooling_mode == "mean":
            answer_start_idx = get_answer_start_idx(model, item['question'])

        if i == 0:
            print(f"Sample prompt: {prompt[:]}")
            if pooling_mode == "mean":
                print(f"Answer starts at token index: {answer_start_idx}")

        saved_activations = []
        with torch.no_grad():
            with model.trace(prompt) as trace:
                for layer in range(num_layers):
                    if pooling_mode == "last":
                        act = model.model.layers[layer].output[0][-1, :].save()
                    else:  # mean
                        act = model.model.layers[layer].output[0].save()
                    saved_activations.append(act)

        # Process activations based on pooling mode
        for layer in range(num_layers):
            if pooling_mode == "last":
                activation = saved_activations[layer].detach().cpu().float().numpy()
            else:  # mean
                answer_activations = saved_activations[layer][answer_start_idx:, :]
                activation = answer_activations.mean(dim=0).detach().cpu().float().numpy()
            layer_activations[layer].append(activation)

        labels.append(item['label'])

    # Convert to numpy arrays
    for layer in range(num_layers):
        layer_activations[layer] = np.stack(layer_activations[layer])  # (N, hidden_dim)

    labels = np.array(labels)
    return layer_activations, labels

def train_and_evaluate_probe(X_train, X_test, y_train, y_test):
    """Train a logistic regression probe and return metrics for both train and test."""
    # Handle class imbalance with balanced weights
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)

    # Test predictions
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    # Train predictions
    y_pred_train = clf.predict(X_train)
    y_prob_train = clf.predict_proba(X_train)[:, 1]

    # Test metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    try:
        test_auc = roc_auc_score(y_test, y_prob_test)
    except ValueError:
        test_auc = np.nan

    # Train metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)
    try:
        train_auc = roc_auc_score(y_train, y_prob_train)
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

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Split data
    print(f"\nSplitting data: {100*(1-TEST_SIZE):.0f}% train, {100*TEST_SIZE:.0f}% test")
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )

    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"Train: {len(train_idx)} samples ({sum(y_train)} uncertain, {len(y_train)-sum(y_train)} no uncertainty)")
    print(f"Test: {len(test_idx)} samples ({sum(y_test)} uncertain, {len(y_test)-sum(y_test)} no uncertainty)")

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
    print("Loading labeled data...")
    data = load_labeled_data(DATA_PATH)
    n_uncertain = sum(1 for d in data if d['label'] == 1)
    n_no_uncertain = sum(1 for d in data if d['label'] == 0)
    print(f"Loaded {len(data)} labeled samples: {n_uncertain} uncertain, {n_no_uncertain} no uncertainty")
    print(f"Pooling mode: {POOLING_MODE}")

    all_results = {}

    for model_display_name, model_id, num_layers in MODELS:
        df = process_model(model_display_name, model_id, num_layers, data)
        all_results[model_display_name] = df

        # Save individual model results
        output_path = f"{OUTPUT_DIR}/probe_results_{model_display_name}_{POOLING_MODE}.csv"
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
