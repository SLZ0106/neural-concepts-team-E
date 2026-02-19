"""
Activation analysis experiment using nnsight on multiple LLMs.
Extracts activations from all layers and visualizes uncertainty vs
no-uncertainty clusters using PCA.
Generates 4 plots per model, each showing 8 layers.

Supports two pooling modes:
- "last": Use last token activation
- "mean": Mean pooling across answer tokens
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import trange
from nnsight import LanguageModel

# Configuration - econ uncertainty (statement + uncertainty label)
DATA_PATH = "/home/shuyilin/7180/synthetic_data/econ_uncertainty_400_robust.json"

# Pooling mode: "last" for last token, "mean" for mean pooling across statement/answer tokens
POOLING_MODE = "mean"

# Classification: 2 = binary, 3 = Low/Medium/High, 4 = no/low/medium/high (econ)
NUM_CLASSES = 4
# Output folder includes dataset name (basename without .json)
_DATASET_NAME = os.path.splitext(os.path.basename(DATA_PATH))[0]
OUTPUT_DIR = f"./outputs/{_DATASET_NAME}/activation_{POOLING_MODE}_{NUM_CLASSES}class"

# Models to analyze: (display_name, huggingface_id, num_layers)
MODELS = [
    ("llama3.1-8b", "meta-llama/Llama-3.1-8B", 32),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B", 28),
    ("gemma2-9b", "google/gemma-2-9b", 42),
    # ("llama3.1-70B", "meta-llama/Llama-3.1-70B", 80)   
]

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

def get_class_keys():
    """Return class keys based on NUM_CLASSES."""
    if NUM_CLASSES == 2:
        return ['no_uncertain', 'uncertain']
    if NUM_CLASSES == 4:
        return ['no', 'low', 'medium', 'high']
    return ['low', 'medium', 'high']

def label_to_key(label):
    if NUM_CLASSES == 2:
        return 'uncertain' if label == 1 else 'no_uncertain'
    if NUM_CLASSES == 4:
        return ['no', 'low', 'medium', 'high'][label]
    return ['low', 'medium', 'high'][label]

def extract_activations_all_layers(model, data, num_layers, pooling_mode):
    """Extract activations from all layers."""
    keys = get_class_keys()
    layer_activations = {layer: {k: [] for k in keys} for layer in range(num_layers)}

    for i in trange(len(data), desc="Extracting activations"):
        item = data[i]
        prompt = format_prompt(item)

        if pooling_mode == "mean":
            content_start_idx = get_pooling_start_idx(model, item)

        if i == 0:
            print(f"Sample prompt: {prompt[:200]}...")
            if pooling_mode == "mean":
                print(f"Content starts at token index: {content_start_idx}")

        with torch.no_grad():
            with model.trace(prompt) as trace:
                saved_activations = list().save()
                for layer in range(num_layers):
                    if pooling_mode == "last":
                        act = model.model.layers[layer].output[0][-1, :].unsqueeze(0).save()
                    else:  # mean
                        act = model.model.layers[layer].output[0][content_start_idx:, :].mean(dim=0).unsqueeze(0).save()
                    saved_activations.append(act)

        key = label_to_key(item['label'])
        for layer in range(num_layers):
            layer_activations[layer][key].append(saved_activations[layer])

    for layer in range(num_layers):
        for k in keys:
            if layer_activations[layer][k]:
                layer_activations[layer][k] = torch.cat(layer_activations[layer][k])
            else:
                layer_activations[layer][k] = torch.empty(0)

    return layer_activations

def get_pooling_label():
    """Get display label for current pooling mode."""
    if POOLING_MODE == "last":
        return "Last Token"
    else:
        return "Answer Mean Pool"

# Colors and labels for each class
CLASS_CONFIG = {
    2: [
        ('no_uncertain', 'blue', 'Low (var=0)'),
        ('uncertain', 'red', 'High (var>0)'),
    ],
    3: [
        ('low', 'blue', 'Low (≤p33)'),
        ('medium', 'orange', 'Medium (p33–p67)'),
        ('high', 'red', 'High (>p67)'),
    ],
    4: [
        ('no', 'green', 'No'),
        ('low', 'blue', 'Low'),
        ('medium', 'orange', 'Medium'),
        ('high', 'red', 'High'),
    ],
}

def _plot_pca_single_ax(ax, layer_activations, layer, class_config):
    keys = get_class_keys()
    to_cat = [layer_activations[layer][k] for k in keys]
    to_cat = [t for t in to_cat if t.numel() > 0]
    if not to_cat:
        return
    all_acts = torch.cat(to_cat).detach().cpu().float().numpy()
    pca = PCA(n_components=2)
    low_dim = pca.fit_transform(all_acts)

    offset = 0
    for cfg in class_config:
        key, color, label = cfg[0], cfg[1], cfg[2]
        n = layer_activations[layer][key].shape[0]
        if n == 0:
            continue
        pts = low_dim[offset:offset + n]
        offset += n
        ax.scatter(pts[:, 0], pts[:, 1], c=color, label=f'{label} (n={n})', alpha=0.7, s=30)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Layer {layer}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def visualize_pca_multi_layer(layer_activations, layers, output_path, model_display_name):
    """Create a 2x4 subplot figure for 8 layers."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    class_config = CLASS_CONFIG[NUM_CLASSES]

    for idx, layer in enumerate(layers):
        _plot_pca_single_ax(axes[idx], layer_activations, layer, class_config)

    plt.suptitle(f'PCA of Layer Activations ({get_pooling_label()}) - {model_display_name} ({NUM_CLASSES}-class)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def generate_plots_for_model(layer_activations, num_layers, model_display_name):
    """Generate 4 plots (8 layers each) for a model."""
    layers_per_plot = 8
    num_plots = (num_layers + layers_per_plot - 1) // layers_per_plot

    for plot_idx in range(num_plots):
        start_layer = plot_idx * layers_per_plot
        end_layer = min(start_layer + layers_per_plot, num_layers)
        layers = list(range(start_layer, end_layer))

        output_path = f"{OUTPUT_DIR}/pca_{model_display_name}_{POOLING_MODE}_layers_{start_layer:02d}_to_{end_layer-1:02d}.png"

        if len(layers) < layers_per_plot:
            visualize_pca_partial(layer_activations, layers, output_path, model_display_name)
        else:
            visualize_pca_multi_layer(layer_activations, layers, output_path, model_display_name)

def visualize_pca_partial(layer_activations, layers, output_path, model_display_name):
    """Create a subplot figure for fewer than 8 layers."""
    n_layers = len(layers)
    ncols = min(4, n_layers)
    nrows = (n_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    class_config = CLASS_CONFIG[NUM_CLASSES]
    for idx, layer in enumerate(layers):
        _plot_pca_single_ax(axes[idx], layer_activations, layer, class_config)

    for idx in range(len(layers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'PCA of Layer Activations ({get_pooling_label()}) - {model_display_name} ({NUM_CLASSES}-class)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def process_model(model_display_name, model_id, num_layers, data):
    """Process a single model: load, extract activations, generate plots."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_display_name} ({model_id})")
    print(f"Pooling mode: {POOLING_MODE}")
    print(f"{'='*60}")

    print(f"Loading model...")
    model = LanguageModel(model_id, device_map="auto")
    print(f"Model loaded: {model}")

    print(f"\nExtracting activations from all {num_layers} layers...")
    layer_activations = extract_activations_all_layers(model, data, num_layers, POOLING_MODE)

    print("\nGenerating PCA visualizations...")
    generate_plots_for_model(layer_activations, num_layers, model_display_name)

    # Free memory
    del model
    del layer_activations
    torch.cuda.empty_cache()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading labeled data...")
    data = load_labeled_data(DATA_PATH)
    if NUM_CLASSES == 2:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        print(f"Loaded {len(data)} samples: {n0} low, {n1} high (2-class)")
    elif NUM_CLASSES == 4:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        n2 = sum(1 for d in data if d['label'] == 2)
        n3 = sum(1 for d in data if d['label'] == 3)
        print(f"Loaded {len(data)} samples: no={n0}, low={n1}, medium={n2}, high={n3} (4-class)")
    else:
        n0 = sum(1 for d in data if d['label'] == 0)
        n1 = sum(1 for d in data if d['label'] == 1)
        n2 = sum(1 for d in data if d['label'] == 2)
        print(f"Loaded {len(data)} samples: {n0} low, {n1} medium, {n2} high (3-class)")
    print(f"Pooling mode: {POOLING_MODE}")

    for model_display_name, model_id, num_layers in MODELS:
        process_model(model_display_name, model_id, num_layers, data)

    print("\n" + "="*60)
    print("All models processed!")
    print("="*60)

if __name__ == "__main__":
    main()
