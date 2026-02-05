"""
Activation analysis experiment using nnsight on multiple LLMs.
Extracts activations from all layers at the last token position and visualizes
uncertainty vs no-uncertainty clusters using PCA.
Generates 4 plots per model, each showing 8 layers.
"""

import json
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import trange
from nnsight import LanguageModel

# Configuration
DATA_PATH = "/projects/frink/wang.xil/concepts_E/data/uncertainty_labeling_sample.json"
OUTPUT_DIR = "/projects/frink/wang.xil/concepts_E"

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

def extract_activations_all_layers(model, data, num_layers):
    """Extract activations from all layers at the last token position."""
    # Initialize storage: {layer: {'uncertain': [], 'no_uncertain': []}}
    layer_activations = {layer: {'uncertain': [], 'no_uncertain': []} for layer in range(num_layers)}

    for i in trange(len(data), desc="Extracting activations"):
        item = data[i]
        prompt = format_prompt(item['question'], item['answer'])
        if i == 0:
            print(f"Sample prompt: {prompt[:]}...")

        saved_activations = []
        with torch.no_grad():
            with model.trace(prompt) as trace:
                # Save activations from all layers
                for layer in range(num_layers):
                    act = model.model.layers[layer].output[0][-1, :].unsqueeze(0).save()
                    saved_activations.append(act)

        # Store activations by label
        key = 'uncertain' if item['label'] == 1 else 'no_uncertain'
        for layer in range(num_layers):
            layer_activations[layer][key].append(saved_activations[layer])

    # Stack activations for each layer
    for layer in range(num_layers):
        layer_activations[layer]['uncertain'] = torch.cat(layer_activations[layer]['uncertain'])
        layer_activations[layer]['no_uncertain'] = torch.cat(layer_activations[layer]['no_uncertain'])

    return layer_activations

def visualize_pca_multi_layer(layer_activations, layers, output_path, model_display_name):
    """Create a 2x4 subplot figure for 8 layers."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        uncertain_acts = layer_activations[layer]['uncertain']
        no_uncertain_acts = layer_activations[layer]['no_uncertain']

        # Combine and apply PCA
        all_activations = torch.cat([uncertain_acts, no_uncertain_acts]).detach().cpu().float().numpy()
        pca = PCA(n_components=2)
        low_dim = pca.fit_transform(all_activations)

        # Split back
        n_uncertain = uncertain_acts.shape[0]
        uncertain_2d = low_dim[:n_uncertain]
        no_uncertain_2d = low_dim[n_uncertain:]

        # Plot
        ax.scatter(uncertain_2d[:, 0], uncertain_2d[:, 1],
                   c='red', label=f'Uncertain (n={n_uncertain})', alpha=0.7, s=30)
        ax.scatter(no_uncertain_2d[:, 0], no_uncertain_2d[:, 1],
                   c='blue', label=f'No Uncertainty (n={no_uncertain_2d.shape[0]})', alpha=0.7, s=30)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'Layer {layer}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'PCA of Layer Activations (Last Token) - {model_display_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def generate_plots_for_model(layer_activations, num_layers, model_display_name):
    """Generate 4 plots (8 layers each) for a model."""
    layers_per_plot = 8
    num_plots = (num_layers + layers_per_plot - 1) // layers_per_plot  # Ceiling division

    for plot_idx in range(num_plots):
        start_layer = plot_idx * layers_per_plot
        end_layer = min(start_layer + layers_per_plot, num_layers)
        layers = list(range(start_layer, end_layer))

        # Pad with empty subplots if needed (for last plot with fewer than 8 layers)
        output_path = f"{OUTPUT_DIR}/pca_{model_display_name}_layers_{start_layer:02d}_to_{end_layer-1:02d}.png"

        if len(layers) < layers_per_plot:
            # Create plot with fewer subplots for the last batch
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

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        uncertain_acts = layer_activations[layer]['uncertain']
        no_uncertain_acts = layer_activations[layer]['no_uncertain']

        # Combine and apply PCA
        all_activations = torch.cat([uncertain_acts, no_uncertain_acts]).detach().cpu().float().numpy()
        pca = PCA(n_components=2)
        low_dim = pca.fit_transform(all_activations)

        # Split back
        n_uncertain = uncertain_acts.shape[0]
        uncertain_2d = low_dim[:n_uncertain]
        no_uncertain_2d = low_dim[n_uncertain:]

        # Plot
        ax.scatter(uncertain_2d[:, 0], uncertain_2d[:, 1],
                   c='red', label=f'Uncertain (n={n_uncertain})', alpha=0.7, s=30)
        ax.scatter(no_uncertain_2d[:, 0], no_uncertain_2d[:, 1],
                   c='blue', label=f'No Uncertainty (n={no_uncertain_2d.shape[0]})', alpha=0.7, s=30)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'Layer {layer}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(layers), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'PCA of Layer Activations (Last Token) - {model_display_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def process_model(model_display_name, model_id, num_layers, data):
    """Process a single model: load, extract activations, generate plots."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_display_name} ({model_id})")
    print(f"{'='*60}")

    print(f"Loading model...")
    model = LanguageModel(model_id, device_map="auto")
    print(f"Model loaded: {model}")

    print(f"\nExtracting activations from all {num_layers} layers...")
    layer_activations = extract_activations_all_layers(model, data, num_layers)

    print("\nGenerating PCA visualizations...")
    generate_plots_for_model(layer_activations, num_layers, model_display_name)

    # Free memory
    del model
    del layer_activations
    torch.cuda.empty_cache()

def main():
    print("Loading labeled data...")
    data = load_labeled_data(DATA_PATH)
    n_uncertain = sum(1 for d in data if d['label'] == 1)
    n_no_uncertain = sum(1 for d in data if d['label'] == 0)
    print(f"Loaded {len(data)} labeled samples: {n_uncertain} uncertain, {n_no_uncertain} no uncertainty")

    for model_display_name, model_id, num_layers in MODELS:
        process_model(model_display_name, model_id, num_layers, data)

    print("\n" + "="*60)
    print("All models processed!")
    print("="*60)

if __name__ == "__main__":
    main()
