import csv
import hashlib
import json
import os
import re
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViTSmall16_Weights,
    resnet18,
    resnet50,
    vit_small_patch16_224,
)

from ciip.model_ciip import CIIP
from ciip.eval_utils import create_ciip_model

def sample_episode(dataset, n_way, k_shot, query_per_class=15):
    """
    Sample an episode for few-shot learning.
    
    Args:
        dataset: list or object with __getitem__ that returns (feature, label)
        n_way: number of classes in the episode
        k_shot: number of support samples per class
        query_per_class: number of query samples per class
        
    Returns:
        support_features: array/tensor of shape [n_way * k_shot, ...]
        support_labels: array of shape [n_way * k_shot]
        query_features: array/tensor of shape [n_way * query_per_class, ...]
        query_labels: array of shape [n_way * query_per_class]
    """
    
    # 1. Get all unique classes in the dataset
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    unique_classes = list(set(all_labels))
    
    # 2. Randomly select n_way classes
    chosen_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_features = []
    support_labels = []
    query_features = []
    query_labels = []
    
    for idx, cls in enumerate(chosen_classes):
        # Get indices of all samples in this class
        cls_indices = [i for i, label in enumerate(all_labels) if label == cls]
        
        # Make sure there are enough samples
        if len(cls_indices) < k_shot + query_per_class:
            raise ValueError(f"Not enough samples for class {cls} to sample support + query")
        
        # Randomly sample without replacement
        selected = np.random.choice(cls_indices, k_shot + query_per_class, replace=False)
        
        support_idx = selected[:k_shot]
        query_idx = selected[k_shot:]
        
        # Extract features and labels for support
        for si in support_idx:
            feat, label = dataset[si]
            support_features.append(feat)
            support_labels.append(idx)  # remap label to [0, n_way-1]
        
        # Extract features and labels for query
        for qi in query_idx:
            feat, label = dataset[qi]
            query_features.append(feat)
            query_labels.append(idx)  # remap label to [0, n_way-1]
    
    # Convert lists to arrays or tensors (depending on your data)
    # Here assuming numpy arrays
    
    support_features = np.stack(support_features)
    support_labels = np.array(support_labels)
    query_features = np.stack(query_features)
    query_labels = np.array(query_labels)
    
    return support_features, support_labels, query_features, query_labels



def plot_primary_over_epochs(
    results,
    k_values,
    metric="accuracy",
    primary_regex=r"epoch(\d+)",
    baselines=("SSL4EO-ResNet50_MoCo", "SSL4EO-ResNet50_DINO", "ResNet50_Random"),
    layout="facets",
    title=None,
    output_file=None,
):
    rx = re.compile(primary_regex) if isinstance(primary_regex, str) else primary_regex

    # --- collect primary runs (name, epoch) ---
    primary = []
    for name in results:
        print(results)
        m = rx.search(name)
        if m:
            try:
                epoch = int(m.group(1))
            except (ValueError, IndexError):
                continue
            primary.append((epoch, name))
    if not primary:
        raise ValueError(f"No primary runs matched regex: {primary_regex}")

    primary.sort(key=lambda x: x[0])
    epochs = [e for e, _ in primary]

    # cache per-k series (means/stds aligned to sorted epochs)
    per_k = {}
    for k in k_values:
        means, stds = [], []
        for _, name in primary:
            means.append(results[name][k][f"{metric}_mean"])
            stds.append(results[name][k][f"{metric}_std"])
        per_k[k] = (np.asarray(means), np.asarray(stds))

    # --- fig/axes ---
    if layout == "facets":
        n = len(k_values)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5*ncols, 4.2*nrows), squeeze=False)
        ax_list = [ax for row in axes for ax in row][:n]
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_list = [ax]

    markers = ['o','s','^','D','P','X','v','>','<','*']

    def draw_baselines_for_k(ax, k, x0, x1):
        """Draw dashed baselines only for this k, and return handles/labels."""
        handles, labels = [], []
        for bname in baselines or []:
            if bname not in results: 
                continue
            if k not in results[bname]:
                continue
            if f"{metric}_mean" not in results[bname][k]:
                continue
            mu = results[bname][k][f"{metric}_mean"]
            (line,) = ax.plot([x0, x1], [mu, mu],
                              linestyle='--', linewidth=1.2, alpha=0.7,
                              label=f"{bname} (k={k})")
            handles.append(line); labels.append(line.get_label())
        return handles, labels

    if layout == "facets":
        for idx, (ax, k) in enumerate(zip(ax_list, k_values)):
            means, stds = per_k[k]
            # primary line — NO label (so only dashed lines appear in legend)
            (pline,) = ax.plot(epochs, means, marker=markers[idx % len(markers)])
            ax.fill_between(epochs, means-stds, means+stds, alpha=0.25, color=pline.get_color())
            # baselines ONLY for this k
            x0, x1 = min(epochs), max(epochs)
            b_handles, b_labels = draw_baselines_for_k(ax, k, x0, x1)
            # legend: only dashed baselines
            if b_handles:
                ax.legend(b_handles, b_labels, loc="lower right", fontsize=9, framealpha=0.35)
            ax.set_xlabel("Epoch"); ax.set_ylabel(metric.capitalize())
            ax.set_title(f"CIIP across epochs (k={k})")
            ax.grid(True, alpha=0.3)
        # hide any unused axes
        if len(ax_list) < axes.size:
            for ax in axes.flatten()[len(ax_list):]:
                ax.axis("off")
        if title:
            fig.suptitle(title, y=0.995, fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
        else:
            fig.tight_layout()
    else:
        ax = ax_list[0]
        # plot all k series (no labels)
        for idx, k in enumerate(k_values):
            means, stds = per_k[k]
            (pline,) = ax.plot(epochs, means, marker=markers[idx % len(markers)])
            ax.fill_between(epochs, means-stds, means+stds, alpha=0.25, color=pline.get_color())
        # draw baselines once per k and label them
        x0, x1 = min(epochs), max(epochs)
        all_bh, all_bl = [], []
        for k in k_values:
            bh, bl = draw_baselines_for_k(ax, k, x0, x1)
            all_bh += bh; all_bl += bl
        if all_bh:
            ax.legend(all_bh, all_bl, loc="lower right", framealpha=0.35)
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric.capitalize())
        ax.set_title(title or f"CIIP across epochs — {metric}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    if output_file:
        fig.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.show()





# Plot Results
def plot_results(results, k_values, metric="accuracy", output_file="few_shot_results.png"):
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        few_shot_k = [k for k in k_values if k != 0]
        # Extract mean and std for the few-shot
        if few_shot_k:
            few_shot_means = np.array([model_results[k][f"{metric}_mean"] for k in few_shot_k])
            few_shot_stds = np.array([model_results[k][f"{metric}_std"] for k in few_shot_k])

            # Plot the line + markers for the few-shot
            line, = plt.plot(few_shot_k, few_shot_means, marker='o', linestyle='-', label=model_name)
            # Use the same color for the shaded area
            line_color = line.get_color()
            plt.fill_between(few_shot_k,
                             few_shot_means - few_shot_stds,
                             few_shot_means + few_shot_stds,
                             alpha=0.3,
                             color=line_color)
        else:
            line = None
            line_color = None

        # Plot the zero-shot point (k==0) with the same color
        if 0 in k_values:
            zero_shot_mean = model_results[0][f"{metric}_mean"]
            zero_shot_std = model_results[0][f"{metric}_std"]

            # If we never plotted a line for few-shot (e.g., no few_shot_k),
            # we need to create a legend entry for the model.
            if line is None:
                # Create a scatter with a label for the legend
                scatter = plt.scatter(0, zero_shot_mean, marker='*', s=150, label=model_name)
            else:
                # Use the same color as the line
                plt.scatter(0, zero_shot_mean, marker='*', s=150, color=line_color)

    plt.xlabel("k Samples for Few Shot")
    plt.ylabel(metric.capitalize())
    plt.title(f"Few Shot + Linear Probe with Classification Head : {metric.capitalize()} vs. k")
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def output_results_to_csv(results, k_values, metric="accuracy", output_dir="."):
    """
    Write the results to a CSV file with model names as rows and k values as columns.
    Each cell contains the metric value (accuracy or F1 score).
    The filename includes the timestamp.
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_file = os.path.join(output_dir, f"results_{metric}_{timestamp}.csv")

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header row: "Model Name" followed by k values
        header = ["Model Name"] + [f"k={k}" for k in k_values] + [f"Std k={k}" for k in k_values]
        writer.writerow(header)

        # Write rows: model name followed by metric values for each k
        for model_name, model_results in results.items():
            row = [model_name]  # Start the row with the model name
            for k in k_values:
                row.append(model_results[k][f'{metric}_mean'])  # Add the metric value for each k
            for k in k_values:
                row.append(model_results[k][f'{metric}_std'])
            writer.writerow(row)

    print(f"Results saved to {output_file}")

def load_model(model_type, weights_path):
    # You can adapt this to your architecture loading logic
    if model_type == "vit16":
        raise NotImplementedError("ViT16 model loading is not implemented in this example.")
        # model = load_vit16(weights_path)
    elif model_type == "resnet50":

        if weights_path is None:
            model = resnet50(weights=None, num_classes=1000, in_chans=13, pretrained=False)
            print(f"First layer shape: {model.conv1.weight.shape}")
        else:    
            print(weights_path)
            model = resnet50(weights_path)
            # print first layer shape
            print(f"First layer shape: {model.conv1.weight.shape}")
    elif model_type == "ciip":
        model = load_ciip_model_checkpoint(weights_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model



def get_deterministic_seed_for_experiment(k, exp):
    """Generate a seed based solely on k and the experiment index."""
    s = f"{k}_{exp}"
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2 ** 32)


########### MODEL ###########
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample

def drop_last_linear_layer(model):
    """
    Replace the model's final linear layer with an identity function so that the features
    from the backbone are returned without applying the last linear transformation.
    """
    if hasattr(model, 'encoder_s2') and hasattr(model.encoder_s2, 'fc'):
        # Replace the fc layer of encoder_s2 with an identity mapping.
        model.encoder_s2.fc = nn.Identity()
    elif hasattr(model, 'fc'):
        # If the model has a top-level fc, replace it.
        model.fc = nn.Identity()
    else:
        print("Warning: No final linear layer found to drop!")
    return model

def create_ciip_model(embed_dim, pre_projection_dim=1024):
    s1_bands = [1, 2]
    s2_bands = list(range(1, 14))  # Bands 1 through 13
    model = CIIP(
        framework="resnet50",
        embed_dim=embed_dim,
        pre_projection_dim=pre_projection_dim,
        s1_resolution=224,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,  # used by transformer
        s1_bands=len(s1_bands),
        s2_resolution=224,
        s2_layers=(3, 4, 6, 3),  # ResNet-34
        s2_width=32,
        s2_patch_size=16,  # used by transformer
        s2_bands=len(s2_bands),
    )
    return model

def load_ciip_model_checkpoint(checkpoint_path):
    

    # print(f'Using embed {embed_dim} with pre-projection dim {pre_projection_dim}')
    try:
        embed_dim = 512
        pre_projection_dim = 1024
        model = create_ciip_model(embed_dim, pre_projection_dim=pre_projection_dim)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu') # , map_location='cuda'
        state_dict = checkpoint["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_state_dict = {k: v for k, v in new_state_dict.items() if "fc" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    except:
        embed_dim = 1024
        pre_projection_dim = 2048
        model = create_ciip_model(embed_dim, pre_projection_dim=pre_projection_dim)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu') # , map_location='cuda'
        state_dict = checkpoint["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_state_dict = {k: v for k, v in new_state_dict.items() if "fc" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    assert set(missing_keys) <= {'encoder_s1.fc.weight', 'encoder_s1.fc.bias', 'encoder_s2.fc.weight', 'encoder_s2.fc.bias'}
    assert not unexpected_keys


    print("Checkpoint loaded successfully, loaded S1 and S2 weights.")

    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {num_params:,}")
    # quit()


    return model