import csv
import hashlib
import datetime
import os
import random
import torch
import copy
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, ViTSmall16_Weights, \
    vit_small_patch16_224
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from eval_utils import create_ciip_model, modify_ciip_for_eurosat, load_ciip_model_checkpoint, CustomTransform

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Define the updated band-wise statistics
MEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B08A': 732.08340178,
    'B09': 12.11327804,
    'B10': 1819.01027855,
    'B11': 1118.92391149,
    'B12': 2594.14080798,
}

STD = {
    'B01': 245.71762908,
    'B02': 333.00778264,
    'B03': 395.09249139,
    'B04': 593.75055589,
    'B05': 566.4170017,
    'B06': 861.18399006,
    'B07': 1086.63139075,
    'B08': 1117.98170791,
    'B08A': 404.91978886,
    'B09': 4.77584468,
    'B10': 1002.58768311,
    'B11': 761.30323499,
    'B12': 1231.58581042,
}

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12")

# Generate mean and std lists in the desired order
mean_list = [MEAN[b] for b in BANDS]
std_list = [STD[b] for b in BANDS]

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # CIIP is 264 and Vit is 224
    # transforms.ToTensor(),         # Seems like TorchGeo is already loadedd as tensor
    transforms.Normalize(mean=mean_list, std=std_list), # Normalize with band-wise stats
])

custom_transform = CustomTransform(transform_pipeline)

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


def preprocess_class_indices(dataset):
    class_indices = {}
    for idx, sample in enumerate(dataset):
        label = sample['label']
        # Convert label to a hashable type if necessary
        if isinstance(label, torch.Tensor):
            label = label.item()  # Convert single-element tensors to a scalar
        elif isinstance(label, np.ndarray):
            label = label.tolist()  # Convert numpy arrays to lists
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    return class_indices


def few_shot_sampler_optimized(class_indices, k_shot_per_class, seed=None):
    """
    Sampler for few-shot datasets with fixed random seed for reproducibility.
    """
    if k_shot_per_class == 0:
        # Use all training samples
        selected_indices = []
        for indices in class_indices.values():
            selected_indices.extend(indices)
        return selected_indices

    #Seed ensures all models see the same k samples per class
    if seed is not None:
        np.random.seed(seed % (2 ** 32))  # Ensure the seed is within the valid range

    selected_indices = []
    for label, indices in class_indices.items():
        if len(indices) >= k_shot_per_class:
            chosen_indices = np.random.choice(indices, k_shot_per_class, replace=False)
            selected_indices.extend(chosen_indices)
        else:
            raise ValueError(f"Class '{label}' has fewer samples ({len(indices)}) than k={k_shot_per_class}.")
    return selected_indices


# Classification Head
class FewShotClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FewShotClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Train a Classification Model
def train_pytorch_classifier(features, labels, num_classes, device):
    model = FewShotClassifier(features.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    features = torch.tensor(features, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    # Training loop
    model.train()
    best_loss = float('inf')
    patience = 5
    trigger_times = 0
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Simple early stopping: if loss doesn't improve, break
        if loss.item() < best_loss:
            best_loss = loss.item()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch} with loss {loss.item():.4f}")
                break

    return model


# Evaluate a Classification Model
def evaluate_pytorch_model(model, test_features, test_labels, device):
    model.eval()
    test_features = torch.tensor(test_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(test_features)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")
    return accuracy, f1


def zero_shot_classifier_euclidean(train_features, train_labels, test_features):
    """
    Compute class prototypes (means) from the train_features and assign each test sample
    the label of the nearest prototype (using Euclidean distance).
    """
    prototypes = {}
    for label in np.unique(train_labels):
        prototypes[label] = train_features[train_labels == label].mean(axis=0)
    predictions = []
    for feat in test_features:
        # Compute Euclidean distances to all prototypes.
        distances = {label: np.linalg.norm(feat - proto) for label, proto in prototypes.items()}
        predicted_label = min(distances, key=distances.get)
        predictions.append(predicted_label)
    return np.array(predictions)


def zero_shot_classifier_cosine(train_features, train_labels, test_features):
    """
    Compute class prototypes (means) and classify by highest cosine similarity.
    """
    prototypes = {}
    for label in np.unique(train_labels):
        # Some prefer to normalize each prototype so they're all unit vectors
        proto = train_features[train_labels == label].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[label] = proto

    predictions = []
    for feat in test_features:
        # Optionally normalize the test feature
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        similarities = {
            label: np.dot(feat_norm, proto)
            for label, proto in prototypes.items()
        }
        predicted_label = max(similarities, key=similarities.get)
        predictions.append(predicted_label)
    return np.array(predictions)


# Extract Features for Classification
def extract_features(model, dataloader, device, use_s2_only=False):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            labels = batch["label"]
            # print(f"Image Tensor Shape Sent to Model: {images.shape}")  # Debugging

            if use_s2_only:
                # Use only the S2 encoder for feature extraction
                features = model.encoder_s2(images)
            else:
                # Use the full model if required in other scenarios
                features = model(images)

            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def get_deterministic_seed_for_experiment(k, exp):
    """Generate a seed based solely on k and the experiment index."""
    s = f"{k}_{exp}"
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (2 ** 32)


# Few-Shot Comparison Pipeline
def few_shot_comparison_pipeline_optimized(data_path, k_values, models, bands, batch_size=16, num_workers=4,
                                           task="classification", num_experiments=5):
    transformed_dataset_train = EuroSAT(data_path, split="train", bands=bands, transforms=custom_transform, download=True)
    transformed_dataset_test = EuroSAT(data_path, split="test", bands=bands, transforms=custom_transform, download=True)

    # Preprocess class indices for the training dataset
    class_indices_train = preprocess_class_indices(transformed_dataset_train)

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model in models.items():
        model_results = {}
        model.to(device).eval()

        print(f"Processing model: {model_name}")
        for k in k_values:
            print(f"  Few-shot training with k={k} samples per class")
            accuracies = []
            f1_scores = []

            # For k==0, run only one experiment; otherwise, run multiple experiments.
            num_exp = num_experiments if k > 0 else 1
            for exp in range(num_exp):
                # Use the optimized few-shot sampler
                # Use a consistent seed for each model and k value
                seed = get_deterministic_seed_for_experiment(k, exp)
                selected_indices = few_shot_sampler_optimized(class_indices_train, k, seed=seed)
                few_shot_train = Subset(transformed_dataset_train, selected_indices)

                dataloader_train = DataLoader(few_shot_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                dataloader_test = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

                # Use S2 encoder only for CIIP, full models for others
                use_s2_only = model_name == "CIIP_Model"

                # For zero-shot (k==0), use the full model (with fc)
                # For few-shot (k > 0), create a deep copy and drop the last linear layer.
                if k == 0:
                    current_model = model  # full model with projection head
                else:
                    current_model = copy.deepcopy(model)
                    current_model = drop_last_linear_layer(current_model)

                train_features, train_labels = extract_features(current_model, dataloader_train, device, use_s2_only=use_s2_only)
                test_features, test_labels = extract_features(current_model, dataloader_test, device, use_s2_only=use_s2_only)

                if k == 0:
                    # Zero-shot: use nearest-prototype classification on features.
                    predictions = zero_shot_classifier_cosine(train_features, train_labels, test_features)
                    accuracy = accuracy_score(test_labels, predictions)
                    f1 = f1_score(test_labels, predictions, average="weighted")
                else:

                    classifier = train_pytorch_classifier(train_features, train_labels, len(np.unique(train_labels)),
                                                          device)
                    accuracy, f1 = evaluate_pytorch_model(classifier, test_features, test_labels, device)

                accuracies.append(accuracy)
                f1_scores.append(f1)
            # Aggregate across experiments
            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
            f1_mean = np.mean(f1_scores)
            f1_std = np.std(f1_scores) if len(f1_scores) > 1 else 0.0
            model_results[k] = {"accuracy_mean": acc_mean, "accuracy_std": acc_std, "accuracy_raw": accuracies,
                                "f1_mean": f1_mean, "f1_std": f1_std, "f1_raw": f1_scores}
            print(f"  k={k}: Accuracy = {acc_mean:.4f} ± {acc_std:.4f}, F1 = {f1_mean:.4f} ± {f1_std:.4f}")
        results[model_name] = model_results
    return results


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

    plt.xlabel("k (Number of Training Samples per Class)")
    plt.ylabel(metric.capitalize())
    plt.title(f"Few-Shot Evaluation: {metric.capitalize()} vs. k")
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def output_results_to_csv(results, k_values, metric="accuracy"):
    """
    Write the results to a CSV file with model names as rows and k values as columns.
    Each cell contains the metric value (accuracy or F1 score).
    The filename includes the timestamp.
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"./few_shot_eval/results_{metric}_{timestamp}.csv"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header row: "Model Name" followed by k values
        header = ["Model Name"] + [f"k={k}" for k in k_values]
        writer.writerow(header)

        # Write rows: model name followed by metric values for each k
        for model_name, model_results in results.items():
            row = [model_name]  # Start the row with the model name
            for k in k_values:
                row.append(model_results[k][f'{metric}_mean'])  # Add the metric value for each k
            writer.writerow(row)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # DATA_PATH = "./data"
    DATA_PATH = "./eurosat_data"
    BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12")

    K_VALUES = [0, 1, 2, 4, 8, 16, 32]
    # K_VALUES = [0, 10, 20, 40, 60, 80, 100]

    ciip_checkpoint_path = "/local/ms-data/SSL4EO/model/2025_03_18-17_19_03-model_resnet50-lr_0.0001-b_128-j_6-p_fp16/epoch_85.pt"

    # Load CIIP model and modify for EuroSAT
    ciip_model = load_ciip_model_checkpoint(ciip_checkpoint_path)
    # ciip_model = modify_ciip_for_eurosat(ciip_model, num_classes=10, freeze_encoder=True)

    models = {
        "CIIP_Model": ciip_model,
        # "SSL4EO-vit_small16": vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO)
        "SSL4EO-ResNet18_MoCo": resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),
        "SSL4EO-ResNet50_DINO": resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_DINO),
        "SSL4EO-ResNet50_MoCo": resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
    }

    for model in models:
        for param in models[model].parameters():
            param.requires_grad = False

    # Run Few-Shot Comparison
    results = few_shot_comparison_pipeline_optimized(DATA_PATH, K_VALUES, models, BANDS,
                                                     batch_size=16, num_experiments=10)
    output_results_to_csv(results, K_VALUES, metric="accuracy")
    output_results_to_csv(results, K_VALUES, metric="f1")
    plot_results(results, K_VALUES, metric="accuracy", output_file="few_shot_eval/few_shot_accuracy.png")
    plot_results(results, K_VALUES, metric="f1", output_file="few_shot_eval/few_shot_f1.png")
