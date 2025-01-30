import csv
import datetime
import os
import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, ViTSmall16_Weights, vit_small_patch16_224
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from eval_utils import create_ciip_model, modify_ciip_for_eurosat, load_ciip_model_checkpoint, CustomTransform


transform_pipeline = transforms.Compose([
    transforms.Resize((264, 264)),  # CIIP is 264 and Vit is 224
    # transforms.ToTensor(),         # Seems like TorchGeo is already loadedd as tensor
    transforms.Normalize(mean=[1353.439, 1117.253, 1042.253, 947.128, 1199.404, 2002.936, 2373.488, 2300.642, 732.159, 12.113, 1820.932, 1119.173, 2598.82], 
                              std=[65.571, 154.376, 188.262, 278.926, 228.244, 355.633, 454.901, 530.549, 98.718, 1.187, 378.496, 304.439, 501.747])
])

custom_transform = CustomTransform(transform_pipeline)


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
    #Seed ensures all models see the same k samples per class
    if seed is not None:
        np.random.seed(seed % (2**32))  # Ensure the seed is within the valid range
    
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    features = torch.tensor(features, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    # Training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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


# Few-Shot Comparison Pipeline
def few_shot_comparison_pipeline_optimized(data_path, k_values, models, bands, batch_size=16, num_workers=4, task="classification"):
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

            # Use the optimized few-shot sampler
            # Use a consistent seed for each model and k value
            seed = hash((model_name, k))
            selected_indices = few_shot_sampler_optimized(class_indices_train, k, seed=seed)
            few_shot_train = Subset(transformed_dataset_train, selected_indices)

            dataloader_train = DataLoader(few_shot_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            dataloader_test = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # Use S2 encoder only for CIIP, full models for others
            use_s2_only = model_name == "CIIP_Model"

            train_features, train_labels = extract_features(model, dataloader_train, device, use_s2_only=use_s2_only)
            test_features, test_labels = extract_features(model, dataloader_test, device, use_s2_only=use_s2_only)

            classifier = train_pytorch_classifier(train_features, train_labels, len(np.unique(train_labels)), device)
            accuracy, f1 = evaluate_pytorch_model(classifier, test_features, test_labels, device)
            print(f"    Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

            model_results[k] = {"accuracy": accuracy, "f1": f1}

        results[model_name] = model_results

    return results


# Plot Results
def plot_results(results, k_values, metric="accuracy", output_file="few_shot_results.png"):
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        metric_values = [model_results[k][metric] for k in k_values]
        plt.plot(k_values, metric_values, label=model_name, marker='o')

    plt.xlabel("k (Number of Training Samples per Class)")
    plt.ylabel(metric.capitalize())
    plt.title(f"Few-Shot Training: {metric.capitalize()} vs. k")
    plt.legend()
    plt.grid()
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
    output_file = f"results_{metric}_{timestamp}.csv"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row: "Model Name" followed by k values
        header = ["Model Name"] + [f"k={k}" for k in k_values]
        writer.writerow(header)
        
        # Write rows: model name followed by metric values for each k
        for model_name, model_results in results.items():
            row = [model_name]  # Start the row with the model name
            for k in k_values:
                row.append(model_results[k][metric])  # Add the metric value for each k
            writer.writerow(row)

    print(f"Results saved to {output_file}")




if __name__ == "__main__":
    # DATA_PATH = "./data"
    DATA_PATH = "D:/Code/ciip/data"
    BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12")    

    K_VALUES = [10, 20, 40, 60, 80, 100]

    ciip_checkpoint_path = "D:/Code/ciip/models/bs_512_11-2024.pt"

    # Load CIIP model and modify for EuroSAT
    ciip_model = load_ciip_model_checkpoint(ciip_checkpoint_path)
    ciip_model = modify_ciip_for_eurosat(ciip_model, num_classes=10, freeze_encoder=True)

    models = {
         "CIIP_Model": ciip_model
        # "SSL4EO-vit_small16": vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO)
        # "SSL4EO-ResNet18_MoCo": resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
        # # "SSL4EO-ResNet50_DINO": resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_DINO),
        # "SSL4EO-ResNet50_MoCo": resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
    }

    # Run Few-Shot Comparison
    results = few_shot_comparison_pipeline_optimized(DATA_PATH, K_VALUES, models, BANDS, batch_size=16)
    output_results_to_csv(results, K_VALUES,  metric="accuracy")
    output_results_to_csv(results, K_VALUES,  metric="f1")
    plot_results(results, K_VALUES, metric="accuracy", output_file="few_shot_accuracy.png")
    plot_results(results, K_VALUES, metric="f1", output_file= "few_shot_f1.png")

