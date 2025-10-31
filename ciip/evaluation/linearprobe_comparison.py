import csv
import hashlib
import os
import copy
import json
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils import *


MEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B09': 732.08340178,
    'B10': 12.11327804,
    'B11': 1819.01027855,
    'B12': 1118.92391149,
    'B8A': 2594.14080798,
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
    'B09': 404.91978886,
    'B10': 4.77584468,
    'B11': 1002.58768311,
    'B12': 761.30323499,
    'B8A': 1231.58581042,
}

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


# Generate mean and std lists in the desired order
mean_list = [MEAN[b] for b in BANDS]
std_list = [STD[b] for b in BANDS]


data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean_list, std=std_list),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean_list, std=std_list),
    ]),
}

train_transform = CustomTransform(data_transforms["train"])
val_transform = CustomTransform(data_transforms["val"])

def drop_last_linear_layer(model):
    """Replace the model's final linear layer with an identity function."""
    logger = logging.getLogger(__name__)
    if hasattr(model, "encoder_s2") and hasattr(model.encoder_s2, "fc"):
        logger.info("Dropping %s", model.encoder_s2.fc)
        model.encoder_s2.fc = nn.Identity()
    elif hasattr(model, "fc"):
        logger.info("Dropping %s", model.fc)
        model.fc = nn.Identity()
    else:
        logger.warning("No final linear layer found to drop")
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


class FewShotClassifierLightning(pl.LightningModule):
    def __init__(self, in_features, num_classes, learning_rate=0.001): # 1. Add learning_rate to __init__
        super().__init__()
        self.model = FewShotClassifier(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate 

        self.save_hyperparameters(ignore=['model', 'criterion']) 

        # Store predictions and labels for epoch-end calculation
        self.validation_predictions = []
        self.validation_labels = []
        self.test_predictions = []
        self.test_labels = []
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        # _logger.info(
        #     f"Epoch {self.current_epoch}, Step {self.global_step} - Train Loss: {loss.item():.4f}"
        # )
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.002)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # # Warmup scheduler: Linear increase
        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=0.3333, end_factor=1.0, total_iters=5
        # )

        # # Cosine annealing scheduler
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=300-5, eta_min=0.00005 # T_max needs to account for warmup epochs
        # )

        # # Sequential scheduler: Warmup then Cosine
        # # Milestones here refer to *epochs* where the scheduler switches
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
        # )

        # Return dict required by Lightning for multiple schedulers or specific configurations
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', # 'epoch' means step every epoch, 'step' means step every batch
                'frequency': 1,
            }
        }

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        # _logger.info(
        #     f"Epoch {self.current_epoch}, Step {self.global_step} - Val Loss: {loss.item():.4f}"
        # )


        # Store predictions and true labels for metric calculation at epoch end
        predictions = torch.argmax(outputs, dim=1)
        self.validation_predictions.append(predictions.cpu())
        self.validation_labels.append(labels.cpu())
        return loss

 
    def on_validation_epoch_end(self):
        # Concatenate all predictions and labels from the epoch
        all_predictions = torch.cat(self.validation_predictions).numpy()
        all_labels = torch.cat(self.validation_labels).numpy()

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        avg_val_loss = self.trainer.callback_metrics.get('val_loss', float('nan'))
        avg_train_loss = self.trainer.callback_metrics.get('train_loss', float('nan'))

        if isinstance(avg_train_loss, torch.Tensor):
            avg_train_loss = avg_train_loss.item()
        if isinstance(avg_val_loss, torch.Tensor):
            avg_val_loss = avg_val_loss.item()

        self.train_losses.append(avg_train_loss)
        self.val_losses.append(avg_val_loss)

        _logger.info(
            f"Epoch {self.current_epoch} End - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}, Val F1-Score: {f1:.4f}"
        )

        # Log metrics
        self.log_dict({
            'val_accuracy': accuracy,
            'val_f1_score': f1,
        }, on_epoch=True, prog_bar=False)

        # Clear lists for the next epoch/run
        self.validation_predictions.clear()
        self.validation_labels.clear()
    
    def test_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        # Store predictions and true labels for metric calculation at epoch end
        predictions = torch.argmax(outputs, dim=1)
        self.test_predictions.append(predictions.cpu())
        self.test_labels.append(labels.cpu())
        return loss

    def on_test_epoch_end(self):
        # Concatenate all predictions and labels from the epoch
        all_predictions = torch.cat(self.test_predictions).numpy()
        all_labels = torch.cat(self.test_labels).numpy()

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        avg_test_loss = self.trainer.callback_metrics.get('test_loss', float('nan')) # Get test_loss if you log it in test_step
        _logger.info(
            f"Epoch {self.current_epoch} Test End - Test Loss: {avg_test_loss:.4f}, "
            f"Test Accuracy: {accuracy:.4f}, Test F1-Score: {f1:.4f}"
        )

        # Log metrics
        self.log_dict({
            'test_accuracy': accuracy,
            'test_f1_score': f1,
        }, on_epoch=True, prog_bar=False)

        # Clear lists for the next epoch/run
        self.test_predictions.clear()
        self.test_labels.clear()

        # Optional: return a dictionary of metrics if you need them externally
        return {'test_accuracy': accuracy, 'test_f1_score': f1}



# Classification Head
class FewShotClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FewShotClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



def train_pytorch_classifier(features, labels, val_features, val_labels, test_features, test_labels, num_classes, device, classifier_batch_size=256, model_name='None', num_workers=8, output_dir=None, plot_name=None):
    # Training Data
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(features_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=classifier_batch_size, shuffle=True, num_workers=num_workers)

    # validation
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=classifier_batch_size, shuffle=False, num_workers=num_workers)  # No shuffle for validation data

    
    # Test Data
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=classifier_batch_size, shuffle=False, num_workers=num_workers) # No shuffle for test data

    #######  FIND OPTIMAL LEARNING RATE #####33
    print("  Running Learning Rate Finder...")

    probe_learning_rate = None

    lr_finder_trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=1, # LR Finder typically runs for 1 epoch or less
        log_every_n_steps=10000,
        enable_checkpointing=False, # No need to save checkpoints for LR find
        logger=False, # No need for full logging for LR find
        enable_model_summary=False
    )

    lr_finder_model = FewShotClassifierLightning(features.shape[1], num_classes)
    tuner = pl.tuner.Tuner(lr_finder_trainer)

    # Run the LR Finder
    lr_finder = tuner.lr_find(
        lr_finder_model,
        train_dataloaders=train_loader, # Use the actual probe train loader
        val_dataloaders=val_loader,     # Use the actual probe val loader
        num_training=100,         # Number of steps to run LR finder for
        max_lr=0.01,                      # Max LR to test. Adjust if you expect much higher/lower.
        min_lr=0.0001
    )

    # Suggests a learning rate. You might want to take something slightly before the minimum.
    
    probe_learning_rate = lr_finder.suggestion()
    print(f"Suggested LR for {model_name}: {probe_learning_rate}")
    _logger.info(f"Suggested LR for {model_name}: {probe_learning_rate}")

    if probe_learning_rate is None:
        probe_learning_rate = 0.00005
        print(f"Falling back on LR for {model_name}: {probe_learning_rate}")
        _logger.info(f"Falling back on LR for {model_name}: {probe_learning_rate}")



    # Instantiate model with the optimal learning rate
    lightning_model = FewShotClassifierLightning(features.shape[1], num_classes, learning_rate=probe_learning_rate)


    # --- Callbacks ---
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        save_weights_only=False,
        filename='best-checkpoint-epoch{epoch:02d}-val_acc{val_accuracy:.4f}',
        verbose=False
    )

    # model_summary_callback = ModelSummary(max_depth=2) # Adjust max_depth as needed


    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=300,                   # Your desired total number of epochs
        accelerator='auto',               # 'cpu', 'gpu', or 'auto'
        devices=1, # Or specify [0, 1] for multiple GPUs
        # logger=logger,                    # Attach the TensorBoard logger
        callbacks=[
            early_stopping_callback,      # Add EarlyStopping
            # model_summary_callback,        # Add ModelSummary
            checkpoint_callback
        ],
        log_every_n_steps=50,             # Log every N training steps
        enable_model_summary=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=1
    )


    #######  TRAIN AND TEST ######

    print("Starting training...")
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    print("Training finished.")

    if output_dir is not None:
        epochs = range(1, len(lightning_model.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, lightning_model.train_losses, label="Train Loss")
        plt.plot(epochs, lightning_model.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Loss")
        plt.legend()
        plot_path = os.path.join(output_dir, plot_name or f"{model_name}_train_val_loss.png")
        plt.savefig(plot_path)
        plt.close()

    # --- Load best model ---
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    _logger.info(f"Loading best model from: {best_model_path}")
    best_model = FewShotClassifierLightning.load_from_checkpoint(best_model_path)

    # --- Evaluation ---
    print("Starting evaluation...")
    results = trainer.test(best_model, test_loader)
    print("Evaluation finished.")


    if results:
        final_accuracy = results[0].get('test_accuracy')
        final_f1 = results[0].get('test_f1_score')
        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        print(f"Final Test F1-Score (weighted): {final_f1:.4f}")

    return best_model.model, results 



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
                # print(f'Shape of images sent to S2 encoder: {images.shape}')
                # features = model.encoder_s2(images)
                features = model.encode_s2(s2=images, normalize=True)
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
def few_shot_comparison_pipeline_optimized(data_path, percents, models, bands, batch_size=16, num_workers=18,
                                           task="classification", num_experiments=10, output_dir=None, **kwargs):

    print('******************* NEW FEW-SHOT COMPARISON PIPELINE *******************')

   
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    transformed_dataset_train = EuroSAT(data_path, split="train", bands=BANDS, transforms=train_transform, download=True)
    transformed_dataset_val = EuroSAT(data_path, split="val", bands=BANDS, transforms=val_transform, download=True)
    transformed_dataset_test = EuroSAT(data_path, split="test", bands=BANDS, transforms=val_transform, download=True)

    print(f"Number of training samples: {len(transformed_dataset_train)}")
    print(f"Number of validation samples: {len(transformed_dataset_val)}")
    print(f"Number of test samples: {len(transformed_dataset_test)}")


    # Preprocess class indices for the training dataset
    # class_indices_train = preprocess_class_indices(transformed_dataset_train)

    for model_config in models:
        model = load_model(model_config["type"], model_config["weights"])
        for param in model.parameters():
            param.requires_grad = False

        model_name = model_config["name"]

        
        model_results = {}
        model.to(device).eval()
        current_model = copy.deepcopy(model)
        
        if hasattr(model, 'encoder_s2') and hasattr(model.encoder_s2, 'fc'):
            # Replace the fc layer of encoder_s2 with an identity mapping.
            _logger.info(f'{model_name} has fc of {model.encoder_s2.fc}')
        elif hasattr(model, 'fc'):
            _logger.info(f'{model_name} has fc of {model.fc}')
        else:
            _logger.info(f'{model_name} has no fc.')
        if model_config.get('drop_last_layer'):
            current_model = drop_last_linear_layer(current_model)
        # continue
            
        # print(current_model)
        current_model = current_model.to(device)
        current_model.eval()

        print(f"Processing model: {model_name}")
        for i, percent in enumerate(percents):
            accuracies = []
            f1_scores = []
            print(f" linear probe training with {percent} of training data")
            num_exp = num_experiments 
            for exp in range(num_exp):
                seed = get_deterministic_seed_for_experiment(i, exp)

                num_samples = int(percent * len(transformed_dataset_train))
                length_of_training_data = len(transformed_dataset_train)
                rng = np.random.default_rng(seed)
                selected_indices = rng.choice(length_of_training_data, num_samples, replace=False)

                train_data = Subset(transformed_dataset_train, selected_indices)

                dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                dataloader_val = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)
                dataloader_test = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False,
                                                num_workers=num_workers)

                # Use S2 encoder only for CIIP
                use_s2_only = (model.__class__.__name__ == "CIIP")
                if use_s2_only:
                    print(f"Using S2 encoder only for model: {model_name}")

                
                train_features, train_labels = extract_features(current_model, dataloader_train, device, use_s2_only=use_s2_only)
                val_features, val_labels = extract_features(current_model, dataloader_val, device, use_s2_only=use_s2_only)
                test_features, test_labels = extract_features(current_model, dataloader_test, device, use_s2_only=use_s2_only)

                if NORMALIZE:
                    train_features = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
                    test_features = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)
         

                # train linear classifier
                plot_filename = f"{model_name}_p{str(percent).replace('.', 'p')}_exp{exp}_loss.png"
                classifier, trainer_results = train_pytorch_classifier(
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    test_features,
                    test_labels,
                    len(np.unique(train_labels)),
                    device,
                    classifier_batch_size=batch_size,
                    model_name=model_name,
                    num_workers=num_workers,
                    output_dir=output_dir,
                    plot_name=plot_filename,
                )

                if trainer_results:
                    final_accuracy = trainer_results[0].get('test_accuracy')
                    final_f1 = trainer_results[0].get('test_f1_score')
                    print(f"Final Test Accuracy: {final_accuracy:.4f}")
                    print(f"Final Test F1-Score (weighted): {final_f1:.4f}")

                accuracies.append(final_accuracy)
                f1_scores.append(final_f1)


            # Aggregate across experiments
            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
            f1_mean = np.mean(f1_scores)
            f1_std = np.std(f1_scores) if len(f1_scores) > 1 else 0.0
            model_results[percent] = {"accuracy_mean": acc_mean, "accuracy_std": acc_std, "accuracy_raw": accuracies,
                                "f1_mean": f1_mean, "f1_std": f1_std, "f1_raw": f1_scores}
            print(f"  percent={percent}: Accuracy = {acc_mean:.4f} ± {acc_std:.4f}, F1 = {f1_mean:.4f} ± {f1_std:.4f}")

        results[model_name] = model_results
    # raise ValueError
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

    plt.xlabel("Percent of Training data")
    plt.ylabel(metric.capitalize())
    plt.title(f"Linear Probe with Classification Head : {metric.capitalize()} vs. k")
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
        header = ["Model Name"] + [f"percent={k}" for k in k_values] + [f"Std percent={k}" for k in k_values]
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
            model = resnet50(weights=None, num_classes=10, in_chans=13, pretrained=False)
            print(f"First layer shape: {model.conv1.weight.shape}")
        else:    
            model = resnet50(weights_path)
            # print first layer shape
            print(f"First layer shape: {model.conv1.weight.shape}")
    elif model_type == "ciip":
        model = load_ciip_model_checkpoint(weights_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def generate_ciip_model_configs(model_root: str, base_name: str, epochs: list = None,
                                drop_last_layer: bool = True):
    """Generate configuration dictionaries for CIIP checkpoints."""
    configs = {}
    for epoch in epochs:
        model_name = f"{base_name}-epoch{epoch}"
        configs[model_name] = {
            "type": "ciip",
            "weights": os.path.join(model_root, f"epoch_{epoch}.pt"),
            "drop_last_layer": drop_last_layer,
        }
    return configs


def discover_checkpoints(
    checkpoint_root: Path,
    *,
    pattern: Optional[str],
    include_init: bool,
    max_checkpoints: Optional[int],
) -> List[Path]:
    """Return checkpoint files sorted using natural ordering."""

    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root '{checkpoint_root}' does not exist")

    regex = re.compile(pattern) if pattern else None
    candidates: List[Path] = []
    for extension in ("*.pt", "*.pth"):
        candidates.extend(checkpoint_root.glob(extension))

    if regex is not None:
        candidates = [path for path in candidates if regex.search(path.name)]
    if not include_init:
        candidates = [path for path in candidates if "epoch_init" not in path.name]


    epochs = [int(p.stem.split('_')[1]) for p in candidates]
    epochs = sorted(set(epochs))

    # if min epoch is > 5
    subtract = False
    if min(epochs) > 5:
        subtract = True
        minimum = min(epochs)
        # subtract min from all
        epochs = [e - minimum for e in epochs]

    keep = set()

    # Always keep these if present
    keep.update(e for e in epochs if e in {1, 2, 3, 5, 10, 15, 20})

    if epochs:
        max_ep = max(epochs)

        # If max epoch < 50, keep it (even if not in the above list)
        if max_ep < 50:
            keep.add(max_ep)

        # If max epoch is between 20 and 100 (inclusive),
        # keep only every 10th epoch AFTER 20 (i.e., 30, 40, 50, ... up to max_ep)
        if 20 < max_ep <= 100:
            keep.update(e for e in epochs if e > 20 and e % 10 == 0 and e <= max_ep)

        if 100 < max_ep:
            keep.update(e for e in epochs if e > 20 and e % 10 == 0 and e <= 50)
            keep.update(e for e in epochs if e > 50 and e % 20 == 0)


    epochs = sorted(keep)

    return epochs


def parse_results_from_csv(csv_file, metric="accuracy"):
    """
    Reads a CSV file produced by output_results_to_csv() and reconstructs
    the original results dictionary:
    
    {
        "model_name": {
            k_value: {f"{metric}_mean": ..., f"{metric}_std": ...},
            ...
        },
        ...
    }
    """
    results = {}

    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Extract k values from the header row
        # The header format is: ["Model Name", "percent=...", ..., "Std percent=..."]
        k_values = [float(h.split('=')[1]) for h in header[1:1 + (len(header)-1)//2]]

        for row in reader:
            model_name = row[0]
            results[model_name] = {}

            # First half are means, second half are stds
            mean_values = [float(v) for v in row[1:1 + len(k_values)]]
            std_values = [float(v) for v in row[1 + len(k_values):1 + 2*len(k_values)]]

            for k, mean, std in zip(k_values, mean_values, std_values):
                results[model_name][k] = {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std
                }

    return results
    
if __name__ == "__main__":

    MODEL_ROOT = "/local/ms-data/SSL4EO/model"

    # Generate CIIP model configs for every 5 epochs
    # ciip_root = f"{MODEL_ROOT}/2025_08_06-MoCoInit/no-copy"
    # ciip_root = f"/home/juro4948/ciip/logs/2025_09_05-13_28_50-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints"
    # epochs = [1, 2, 5, 10, 20, 50, 80, 100, 120,140]
    # MODEL_CONFIGS = generate_ciip_model_configs(
    #     model_root=ciip_root,
    #     base_name="2025_09_05_MoCoInit-hal",
    #     epochs=epochs,
    # )
    # ciip_root = f"/home/juro4948/ciip/logs/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints"
    ciip_root = '/home/juro4948/ciip/logs/2025_09_10-11_37_00-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints'
    epochs = [1, 2, 3, 5, 10, 15, 20, 30]
    base_name="2025_09_10_RandomInit-hal-bs5120"
    MODEL_CONFIGS = generate_ciip_model_configs(
        model_root=ciip_root,
        base_name=base_name,
        epochs=epochs,
    )


    # Add any additional non-CIIP models here
    MODEL_CONFIGS.update({
        "SSL4EO-ResNet50_MoCo": {
            "type": "resnet50",
            "weights": ResNet50_Weights.SENTINEL2_ALL_MOCO,
            'drop_last_layer': True
        },
        "SSL4EO-ResNet50_DINO": {   
            "type": "resnet50",
            "weights": ResNet50_Weights.SENTINEL2_ALL_DINO,
            'drop_last_layer': True
        },
        "ResNet50_Random": {
            "type": "resnet50",
            'weights': None,
            'drop_last_layer': True
        },
    })

    CONFIG = {
        "data_path": "/local/ms-data/EuroSAT/",  # Path to EuroSAT dataset
        "percents": [0.1, 1],  # [0.1, 1],
        "batch_size": 512,
        "num_workers": 8,
        "num_experiments": 1,
        "bands": BANDS,  # assuming BANDS is defined elsewhere
        # Evaluate all generated CIIP checkpoints plus extra models
        "models": list(MODEL_CONFIGS.keys()),
    }

    #  '2025_8_3_RandomInit-deltaAI-epoch40', '2025_8_3_RandomInit-deltaAI-epoch30', '2025_8_3_RandomInit-deltaAI-epoch20', '2025_8_3_RandomInit-deltaAI-epoch10'


    # add model dict to config
    NORMALIZE = False

    models = []
    model_info = []
    for model_name, config in MODEL_CONFIGS.items():
        if model_name not in CONFIG["models"]:
            continue
        model_info.append({
            "name": model_name,
            "type": config["type"],
            "weights": config["weights"],
            "weights_str": str(config["weights"]),
            "normalized": NORMALIZE,
            'drop_last_layer': config['drop_last_layer']
            
        })

    CONFIG["models"] = model_info
    CONFIG['notes'] = ' ' 

    ## Setup Logging and Output Directory ##
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    experiment_name = f"{timestamp}"
    output_dir = os.path.join("results", "linearprobe-clf", f"normalized-{NORMALIZE}", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    log_filepath = os.path.join(output_dir, "training_log.txt")
    custom_logger = logging.getLogger(__name__)
    custom_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not custom_logger.handlers:
        custom_logger.addHandler(file_handler)

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    _logger = custom_logger

    results = few_shot_comparison_pipeline_optimized(output_dir=output_dir, **CONFIG)
    print(results)
    _logger.info(results)

    # Save metrics
    output_results_to_csv(results, CONFIG["percents"], metric="accuracy", output_dir=output_dir)
    output_results_to_csv(results, CONFIG["percents"], metric="f1", output_dir=output_dir)

    # Save plots
    plot_results(results, CONFIG["percents"], metric="accuracy", output_file=os.path.join(output_dir, "linearprobe_accuracy.png"))
    plot_results(results, CONFIG["percents"], metric="f1", output_file=os.path.join(output_dir, "linearprobe_f1.png"))

    plot_primary_over_epochs(results, CONFIG["percents"], metric="accuracy", primary_regex=r"RandomInit-hal-epoch(\d+)", 
        baselines=("SSL4EO-ResNet50_MoCo","SSL4EO-ResNet50_DINO","ResNet50_Random"),
        layout="facets",                         # or "single"
        title=f"{base_name} Linear probe clf head accuracy vs epoch",
        output_file=os.path.join(output_dir, "svm_accuracy_per_epoch.png"))
    plot_primary_over_epochs(results, CONFIG["percents"], metric="f1", primary_regex=r"RandomInit-hal-epoch(\d+)",
        baselines=("SSL4EO-ResNet50_MoCo","SSL4EO-ResNet50_DINO","ResNet50_Random"),
        layout="facets",                         # or "single"
        title=f"{base_name} Linear probe clf head f1 vs epoch",
        output_file=os.path.join(output_dir, "svm_f1_per_epoch.png"))

    _logger.info(f'Saved results to {output_dir}')
    print(f'Saved results to {output_dir}')

    SAVE_CONFIG = CONFIG
    SAVE_CONFIG['models'] = []
    try:
        for model_info in CONFIG['models']:
            if "weights" in model_info:
                del model_info["weights"]
            SAVE_CONFIG['models'].append(model_info)
    except:
        pass
        
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(SAVE_CONFIG, f, indent=4)
