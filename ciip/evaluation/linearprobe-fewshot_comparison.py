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
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, ViTSmall16_Weights, \
    vit_small_patch16_224
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import itertools 

# from model import ResNet50
import json
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from sklearn.model_selection import train_test_split
import logging

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
        'train':  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(mean=mean_list, std=std_list),
        ]),'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(mean=mean_list, std=std_list),
        ]),
    }
train_transform = CustomTransform(data_transforms['train'])
val_transform = CustomTransform(data_transforms['val'])


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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.002)

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
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'epoch', # 'epoch' means step every epoch, 'step' means step every batch
            #     'frequency': 1,
            # }
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

def train_pytorch_classifier(
    features,
    labels,
    test_features,
    test_labels,
    num_classes,
    device,
    classifier_batch_size=256,
    lr=None,
    epochs=None,
    model_name='None',
    k_value=None, # Pass the current k value to the function
    random_seed=42 # To ensure deterministic splits for the probe's data
):

    # Training Data for the Linear Probe
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(features_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=classifier_batch_size, shuffle=True)

    # Test Data for Final Evaluation
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=classifier_batch_size, shuffle=False) # No shuffle for test data
    
    callbacks = []
    val_loader = None


    # # --- Conditional Validation Setup ---
    # val_loader = None
    # 
    # min_k_for_probe_val = 30 # Threshold for using a validation set for the probe

    # probe_learning_rate = 0.001 # starting lr

    # if k_value is not None and k_value > min_k_for_probe_val:
    #     print(f"k_value ({k_value}) is > {min_k_for_probe_val}. Attempting to create validation set for the linear probe.")
    #     try:
    #         # Split the provided features/labels (which are already the 'k' samples)
    #         # into training and validation for the probe
    #         X_probe_train, X_probe_val, y_probe_train, y_probe_val = train_test_split(
    #             features, labels,
    #             test_size=0.20, # 20% for validation of the probe's training
    #             random_state=random_seed,
    #             stratify=labels # Stratify to maintain class distribution
    #         )

    #         val_features_tensor = torch.tensor(X_probe_val, dtype=torch.float32)
    #         val_labels_tensor = torch.tensor(y_probe_val, dtype=torch.long)
    #         val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    #         val_loader = DataLoader(val_dataset, batch_size=classifier_batch_size, shuffle=False)

    #         # Re-create train_loader with the actual probe training split
    #         train_features_tensor = torch.tensor(X_probe_train, dtype=torch.float32)
    #         train_labels_tensor = torch.tensor(y_probe_train, dtype=torch.long)
    #         train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    #         train_loader = DataLoader(train_dataset, batch_size=classifier_batch_size, shuffle=True)

    #         print(f"  Probe training size: {len(X_probe_train)} | Probe validation size: {len(X_probe_val)}")
    #         print("  Running Learning Rate Finder...")

    #         lr_finder_trainer = pl.Trainer(
    #             accelerator='auto',
    #             devices=1,
    #             max_epochs=1, # LR Finder typically runs for 1 epoch or less
    #             log_every_n_steps=1000,
    #             enable_checkpointing=False, # No need to save checkpoints for LR find
    #             logger=False, # No need for full logging for LR find
    #             enable_model_summary=False
    #         )

    #         lr_finder_model = FewShotClassifierLightning(features.shape[1], num_classes)
    #         tuner = pl.tuner.Tuner(lr_finder_trainer)

    #         # Run the LR Finder
    #         lr_finder = tuner.lr_find(
    #             lr_finder_model,
    #             train_dataloaders=train_loader, # Use the actual probe train loader
    #             val_dataloaders=val_loader,     # Use the actual probe val loader
    #             num_training=100,         # Number of steps to run LR finder for
    #             max_lr=0.01,                      # Max LR to test. Adjust if you expect much higher/lower.
    #             min_lr=0.00001
    #         )

    #         # # Suggests a learning rate. You might want to take something slightly before the minimum.
    #         # probe_learning_rate = lr_finder.suggestion()
    #         # print(f"Suggested LR for k={k_value}: {probe_learning_rate}")
    #         # _logger.info(f"Suggested LR for k={k_value}: {probe_learning_rate}")

    #         # if probe_learning_rate is None:
    #         #     probe_learning_rate = 0.0005
    #         #     print(f"Falling back on LR for {model_name}: {probe_learning_rate}")
    #         #     _logger.info(f"Falling back on LR for {model_name}: {probe_learning_rate}")

    #         checkpoint_callback = ModelCheckpoint(
    #             monitor='val_loss',
    #             mode='min',
    #             save_top_k=1,
    #             save_weights_only=False,
    #             filename='best-lp-epoch{epoch:02d}-val_loss{val_loss:.4f}',
    #             verbose=False
    #         )
    #         callbacks.append(checkpoint_callback)
    #         best_model_selection_metric = 'val_loss'

    #     except ValueError as e:
    #         print(f"  Warning: Could not create stratified validation split for k={k_value} ({e}). Proceeding without probe validation.")
    #         # Fallback to no validation, so no val_loader and no specific callbacks
    #         checkpoint_callback = ModelCheckpoint(
    #             save_top_k=0,
    #             save_last=True,
    #             filename='last-lp-epoch{epoch:02d}',
    #             verbose=False
    #         )
    #         callbacks.append(checkpoint_callback)
    #         best_model_selection_metric = 'last'
    # else:
        # print(f"k_value ({k_value}) is <= {min_k_for_probe_val}. Training linear probe without a separate validation set.")
        # Only save the last model if no validation is used

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename='last-lp-epoch{epoch:02d}',
        verbose=False
    )
    callbacks.append(checkpoint_callback)
    best_model_selection_metric = 'last'



    # Instantiate model with the optimal learning rate
    lightning_model = FewShotClassifierLightning(features.shape[1], num_classes, learning_rate=lr)

    
    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        # logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1000,
        enable_progress_bar=False,
        enable_model_summary=False,
        check_val_every_n_epoch=10
    )


    # --- Training ---
    print("Starting linear probe training...")
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader # This will be None if no validation is used
    )
    print("Linear probe training finished.")

    # --- Load best model for evaluation ---
    best_model_path = None
    if best_model_selection_metric == 'val_loss' and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from validation: {best_model_path}")
        _logger.info(f"Loading best model from: {best_model_path}")
    elif best_model_selection_metric == 'last' and hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path:
        best_model_path = checkpoint_callback.last_model_path
        print(f"Loading last model checkpoint: {best_model_path}")
        _logger.info(f"Loading last model checkpoint: {best_model_path}")

    else:
        print("No specific best model checkpoint saved; using the model trained in memory (last epoch).")
        # In this case, best_model will be lightning_model directly
    
    if best_model_path:
        best_model = FewShotClassifierLightning.load_from_checkpoint(best_model_path)
    else:
        best_model = lightning_model # Use the model that finished training

    # --- Evaluation ---
    print("Starting final evaluation on test set...")
    best_model.to(device) # Ensure model is on the correct device for testing
    results = trainer.test(best_model, test_loader)
    print("Final evaluation finished.")

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
                features = model.encoder_s2(images)
            else:
                # Use the full model if required in other scenarios
                features = model(images)

            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def sample_hyperparams():
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [16, 32, 64, 128]
    num_epochs = [200]

    # Create all combinations
    hyperparam_grid = []
    for lr, batch_size, epochs in itertools.product(learning_rates, batch_sizes, num_epochs):
        hyperparam_grid.append({
            "lr": lr,
            "classifier_batch_size": batch_size,
            "num_epochs": epochs
        })

    return hyperparam_grid


# Few-Shot Comparison Pipeline
def few_shot_comparison_pipeline_optimized(data_path, k_values, models, bands, num_workers=8,
                                           task="classification", num_experiments=10, zeroshot_similarity='cosine'):

    results = {}
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(device)

    hyperparam_grid = sample_hyperparams()

    transformed_dataset_train = EuroSAT(data_path, split="train", bands=BANDS, transforms=train_transform, download=True)
    transformed_dataset_val = EuroSAT(data_path, split="val", bands=BANDS, transforms=val_transform, download=True)
    transformed_dataset_test = EuroSAT(data_path, split="test", bands=BANDS, transforms=val_transform, download=True)

    # add val to train dataset
    transformed_dataset_train = torch.utils.data.ConcatDataset([transformed_dataset_train, transformed_dataset_val])

    for model_config in models:
        model = load_model(model_config["type"], model_config["weights"])
        for param in model.parameters():
            param.requires_grad = False

        model_name = model_config["name"]

        
        model_results = {}
        model.to(device).eval()
        current_model = copy.deepcopy(model)
        current_model = drop_last_linear_layer(current_model)
        current_model = current_model.to(device)
        current_model.eval()

        print(f"\nProcessing model: {model_name}")

        dataloader_train = DataLoader(transformed_dataset_train, batch_size=64, shuffle=True, num_workers=num_workers)
        dataloader_test = DataLoader(transformed_dataset_test, batch_size=64, shuffle=False, num_workers=num_workers)

        # Use S2 encoder only for CIIP
        use_s2_only = (model.__class__.__name__ == "CIIP")
        if use_s2_only:
            print(f"Using S2 encoder only for model: {model_name}")

        train_features, train_labels = extract_features(current_model, dataloader_train, device, use_s2_only=use_s2_only)
        test_features, test_labels = extract_features(current_model, dataloader_test, device, use_s2_only=use_s2_only)

        # Zero-Shot and Few-Shot Loop
        for i, k in enumerate(k_values):

            _logger.info('\n')
            _logger.info(f' ******* STARTING TRAINING WITH K={k} ON {model_name} ******* ')


            accuracies = []
            f1_scores = []
            euc_acc = []
            euc_f1 = []
            num_exp = num_experiments

            if k == 0:
                print(f"Performing zero-shot evaluation for model: {model_name}")
                unique_labels = np.unique(train_labels)
                class_prototypes = {}
                for label in unique_labels:
                    # Calculate mean feature vector for each class
                    class_prototypes[label] = np.mean(train_features[train_labels == label], axis=0)

                # Convert prototypes to a torch tensor for batch processing
                prototype_tensor = torch.tensor(list(class_prototypes.values()), dtype=torch.float32).to(device)
                unique_labels_sorted = sorted(list(class_prototypes.keys()))
                label_to_idx = {label: i for i, label in enumerate(unique_labels_sorted)}

                # Evaluate on test features
                test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
                test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)


                # --- Euclidean Similarity ---
                diff_euclidean = test_features_tensor.unsqueeze(1) - prototype_tensor.unsqueeze(0)
                distances_euclidean = torch.norm(diff_euclidean, dim=2, p=2) # L2 norm (Euclidean distance)
                predicted_prototype_indices_euclidean = torch.argmin(distances_euclidean, dim=1).cpu().numpy()
                predicted_labels_euclidean = [unique_labels_sorted[idx] for idx in predicted_prototype_indices_euclidean]

                acc_euclidean = accuracy_score(test_labels_tensor.numpy(), predicted_labels_euclidean)
                f1_euclidean = f1_score(test_labels_tensor.numpy(), predicted_labels_euclidean, average="weighted")

                print(f"  Zero-Shot (Euclidean) Accuracy: {acc_euclidean:.4f}")
                print(f"  Zero-Shot (Euclidean) F1-Score: {f1_euclidean:.4f}")
       
                _logger.info(f"  Zero-Shot (Euclidean) Accuracy: {acc_euclidean:.4f}")
                _logger.info(f"  Zero-Shot (Euclidean) F1-Score: {f1_euclidean:.4f}")

                
                # --- Cosine Similarity ---
                # Normalize features and prototypes for cosine similarity
                test_features_norm = test_features_tensor / (torch.norm(test_features_tensor, dim=1, keepdim=True) + 1e-8)
                prototype_tensor_norm = prototype_tensor / (torch.norm(prototype_tensor, dim=1, keepdim=True) + 1e-8)

                similarities_cosine = torch.matmul(test_features_norm, prototype_tensor_norm.T)
                predicted_prototype_indices_cosine = torch.argmax(similarities_cosine, dim=1).cpu().numpy()
                predicted_labels_cosine = [unique_labels_sorted[idx] for idx in predicted_prototype_indices_cosine]

                acc_cosine = accuracy_score(test_labels_tensor.numpy(), predicted_labels_cosine)
                f1_cosine = f1_score(test_labels_tensor.numpy(), predicted_labels_cosine, average="weighted")

                print(f"  Zero-Shot (Cosine) Accuracy: {acc_cosine:.4f}")
                print(f"  Zero-Shot (Cosine) F1-Score: {f1_cosine:.4f}")

                _logger.info(f"  Zero-Shot (Cosine) Accuracy: {acc_cosine:.4f}")
                _logger.info(f"  Zero-Shot (Cosine) F1-Score: {f1_cosine:.4f}")


                if zeroshot_similarity == 'euclidean':
                    accuracies.append(acc_euclidean)
                    f1_scores.append(f1_euclidean)
                elif zeroshot_similarity == 'cosine':
                    accuracies.append(acc_cosine) # Or acc_euclidean, or average them, or store separately
                    f1_scores.append(f1_cosine)   # Or f1_euclidean
                else:
                    raise ValueError

                acc_mean = np.mean(accuracies)
                acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
                f1_mean = np.mean(f1_scores)
                f1_std = np.std(f1_scores) if len(f1_scores) > 1 else 0.0
                model_results[k] = {"accuracy_mean": acc_mean, "accuracy_std": acc_std,
                                "f1_mean": f1_mean, "f1_std": f1_std, "meta": "zeroshot"}

            else:

                results_summary = []
                for config in hyperparam_grid:

                        
                    all_acc = []
                    all_f1 = []

                    for exp in range(num_exp):
                        seed = get_deterministic_seed_for_experiment(i, exp)
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)


                        features_tensor = torch.tensor(train_features, dtype=torch.float32)
                        labels_tensor = torch.tensor(train_labels, dtype=torch.long)
                        train_dataset = TensorDataset(features_tensor, labels_tensor)

                        support_features, support_labels, query_features, query_labels = sample_episode(
                            train_dataset, 
                            n_way=5, 
                            k_shot=k, 
                            query_per_class=20
                        )

                        _, trainer_results = train_pytorch_classifier(
                            features=support_features,
                            labels=support_labels,
                            test_features=query_features,
                            test_labels=query_labels,
                            num_classes=5,
                            device=device,
                            classifier_batch_size=config["classifier_batch_size"],
                            lr=config['lr'],
                            epochs=config['num_epochs'],
                            model_name=model_name,
                            k_value=k,
                            random_seed=seed
                        )


                        if trainer_results:
                            if len(trainer_results) > 1:
                                raise ValueError
                            final_accuracy = trainer_results[0].get('test_accuracy')
                            final_f1 = trainer_results[0].get('test_f1_score')
                            print(f"Final Test Accuracy: {final_accuracy:.4f}")
                            print(f"Final Test F1-Score (weighted): {final_f1:.4f}")

                        all_acc.append(final_accuracy)
                        all_f1.append(final_f1)

                    mean_acc = np.mean(all_acc)
                    std_acc = np.std(all_acc)
                    mean_f1 = np.mean(all_f1)
                    std_f1 = np.std(all_f1)
                    results_summary.append((config, mean_acc, std_acc, mean_f1, std_f1))
                    print(f"{config} → mean={mean_acc:.2f}%, std={std_acc:.2f}%")
                    _logger.info(f"{config} → mean={mean_acc:.2f}%, std={std_acc:.2f}%")

 
                # Pick best config by mean accuracy
                best_config = max(results_summary, key=lambda x: x[1])
                print(f'\n*** MODEL {model_name} ***')
                print(f"\nBest config for k={k}: {best_config}")
                # _logger.info(f"\nBest config for k={k}: {best_config}")

                acc_mean = best_config[1]
                acc_std = best_config[2]
                f1_mean = best_config[3]
                f1_std = best_config[4]
                model_results[k] = {"accuracy_mean": acc_mean, "accuracy_std": acc_std,
                                    "f1_mean": f1_mean, "f1_std": f1_std, "meta": results_summary, "best_config": best_config}
                print(f"  k={k}: Accuracy = {acc_mean:.4f} ± {acc_std:.4f}, F1 = {f1_mean:.4f} ± {f1_std:.4f}")

        

        print(f'\n*** MODEL {model_name} ***')
        _logger.info(f'Results for MODEL {model_name}: {model_results}')


        results[model_name] = model_results
    return results





if __name__ == "__main__":

    MODEL_ROOT = "/local/ms-data/SSL4EO/model"
    MODEL_CONFIGS = {
        "2025_8_3_RandomInit-deltaAI-epoch1": {
            "type": "ciip",
            "weights": f"{MODEL_ROOT}/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/epoch_1.pt",
        },
        "2025_8_3_RandomInit-deltaAI-epoch2": {
            "type": "ciip",
            "weights": f"{MODEL_ROOT}/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/epoch_2.pt",
        },
       
        "2025_8_3_RandomInit-deltaAI-epoch10": {
            "type": "ciip",
            "weights": f"{MODEL_ROOT}/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/epoch_10.pt",
        },

        "2025_8_3_RandomInit-deltaAI-epoch53": {
            "type": "ciip",
            "weights": f"{MODEL_ROOT}/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/epoch_53.pt",
        },
        "2025_8_6_MoCoInit-deltaAI-epoch62": {
            "type": "ciip",
            "weights": f"{MODEL_ROOT}/2025_08_06-MoCoInit/no-copy/epoch_62.pt",
        },
        "SSL4EO-ResNet50_MoCo": {
            "type": "resnet50",
            "weights": ResNet50_Weights.SENTINEL2_ALL_MOCO,
        },
        "SSL4EO-ResNet50_DINO": {   
            "type": "resnet50",
            "weights": ResNet50_Weights.SENTINEL2_ALL_DINO,
        },
        "ResNet50_Random": {
            "type": "resnet50",
            'weights': None,
        }
    }

    CONFIG = {
        "data_path": "/local/ms-data/EuroSAT/",  # Path to EuroSAT dataset
        "k_values": [1, 4, 32, 100], #[0, 1, 5, 10, 16, 32, 100, 1000], #[0, 1, 4, 8, 10, 16, 32, 50, 100, 200],
        # "batch_size": 512,
        "num_workers": 8,
        "num_experiments": 5,
        "bands": BANDS,  # assuming BANDS is defined elsewhere
        "models": [ '2025_8_3_RandomInit-deltaAI-epoch53', "SSL4EO-ResNet50_MoCo", "ResNet50_Random"], 
        # "2025_8_6_MoCoInit-deltaAI-epoch62", "SSL4EO-ResNet50_MoCo", "SSL4EO-ResNet50_DINO"]
    }

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
            "normalized": NORMALIZE
        })

    CONFIG["models"] = model_info

    ## Setup Logging ##
    # lightning logger
    logging.getLogger("lightning").setLevel(logging.WARNING) 


    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    experiment_name = f"{timestamp}"


    log_dir = f"results/logs/fewshot/normalization-{NORMALIZE}/"
    log_filename = experiment_name + "-training_log.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)

    custom_logger = logging.getLogger(__name__) # Use __name__ or a custom name like "my_trainer_logger"
    custom_logger.setLevel(logging.INFO)    

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not custom_logger.handlers: # Only add if no handlers are present
        custom_logger.addHandler(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    _logger = logging.getLogger(__name__)

    results = few_shot_comparison_pipeline_optimized(**CONFIG)
    print(results)

    _logger.info(results)

    
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    experiment_name = f"{timestamp}"
    output_dir = os.path.join(f"fewshot/linearprobe-normalized-{NORMALIZE}", experiment_name)
    os.makedirs(output_dir, exist_ok=True)


    # Save metrics
    output_results_to_csv(results, CONFIG["k_values"], metric="accuracy", output_dir=output_dir)
    output_results_to_csv(results, CONFIG["k_values"], metric="f1", output_dir=output_dir)

    # Save plots
    plot_results(results, CONFIG["k_values"], metric="accuracy", output_file=os.path.join(output_dir, "linearprobe_accuracy.png"))
    plot_results(results, CONFIG["k_values"], metric="f1", output_file=os.path.join(output_dir, "linearprobe_f1.png"))


    SAVE_CONFIG = CONFIG
    SAVE_CONFIG['models'] = []
    try:
        for model_info in CONFIG['models']:
            if "weights" in model_info:
                del model_dict["weights"]
            SAVE_CONFIG['models'].append(model_info)
    except:
        pass
        
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)