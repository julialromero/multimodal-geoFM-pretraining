# coding: utf-8
"""Comparison script using SVM classifier on frozen features.
This script mirrors linearprobe_comparison.py but trains a
scikit-learn SVM classifier instead of a PyTorch linear head.
"""

import copy
import hashlib
import json
import logging
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from torchgeo.datasets import EuroSAT
from torchgeo.models import ResNet50_Weights

from utils import (
    CustomTransform,
    drop_last_linear_layer,
    get_deterministic_seed_for_experiment,
    load_model,
    plot_results,
    output_results_to_csv,
)

# Dataset normalization statistics for EuroSAT bands
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


def extract_features(model, dataloader, device, use_s2_only=False):
    """Extract backbone features for an entire dataloader."""
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            labels = batch["label"]
            if use_s2_only:
                features = model.encode_s2(s2=images, normalize=True)
            else:
                features = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def train_svm_classifier(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """Fit a linear SVM on frozen features and evaluate on validation and test sets."""
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    clf.fit(train_features, train_labels)

    val_pred = clf.predict(val_features)
    test_pred = clf.predict(test_features)

    val_acc = accuracy_score(val_labels, val_pred)
    val_f1 = f1_score(val_labels, val_pred, average="weighted")
    test_acc = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred, average="weighted")

    results = [{
        'val_accuracy': val_acc,
        'val_f1_score': val_f1,
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
    }]
    return clf, results


def few_shot_comparison_pipeline_svm(data_path, percents, models, bands, batch_size=16, num_workers=18,
                                     num_experiments=10, output_dir=None, normalize=False):
    """Run few-shot comparison using an SVM classifier on frozen features."""
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformed_dataset_train = EuroSAT(data_path, split="train", bands=BANDS, transforms=train_transform, download=True)
    transformed_dataset_val = EuroSAT(data_path, split="val", bands=BANDS, transforms=val_transform, download=True)
    transformed_dataset_test = EuroSAT(data_path, split="test", bands=BANDS, transforms=val_transform, download=True)

    for model_config in models:
        model = load_model(model_config["type"], model_config["weights"])
        for param in model.parameters():
            param.requires_grad = False
        model_name = model_config["name"]
        model_results = {}
        current_model = copy.deepcopy(model).to(device).eval()
        if model_config.get('drop_last_layer'):
            current_model = drop_last_linear_layer(current_model)

        for i, percent in enumerate(percents):
            accuracies, f1_scores = [], []
            for exp in range(num_experiments):
                seed = get_deterministic_seed_for_experiment(i, exp)
                num_samples = int(percent * len(transformed_dataset_train))
                rng = np.random.default_rng(seed)
                selected_indices = rng.choice(len(transformed_dataset_train), num_samples, replace=False)

                train_data = Subset(transformed_dataset_train, selected_indices)
                dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                dataloader_val = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                dataloader_test = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

                use_s2_only = (model.__class__.__name__ == "CIIP")
                train_features, train_labels = extract_features(current_model, dataloader_train, device, use_s2_only=use_s2_only)
                val_features, val_labels = extract_features(current_model, dataloader_val, device, use_s2_only=use_s2_only)
                test_features, test_labels = extract_features(current_model, dataloader_test, device, use_s2_only=use_s2_only)

                if normalize:
                    train_features = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
                    val_features = val_features / (np.linalg.norm(val_features, axis=1, keepdims=True) + 1e-8)
                    test_features = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

                _, trainer_results = train_svm_classifier(
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                    test_features,
                    test_labels,
                )
                final_accuracy = trainer_results[0]['test_accuracy']
                final_f1 = trainer_results[0]['test_f1_score']
                accuracies.append(final_accuracy)
                f1_scores.append(final_f1)

            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies) if len(accuracies) > 1 else 0.0
            f1_mean = np.mean(f1_scores)
            f1_std = np.std(f1_scores) if len(f1_scores) > 1 else 0.0
            model_results[percent] = {
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_raw": accuracies,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_raw": f1_scores,
            }
        results[model_name] = model_results
    return results


def generate_ciip_model_configs(model_root: str, base_name: str, start_epoch: int, end_epoch: int, step: int = 5,
                                drop_last_layer: bool = True):
    """Generate configuration dictionaries for CIIP checkpoints."""
    configs = {}
    for epoch in range(start_epoch, end_epoch + 1, step):
        model_name = f"{base_name}-epoch{epoch}"
        configs[model_name] = {
            "type": "ciip",
            "weights": os.path.join(model_root, f"epoch_{epoch}.pt"),
            "drop_last_layer": drop_last_layer,
        }
    return configs


if __name__ == "__main__":
    MODEL_ROOT = "/local/ms-data/SSL4EO/model"
    ciip_root = f"/home/juro4948/ciip/logs/2025_09_05-13_28_50-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints"
    MODEL_CONFIGS = generate_ciip_model_configs(
        model_root=ciip_root,
        base_name="2025_09_05_MoCoInit-hal",
        start_epoch=5,
        end_epoch=200,
        step=5,
    )

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
        "data_path": "/local/ms-data/EuroSAT/",
        "percents": [1],
        "batch_size": 512,
        "num_workers": 8,
        "num_experiments": 1,
        "bands": BANDS,
        "models": list(MODEL_CONFIGS.keys()),
    }

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
            'drop_last_layer': config['drop_last_layer'],
        })
    CONFIG["models"] = model_info
    CONFIG['notes'] = 'SVM linear probe comparison.'

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    experiment_name = f"{timestamp}"
    output_dir = os.path.join("results", "svm-linearprobe", f"normalized-{NORMALIZE}", experiment_name)
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

    results = few_shot_comparison_pipeline_svm(output_dir=output_dir, normalize=NORMALIZE, **CONFIG)
    print(results)
    _logger.info(results)

    output_results_to_csv(results, CONFIG["percents"], metric="accuracy", output_dir=output_dir)
    output_results_to_csv(results, CONFIG["percents"], metric="f1", output_dir=output_dir)

    plot_results(results, CONFIG["percents"], metric="accuracy", output_file=os.path.join(output_dir, "svm_accuracy.png"))
    plot_results(results, CONFIG["percents"], metric="f1", output_file=os.path.join(output_dir, "svm_f1.png"))

    _logger.info(f'Saved results to {output_dir}')
    print(f'Saved results to {output_dir}')

    SAVE_CONFIG = CONFIG
    SAVE_CONFIG['models'] = []
    try:
        for model_info in CONFIG['models']:
            if "weights" in model_info:
                del model_info["weights"]
            SAVE_CONFIG['models'].append(model_info)
    except Exception:
        pass

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(SAVE_CONFIG, f, indent=4)
