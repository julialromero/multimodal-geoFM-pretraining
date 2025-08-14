import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from geopy.distance import geodesic
from pyproj import Proj, Transformer
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import hdbscan
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
import umap
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import ast
from omegaconf import OmegaConf, DictConfig
import hydra
import sys
# parent_dir = '/home/juro4948/ciip/ciip/open_clip_train/'
parent_dir = '/home/jema2085/ciip/ciip/open_clip_train/'
sys.path.insert(0, parent_dir)
from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master
from open_clip_train.distributed import is_master, init_distributed_device
from utils import create_model
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn


def compare_keys(model, state_dict):
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    return missing, unexpected


def load_model(args, chkpt_path, w_path, device='cuda'):
    # device = init_distributed_device(args.datamodule)
    model = create_model(
        args,
        device=device
    )
    
    # load the model state dict weights
    path = chkpt_path
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint


    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        print(f"Failed to load state_dict with strict=True, trying to remove 'module.' prefix.")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            print(f"Failed to load state_dict with strict=True after removing 'module.' prefix, trying to remove 'encoder_s1.W'.")
            # If the model has an encoder_s1.W, remove it from the state_dict
            if 'encoder_s1.W' in state_dict:
                print(f"Removing encoder_s1.W from state_dict")
                del state_dict['encoder_s1.W']

            missing, unexpected = compare_keys(model, state_dict)
            if missing:
                print("Missing keys in checkpoint:", missing)
            if unexpected:
                print("Unexpected keys in checkpoint:", unexpected)
            if not missing and not unexpected:
                print("All keys match between model and checkpoint state_dict.")

            model.load_state_dict(state_dict, strict=True)


    if w_path is not None and os.path.isfile(w_path):
        W = torch.load(w_path)
        model.encoder_s1.register_buffer('W', W)
        model.encoder_s1.apply_orthogonal_matrix = True
        print('Loaded W from {}, shape: {}'.format(w_path, W.shape))
    model = model.to(device, dtype=get_input_dtype(args.model.precision), non_blocking=True)

    return model

# From "It's Not a Modality Gap": Linear Separability (from Welle (2023)) is the percentage of image and text
# [or in our case, image and image] embeddings that can be distinguished by a linear classifier operating in
# CLIP space. We used 80% of the dataset to train a linear model to classify CLIP embeddings as originating from
# either "image" of "text" input. We then tested the performance of the classifier on the remaining 20% of the
# dataset and reported the accuracy. If a set of embeddings are 100% linearly separable, this means that the
# space occupied by each modality is completely disjoint. Conversely, 50% linear separability means that the
# image and text embeddings are overlapping in CLIP space, meaning that they occupy the same region of the
# latent space; i.e. there is no gap between the embeddings. To summarize, if we can effectively close the gap
# we will find that the distance between centroids is small and linear separability is close to 50%.
def calc_linear_separability(s1_embeddings, s2_embeddings):
    combined_embeddings = np.vstack([s1_embeddings, s2_embeddings])
    labels = np.array([0] * len(s1_embeddings) + [1] * len(s2_embeddings))

    X_train, X_test, y_train, y_test = train_test_split(combined_embeddings, labels, test_size=0.90, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    separability = accuracy_score(y_test, y_pred)

    return separability

# From Two Effects, One Trigger
def calc_relative_modality_gap(s1_embeddings, s2_embeddings):
    # # normalize s1
    # s1_embeddings = s1_embeddings / (torch.norm(s1_embeddings, dim=1, keepdim=True) + 1e-8)
    # s2_embeddings = s2_embeddings / (torch.norm(s2_embeddings, dim=1, keepdim=True) + 1e-8)
    s1_embeddings = np.array(s1_embeddings)
    s2_embeddings = np.array(s2_embeddings)

    # Cross-modal dissimilarity (d(x_i, y_i))
    cross_dissim = 1 - np.sum(s1_embeddings * s2_embeddings, axis=1)
    avg_cross_dissim = np.mean(cross_dissim)

    # Intra-modal dissimilarities
    avg_s1_dissim = np.mean(1 - cosine_similarity(s1_embeddings))
    avg_s2_dissim = np.mean(1 - cosine_similarity(s2_embeddings))

    rmg = avg_cross_dissim / (avg_s1_dissim + avg_s2_dissim + avg_cross_dissim)

    return rmg

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        model,
        encoder='s1',  # 's1', 's2', or 'both'
        device='cuda',
        transform=None,
        cache_dir=None,
        input_dtype=None,
        autocast=None
    ):
        """
        Args:
            base_dataset (torch.utils.data.Dataset): Dataset returning raw samples.
            model (torch.nn.Module): Loaded model with encoder_s1 / encoder_s2.
            encoder (str): 's1', 's2', or 'both' — which encoder(s) to use.
            device (str): Device for inference.
            transform (callable, optional): Optional transform to apply to input before model.
            cache_dir (str, optional): Directory to store cached embeddings.
        """
        self.base_dataset = base_dataset
        self.model = model #.eval().to(device)
        self.encoder = encoder
        self.device = device
        self.transform = transform
        self.cache_dir = cache_dir
        self.input_dtype = input_dtype
        self.autocast = autocast
#
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.base_dataset)

    def _get_cache_path(self, idx):
        return os.path.join(self.cache_dir, f"{idx}.pt")

    @torch.no_grad()
    def __getitem__(self, idx):
        # Check if cached
        if self.cache_dir:
            cache_path = self._get_cache_path(idx)
            if os.path.exists(cache_path):
                return torch.load(cache_path)

        # Load raw sample
        if isinstance(self.base_dataset, torch.utils.data.Subset):
            base_dataset = self.base_dataset.dataset
            indices = self.base_dataset.indices
            get_uid = lambda idx: base_dataset.get_sample_uid(indices[idx])
        else:
            get_uid = lambda idx: self.base_dataset.get_sample_uid(idx)
  
        
        sample = self.base_dataset[idx]
        
        # if self.transform:
        #     sample = self.transform(sample)

        s1, s2 = sample
        s1 = torch.tensor(s1).unsqueeze(0).to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        s2 = torch.tensor(s2).unsqueeze(0).to(device=self.device, dtype=self.input_dtype, non_blocking=True)



        # Apply model encoder
        result = {}
        result['uid'], filepath = get_uid(idx)
        # print(sample)
        with self.autocast():
            if self.encoder in ('s1', 'both'):
                out1 = self.model.encode_s1(s1, normalize=True)


                # apply orthogonal matrix W if it exists
                # if hasattr(self.model.encoder_s1, 'W'):
                #     out1 = out1 @ self.model.encoder_s1.W

                # NORMALIZE
                # out1  = out1 / out1.norm(dim=1, keepdim=True)
                result['s1'] = out1.cpu()

            if self.encoder in ('s2', 'both'):
                out2 = self.model.encode_s2(s2, normalize=True)
                # NORMALIZE
                # out2 = out2 / out2.norm(dim=1, keepdim=True)
                result['s2'] = out2.cpu()



        if self.cache_dir:
            torch.save(result, self._get_cache_path(idx))

        return result
    

class Sampler():
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        print(f"Dataset length: {self.length}")

    def sample(self, num_samples):
        samples = []
        sampled_indices = set()
        while len(samples) < num_samples:
            idx = np.random.randint(0, self.length)
            if idx not in sampled_indices:
                sampled_indices.add(idx)
                samples.append(self.dataset[idx])           
        
        return samples


def get_non_corresponding_cosine_similarities(s1, s2):
    """
    Computes cosine similarity between s1[i] and s2[j] for all i != j.

    Args:
        s1 (torch.Tensor): Tensor of shape (N, D).
        s2 (torch.Tensor): Tensor of shape (N, D).

    Returns:
        torch.Tensor: A 1D tensor containing all off-diagonal cosine similarities.
                      If N=1, returns an empty tensor.
    """
    N = s1.shape[0]
    if N == 0:
        return torch.empty(0, dtype=s1.dtype, device=s1.device)
    if N == 1: # No non-corresponding elements for a batch size of 1
        return torch.empty(0, dtype=s1.dtype, device=s1.device)


    # 1. L2 Normalize both tensors
    s1_norm = F.normalize(s1, p=2, dim=1)
    s2_norm = F.normalize(s2, p=2, dim=1)

    # 2. Compute the full pairwise cosine similarity matrix
    # Resulting shape: (N, N)
    similarity_matrix = torch.matmul(s1_norm, s2_norm.T)

    # 3. Create a mask to select off-diagonal elements
    # The diagonal elements (i == j) correspond to positive pairs.
    # The off-diagonal elements (i != j) correspond to negative (non-corresponding) pairs.
    mask = torch.ones(N, N, dtype=torch.bool, device=s1.device)
    mask = mask.fill_diagonal_(False) # Set diagonal elements to False

    # 4. Use the mask to extract the non-corresponding similarities
    non_corresponding_sims = similarity_matrix[mask]

    return non_corresponding_sims.cpu().numpy()
    
def process_sampled_to_df(sampled, sat='s1'):
    dfs = []
    for d in sampled:
        # print(d)
        arr = d[sat]
        # Convert the 2D numpy array to a DataFrame
        df = pd.DataFrame(arr)
        # Optionally, add an identifier column if needed, e.g., from another key in the dictionary
        df['uid'] = d['uid']
        dfs.append(df)

    # list of dfs to df
    df = pd.concat(dfs)
    return df

from torch.utils.data import Dataset, DataLoader
class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

CONF = "prod_default"

from data import get_data
# @hydra.main(config_path="/home/juro4948/ciip/ciip/open_clip_train/configs", config_name=CONF)
@hydra.main(config_path="/home/jema2085/ciip/ciip/open_clip_train/configs", config_name=CONF)

def main(args: DictConfig):
    print('----')
    # print(f"Loaded args: {args}")

    # path to the directory containing the experiment model checkpoints
    # chkpt_path = '/local/ms-data/SSL4EO/model/2025_07_03-RandomInit-bs4096/checkpoints'
    chkpt_path = '/local/ms-data/SSL4EO/model/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/'
    # '/local/ms-data/SSL4EO/model/2025-08-03_12-52-38-test-compute/2025_08_03-13_00_14-model_resnet50-lr_5e-05-b_256-j_4-p_amp/checkpoints/'
    # /local/ms-data/SSL4EO/model/2025_08_06-MoCoInit/no-copy/'
   
    experiment_name = chkpt_path.split('/')[-2]
    print(f'Experiment name: {experiment_name}')

    # orthogonal experiment -> '/home/juro4948/ciip/logs/2025_07_02-12_18_14-model_resnet50-lr_0.0001-b_128-j_6-p_fp16/checkpoints'
 
    # path to the orthogonal matrix (static)
    w_path = os.path.join(chkpt_path, 'W.pt') if os.path.exists(os.path.join(chkpt_path, 'W.pt')) else None
    print(f'Using orthogonal matrix from: {w_path}')

    num_pairs = 2000

    device = init_distributed_device(args.datamodule)
    # device = torch.device(args.datamodule.device)
    input_dtype = get_input_dtype(args.model.precision)
    autocast = get_autocast(args.model.precision)

   
    data = get_data(args)

    # subset the data to ensure the same samples across all models
    train_dataset = data['train'].dataloader.dataset
    subset_indices = np.random.RandomState(42).choice(len(train_dataset), num_pairs, replace=False)
    subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    models = os.listdir(chkpt_path)
    models = [m for m in models if m.endswith('.pt') and 'epoch' in m]
    # rename epoch_init to epoch_0
    models = [m.replace('epoch_init', 'epoch_0') for m in models]
    models = sorted(models, key=lambda x: int(x.split('_')[1].split('.')[0]))

    
    # centroids = []
    paired_l2_norms = []
    centroid_l2_norms = []
    model_checkpoints = []
    cosine_sims = []
    paired_cosine_sims = []
    negative_cosine_sims = []
    linear_separabilities = []
    relative_modality_gaps = []

    for model_epoch_fn in models:
        print(f'-----Processing model: {model_epoch_fn}------')
        model_checkpoints.append(int(model_epoch_fn.split('_')[1].split('.')[0]))

        if 'epoch_0' in model_epoch_fn:
            model_epoch_fn = 'epoch_init.pt'
            
        path = os.path.join(chkpt_path, model_epoch_fn)


        model = load_model(args, path, w_path=w_path, device=device)
        model.eval()
        # returns the normalized embeddings for s1 and s2
        with torch.no_grad():
            embedding_dataset = EmbeddingDataset(
                base_dataset=subset_dataset,
                model=model,
                encoder='both',  # or 'both'
                device=device,
                cache_dir=None, #EMBEDDING_OUTPUT_PATH
                input_dtype=input_dtype,
                autocast=autocast
            )

            sampler = Sampler(embedding_dataset)
            sampled = sampler.sample(num_samples=num_pairs)


            # # datalaoder 
            # loader = DataLoader(
            #     embedding_dataset,
            #     batch_size=512,  # Adjust based on memory constraints
            #     shuffle=False,
            #     num_workers=4,  # Adjust based on your system
            #     pin_memory=True
            # )
            # NUM_WARMUP_BATCHES = 100
            # for i, (s1, s2) in enumerate(loader):
            #     s1 = s1.to(device)
            #     s2 = s2.to(device)
            #     _ = model.encode_s1(s1)
            #     _ = model.encode_s2(s2)
            #     if i >= NUM_WARMUP_BATCHES:
            #         break

            uids = [sample['uid'] for sample in sampled]
            s1 = process_sampled_to_df(sampled, sat='s2').drop(columns=['uid'], errors='ignore').reset_index(drop=True)
            s2 = process_sampled_to_df(sampled, sat='s1').drop(columns=['uid'], errors='ignore').reset_index(drop=True)


      
        # --- Paired L2 norm ---
        l2_norm_s1_s2 = np.linalg.norm(s1.values - s2.values, axis=1)
        paired_l2_norms.append(l2_norm_s1_s2)

        # --- Convert to torch ---
        s1 = torch.tensor(s1.to_numpy(), dtype=torch.float32)
        s2 = torch.tensor(s2.to_numpy(), dtype=torch.float32)

        # --- Normalize each sample (row-wise) ---
        s1 = F.normalize(s1, p=2, dim=1)
        s2 = F.normalize(s2, p=2, dim=1)

        # --- Paired cosine similarity ---
        cos_sim_s1_s2 = F.cosine_similarity(s1, s2, dim=1).cpu().numpy()
        paired_cosine_sims.append(cos_sim_s1_s2)

        neg_sims = get_non_corresponding_cosine_similarities(s1, s2)
        negative_cosine_sims.append(neg_sims)

        # --- Centroid calculation ---
        s1_centroid = s1.mean(dim=0, keepdim=True)
        s2_centroid = s2.mean(dim=0, keepdim=True)

        # --- Normalize centroids ---
        s1_centroid = F.normalize(s1_centroid, p=2, dim=1)
        s2_centroid = F.normalize(s2_centroid, p=2, dim=1)


        # --- L2 norm between centroids ---
        l2_norm_centroids = torch.norm(s1_centroid - s2_centroid, p=2).item()
        centroid_l2_norms.append(l2_norm_centroids)

        # --- Cosine similarity between centroids ---
        cos_sim_centroids = F.cosine_similarity(s1_centroid, s2_centroid).item()
        cosine_sims.append(cos_sim_centroids)

        linear_separability = calc_linear_separability(s1, s2)
        linear_separabilities.append(linear_separability)

        relative_modality_gap = calc_relative_modality_gap(s1, s2)
        relative_modality_gaps.append(relative_modality_gap)

        if int(os.environ.get("RANK", "0")) == 0:
            logging.info(f'L2 Norm between centroids for {model_epoch_fn}: {l2_norm_centroids}')
            logging.info(f'Cosine Similarity between centroids of S1 and S2 for {model_epoch_fn}: {cos_sim_centroids}')
            logging.info(f'Linear separability of S1 and S2 for {model_epoch_fn}: {linear_separability}')
            logging.info(f'Relative Modality Gap (RMG) of S1 and S2 for {model_epoch_fn}: {relative_modality_gap}')
    
    # print(f'Length of model checkpoints: {len(model_checkpoints)}')
    # print(f'Length of paired L2 norms: {len(paired_l2_norms)}')
    # print(f'Length of centroid L2 norms: {len(centroid_l2_norms)}')
    # print(f'Length of cosine sims: {len(cosine_sims)}')
    print(f'Model checkpoints: {model_checkpoints}')

    # now plot each
    plt.plot(model_checkpoints, paired_l2_norms)
    plt.figure()
    # plt.hist(l2_norm_s1_s2, bins=20, alpha=0.7, label=model_checkpoints, color='blue')
    # plot mean and std of l2 norms
    mean_l2_norms = [np.mean(norms) for norms in paired_l2_norms]
    std_l2_norms = [np.std(norms) for norms in paired_l2_norms]
    plt.errorbar(model_checkpoints, mean_l2_norms, yerr=std_l2_norms, fmt='o', capsize=5, label='Mean ± Std Dev', color='blue', linestyle='-')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Pairwise L2 Norms')
    plt.title(f'Pairwise S1-S2 L2 Norms - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('pairwise-l2norm.png', bbox_inches='tight')
    plt.show()

    # plot paired cosine sims
    # plt.plot(model_checkpoints, paired_cosine_sims)
    plt.figure()
    mean_cosine_sims = [np.mean(sims) for sims in paired_cosine_sims]
    std_cosine_sims = [np.std(sims) for sims in paired_cosine_sims]
    
    # plt.plot(model_checkpoints, negative_cosine_sims)
    mean_negative_cosine_sims = [np.mean(sims) for sims in negative_cosine_sims]
    std_negative_cosine_sims = [np.std(sims) for sims in negative_cosine_sims]
    plt.errorbar(model_checkpoints, mean_cosine_sims, yerr=std_cosine_sims, fmt='o', capsize=5, label='Positive Pairs', color='blue', linestyle='-')
    plt.errorbar(model_checkpoints, mean_negative_cosine_sims, yerr=std_negative_cosine_sims, fmt='o', capsize=5, label='Negative Pairs', color='orange', linestyle='--')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Pairwise Cosine Similarities')
    plt.legend()
    plt.title(f'Pairwise S1-S2 Cosine Similarities - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('pairwise-cosine-sim.png', bbox_inches='tight')
    plt.show()


    plt.figure()
    plt.plot(model_checkpoints, centroid_l2_norms, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Centroid L2 Norm')
    plt.title(f'Centroid S1-S2 L2 Norm - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('centroid-l2norm.png', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(model_checkpoints, cosine_sims, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Centroid Cosine Sim')
    plt.title(f'Centroid S1-S2 Cosine Sim - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('centroid-cosine-sim.png', bbox_inches='tight')
    plt.show()

    # Linear Separability Plot
    plt.figure()
    plt.plot(model_checkpoints, linear_separabilities, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Linear Separability')
    plt.title(f'Linear Separability S1-S2 - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('linear-separability.png', bbox_inches='tight')
    plt.show()

    # RMG Plot
    plt.figure()
    plt.plot(model_checkpoints, relative_modality_gaps, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Relative Modality Gap (RMG)')
    plt.title(f'Relative Modality Gap (RMG) S1-S2 - {experiment_name} (n={num_pairs})')
    plt.grid()
    plt.savefig('rmg.png', bbox_inches='tight')
    plt.show()

 



if __name__ == "__main__":
    main()