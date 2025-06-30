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
from open_clip_train.precision import get_autocast
from open_clip import get_input_dtype


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = sorted(os.listdir(root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(os.path.join(self.root, sample)) 
        # convert tensor to np
        data['s1_features'] = data['s1_features'].numpy()
        data['s2_features'] = data['s2_features'].numpy()

        return data

import sys
parent_dir = '/home/juro4948/ciip/ciip/open_clip_train/'
sys.path.insert(0, parent_dir)
from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master
from open_clip_train.distributed import is_master, init_distributed_device
from utils import create_model


def load_model(args, chkpt_path,w_path, device='cuda'):
    device = init_distributed_device(args.datamodule)
    model = create_model(
        args,
        device=device
    )
    # print(f'Model created: {model.__class__.__name__}, device: {device}')

    if chkpt_path is None:
        print(f"No checkpoint path provided, initializing model per prod_default init method:")
        if not args.model.pretrain.load:
            print(f"Random Initialization")
        else:
            print(f"Load weights: s1: {args.model.pretrain.s1_weights}, s2c: {args.model.pretrain.s2_weights}")
        pass
    else:
        checkpoint = torch.load(chkpt_path)
        if 'epoch_init' in chkpt_path:
            print(f"Loading model from {chkpt_path} for initialization")
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(state_dict)
        else:
            checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            # print(checkpoint['state_dict']['encoder_s1.W'])
            if 'encoder_s1.W' in checkpoint['state_dict']:
                print(f"removing encoder_s1.W from state_dict")
                del checkpoint['state_dict']['encoder_s1.W']
            model.load_state_dict(checkpoint['state_dict'])
            model.encoder_s1.apply_orthogonal_matrix = True
            print(f"Loaded model from {chkpt_path}")

        # set W parameter
  
    if 'W' in model.state_dict():
        if  w_path is not None: 
            raise ValueError(f"W is already in the model state dict, but also provided W_PATH: {w_path}. Please remove one of them.")
        W = model.state_dict()['W']
        print(f"Loaded W from model state dict, shape: {W.shape}")
    elif w_path is not None and os.path.isfile(w_path):
        W = torch.load(w_path)
        model.encoder_s1.register_buffer('W', W)
        print(f"Loaded W from {w_path}, shape: {W.shape}")
    
    model = model.to(args.datamodule.device, dtype=get_input_dtype(args.model.precision), non_blocking=True)
    return model


class EmbeddingDataset(Dataset):
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
        self.model = model.eval().to(device)
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
                out1 = self.model.encoder_s1(s1)
                # NORMALIZE
                out1  = out1 / out1.norm(dim=1, keepdim=True)
                result['s1'] = out1.cpu()

            if self.encoder in ('s2', 'both'):
                out2 = self.model.encoder_s2(s2)
                # NORMALIZE
                out2 = out2 / out2.norm(dim=1, keepdim=True)
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

def cosine_similarity_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity matrix between two sets of vectors.

    Args:
        X (torch.Tensor): Tensor of shape (N, D)
        Y (torch.Tensor): Tensor of shape (M, D)

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (N, M)
                      where entry (i, j) is the cosine similarity between X[i] and Y[j]
    """
    # Normalize rows to unit vectors
    X_norm = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Y_norm = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Compute cosine similarity as dot product of normalized vectors
    sim_matrix = X_norm @ Y_norm.T  # shape: (N, M)

    return sim_matrix


def unique_cosine_similarities(X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
    """
    Computes cosine similarity between all unique pairs of embeddings.
    
    If Y is None, computes the upper triangle (without diagonal) of cosine similarity matrix of X vs X.
    If Y is provided and X.shape[0] == Y.shape[0], returns pairwise cosine similarity: X[i] vs Y[i].
    
    Args:
        X (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        Y (torch.Tensor, optional): Tensor of shape (N, D) or (M, D)
        
    Returns:
        torch.Tensor: 1D tensor of cosine similarities for each unique pair.
    """
    if Y is None:
        sim_matrix = cosine_similarity_matrix(X, X)
        # Extract upper triangle without the diagonal
        N = sim_matrix.size(0)
        iu = torch.triu_indices(N, N, offset=1)
        unique_sims = sim_matrix[iu[0], iu[1]]
        return unique_sims
    else:
        assert X.shape == Y.shape, "If Y is provided, X and Y must have the same shape to compute pairwise similarities"
        sim_matrix = cosine_similarity_matrix(X, Y)
        N = sim_matrix.shape[0]
        iu = np.triu_indices(N, k=1)
        upper_triangle_sims = sim_matrix[iu]
        return upper_triangle_sims
    
def centroid_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity between the centroids (mean vectors) of two embedding sets.

    Args:
        X (torch.Tensor): Tensor of shape (N, D)
        Y (torch.Tensor): Tensor of shape (M, D)

    Returns:
        torch.Tensor: Scalar tensor with cosine similarity between centroids
    """
    centroid_X = X.mean(dim=0)
    centroid_Y = Y.mean(dim=0)
    
    similarity = F.cosine_similarity(centroid_X.unsqueeze(0), centroid_Y.unsqueeze(0), dim=1)
    return similarity.item()

# Centroid distances (L2 and cosine similarity)
def get_centroid_stats(s1, s2, model_name):
    centroid_l2 = np.linalg.norm(s1.mean(axis=0) - s2.mean(axis=0))
    cosine = centroid_cosine_similarity(
        torch.tensor(s1.mean(axis=0)).unsqueeze(0),
        torch.tensor(s2.mean(axis=0)).unsqueeze(0)
    )
    print(f"L2 Norm between centroids for {model_name}: {centroid_l2}")
    print(f"Cosine similarity between centroids for {model_name}: {cosine}")
    return centroid_l2, cosine


def plot_histograms(values_list, labels, title, xlabel, ylabel, colors=None):
    """Helper to plot multiple histograms."""
    plt.figure(figsize=(8, 5))
    for values, label, color in zip(values_list, labels, colors or ['blue', 'red', 'orange']):
        plt.hist(values, bins=20, alpha=0.7, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_model_pairwise_distances(model_name, sampler, num_samples):
    """Sample from model, compute intra/inter distances and cosine sims."""
    sampled = sampler.sample(num_samples)
    s1_df = process_sampled_to_df(sampled, sat='s1').drop(columns=['uid'], errors='ignore').reset_index(drop=True)
    s2_df = process_sampled_to_df(sampled, sat='s2').drop(columns=['uid'], errors='ignore').reset_index(drop=True)

    # Euclidean distances
    dist_s1 = euclidean_distances(s1_df, s1_df)
    dist_s2 = euclidean_distances(s2_df, s2_df)
    dist_cross = euclidean_distances(s1_df, s2_df)

    intra_s1 = dist_s1[np.triu_indices_from(dist_s1, k=1)]
    intra_s2 = dist_s2[np.triu_indices_from(dist_s2, k=1)]
    inter_s1s2 = dist_cross.flatten()

    plot_histograms(
        [intra_s1, intra_s2, inter_s1s2],
        labels=['S1-only', 'S2-only', 'S1-S2 inter'],
        title=f'Euclidean Distances: {model_name}',
        xlabel='Distance', ylabel='Frequency'
    )

    # Cosine similarities
    s1_tensor = torch.tensor(s1_df.values, dtype=torch.float32)
    s2_tensor = torch.tensor(s2_df.values, dtype=torch.float32)

    cos_cross = unique_cosine_similarities(s1_tensor, s2_tensor).numpy()
    plot_histograms(
        [cos_cross],
        labels=['All S1-S2'],
        title=f'Cosine Similarities: {model_name}',
        xlabel='Cosine Similarity', ylabel='Frequency',
        colors=['orange']
    )

    return s1_df, s2_df



CONF = "prod_default"
from omegaconf import DictConfig
import hydra
from data import get_data
@hydra.main(config_path="/home/juro4948/ciip/ciip/open_clip_train/configs", config_name=CONF)
# def main(args):
def main(args: DictConfig):
    print('----')

    # path to the director
    # y containing the experiment model checkpoints
    chkpt_path = '/local/ms-data/SSL4EO/model/RandomInit-bs128-fp16-orthogonaloutput/checkpoints/'
    
    # path to the orthogonal matrix (static)
    w_path = '/home/juro4948/ciip/logs/2025_06_08-14_09_50-model_resnet50-lr_0.0001-b_128-j_6-p_fp16/checkpoints/W.pt'
    num_pairs = 10000 

    input_dtype = get_input_dtype(args.model.precision)
    autocast = get_autocast(args.model.precision)
    
    data = get_data(args)
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

    for model_epoch_fn in models:
        model_checkpoints.append(int(model_epoch_fn.split('_')[1].split('.')[0]))

        if 'epoch_0' in model_epoch_fn:
            model_epoch_fn = 'epoch_init.pt'
            
        path = os.path.join(chkpt_path, model_epoch_fn)

        model = load_model(args, path, w_path=w_path, device='cuda')
        model.eval()

        embedding_dataset = EmbeddingDataset(
            base_dataset=subset_dataset,
            model=model,
            encoder='both',  # or 'both'
            device=args.datamodule.device,
            cache_dir=None, #EMBEDDING_OUTPUT_PATH
            input_dtype=input_dtype,
            autocast=autocast
        )

        sampler = Sampler(embedding_dataset)

        sampled = sampler.sample(num_samples=num_pairs)
        # uids = [sample['uid'] for sample in sampled]
        s1 = process_sampled_to_df(sampled, sat='s1').drop(columns=['uid'], errors='ignore').reset_index(drop=True)
        s2 = process_sampled_to_df(sampled, sat='s2').drop(columns=['uid'], errors='ignore').reset_index(drop=True)

        print(f'---- MODEL {model_epoch_fn} -----')
        # Calculate L2 norm between s1 and s2 
        l2_norm_s1_s2 = np.linalg.norm(s1.values - s2.values, axis=1)
        # print(f'L2 norm bw S1-S2: {l2_norm_s1_s2}')
        paired_l2_norms.append(l2_norm_s1_s2)
        

        # calculate the distance between centroids of s1 and s2 for both models
        s1_centroid = s1.mean(axis=0)
        s2_centroid = s2.mean(axis=0)
        l2_norm_centroids = np.linalg.norm(s1_centroid - s2_centroid)
        print(f'L2 Norm between centroids of S1 and S2 for {model_epoch_fn}: {l2_norm_centroids}')
        centroid_l2_norms.append(l2_norm_centroids)
        

        # print cosine similarity between s1 and s2 centroids for both models
        cos_sim_centroids = centroid_cosine_similarity(torch.tensor(s1_centroid).unsqueeze(0), torch.tensor(s2_centroid).unsqueeze(0))
        print(f'Cosine Similarity between centroids of S1 and S2 for {model_epoch_fn}: {cos_sim_centroids}')
        cosine_sims.append(cos_sim_centroids)

    
    print(f'Length of model checkpoints: {len(model_checkpoints)}')
    print(f'Length of paired L2 norms: {len(paired_l2_norms)}')
    print(f'Length of centroid L2 norms: {len(centroid_l2_norms)}')
    print(f'Length of cosine sims: {len(cosine_sims)}')
    # now plot each
    # plt.plot(model_checkpoints, paired_l2_norms)
    plt.figure()
    # plt.hist(l2_norm_s1_s2, bins=20, alpha=0.7, label=model_checkpoints, color='blue')
    # plot mean and std of l2 norms
    mean_l2_norms = [np.mean(norms) for norms in paired_l2_norms]
    std_l2_norms = [np.std(norms) for norms in paired_l2_norms]
    plt.errorbar(model_checkpoints, mean_l2_norms, yerr=std_l2_norms, fmt='o', capsize=5, label='Mean ± Std Dev', color='blue', linestyle='-')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Pairwise L2 Norms')
    plt.title('Pairwise S1-S2 L2 Norms over Orthogonal Cone Init Model Training Process (n={})'.format(num_pairs))
    plt.grid()
    plt.savefig('pairwise-l2norm.png')
    plt.show()

    plt.figure()
    plt.plot(model_checkpoints, centroid_l2_norms, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Centroid L2 Norm')
    plt.title('Centroid S1-S2 L2 Norm over Orthogonal Cone Init Model Training Process (n={})'.format(num_pairs))
    plt.grid()
    plt.savefig('centroid-l2norm.png')
    plt.show()

    plt.figure()
    plt.plot(model_checkpoints, cosine_sims, marker='o')
    plt.xlabel('Model checkpoints')
    plt.ylabel('Centroid Cosine Sim')
    plt.title('Centroid S1-S2 Cosine Sim over Orthogonal Cone Init Model Training Process (n={})'.format(num_pairs))
    plt.grid()
    plt.savefig('pairwise-cosine-sim.png')
    plt.show()



 



if __name__ == "__main__":
    main()