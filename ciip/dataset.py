# code for pytorch dataset class
# load samples and ground truth label


# from chatgpt
# args.csv_img1_key = 'image1_path'
# args.csv_img2_key = 'image2_path'
# args.csv_label_key = 'label'
# args.csv_separator = ','
import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None





class S12Dataset(Dataset):
    def __init__(self, root, transforms, img1_key, img2_key, label_key, sep="\t"):
        self.root = root
        self.num_locations = None
        self.length = None
        self.s1_paths = []
        self.s2_paths = []


        s1_dir = os.path.join(self.root, 's1')
        s2_dir = os.path.join(self.root, 's2-l2a')


        s1_samples = os.listdir(s1_dir)
        s2_samples = os.listdir(s2_dir)

        ##### NOT NEEDED FOR NON-MAC OS #####
        # remove DS store file
        for file in s1_samples:
            if file.startswith('.'):
                s1_samples.remove(file)
                s2_samples.remove(file)
        #####################################


        assert len(s1_samples) == len(s2_samples), 'Number of locations in S1 and S2 should be the same'


        self.num_locations = len(s1_samples)
        self.length = self.num_locations * 4

        print(f'Number of locations in dataset: {self.num_locations}')


        # location list
        self.locations = s1_samples


        # create path lists
        s1_location_paths = [os.path.join(s1_dir, location_id) for location_id in self.locations]
        s2_location_paths = [os.path.join(s2_dir, location_id) for location_id in self.locations]


        # YYYYMMDD
        # different seasons are negative samples

        logging.debug('Done loading data.')



    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        ### get the sample corresponding to idx
        location_idx, season_idx = self.int_to_filepath(idx)

        location_folder = self.locations[location_idx]

        path_to_s1 = os.path.join(self.root, 's1', location_folder)
        path_to_s2 = os.path.join(self.root, 's2-l2a', location_folder)

        s1_season_folders = sorted(os.listdir(path_to_s1))
        s2_season_folders = sorted(os.listdir(path_to_s2))

        path_to_s1_season = os.path.join(path_to_s1, s1_season_folders[season_idx])
        path_to_s2_season = os.path.join(path_to_s1, s2_season_folders[season_idx])

        ### load and stack s1 images
        vh_path = os.path.join(path_to_s1_season, 'VH.tif')
        vv_path = os.path.join(path_to_s1_season, 'VV.tif')
        vh_image, _ = self.read_raster_image(vh_path)
        vv_image, _ = self.read_raster_iamge(vv_path)

        # Normalize the VH and VV bands
        vh_image = self.normalize_image(vh_image)
        vv_image = self.normalize_image(vv_image)
                
        # Create an RGB composite using VH, VV, and their average
        # composite_image = np.stack((vh_image, vv_image, (vh_image + vv_image) / 2), axis=-1)
        s1_composite_image = np.stack((vh_image, vv_image, vv_image / vh_image), axis=-1)
                
        
        # Use Blue (B2), Green (B3), and Red (B4) bands for Sentinel-2 RGB composite
        # band_paths = [os.path.join(path_to_s2_season, f'B{band}.tif') for band in [2, 3, 4]]
        # stack every S2 band instead of just RGB
        band_paths = [os.path.join(path_to_s2_season, f'B{band}.tif') for band in range(1, 13)]
 
       # Ensure all bands have the same shape
        band_images = [self.read_raster_image(band_path)[0] for band_path in band_paths]
        shapes = [band_image.shape for band_image in band_images]
        min_shape = min(shapes, key=lambda x: x[0] * x[1])
        band_images = [band_image[:min_shape[0], :min_shape[1]] for band_image in band_images]
        
        # Normalize the bands
        band_images = [self.normalize_image(band_image) for band_image in band_images]
        
        s2_composite_image = np.stack(band_images, axis=-1)  # Create an RGB composite
        

        # img1 = self.transforms(Image.open(str(self.img1_paths[idx])))
        # img2 = self.transforms(Image.open(str(self.img2_paths[idx])))
        return (s1_composite_image, s2_composite_image)


    def int_to_filepath(self, i):
        if i < 0 or i >= self.length:
            raise ValueError("Integer i must be in the range [0, n)")
        
        location_idx = i // 4
        season_idx = i % 4
        
        return location_idx, season_idx

    def read_raster_image(image_path):
        with rasterio.open(image_path) as src:
            return src.read(1), src.profile

    def normalize_image(image):
        """Normalize image data to the range [0, 1]"""
        image_min, image_max = np.min(image), np.max(image)
        return (image - image_min) / (image_max - image_min)

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for (img1, img2), labels in dataloader:
        n_batches += 1
        n_elements += len(labels)
        assert len(img1) == len(img2) == len(labels)
    return n_elements, n_batches


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = ImagePairDataset(
        input_filename,
        preprocess_fn,
        img1_key=args.csv_img1_key,
        img2_key=args.csv_img2_key,
        label_key=args.csv_label_key,
        sep=args.csv_separator,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0):
    class SyntheticImagePairDataset(Dataset):
        def __init__(self, transform=None, image_size=(224, 224), dataset_size=100):
            self.transform = transform
            self.image_size = image_size
            self.image = Image.new('RGB', image_size)
            self.dataset_size = dataset_size

        def __len__(self):
            return self.dataset_size

        def __getitem__(self, idx):
            if self.transform is not None:
                img1 = self.transform(self.image)
                img2 = self.transform(self.image)
            label = torch.tensor(1.0 if idx == 0 else 0.0, dtype=torch.float32)
            return (img1, img2), label

    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticImagePairDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    return data

