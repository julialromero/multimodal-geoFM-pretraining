# code for pytorch dataset class

import logging
import os
import random

try:  # pragma: no cover - exercised when numpy is missing
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled by runtime checks
    np = None  # type: ignore

try:  # pragma: no cover - exercised when rasterio is missing
    import rasterio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled by runtime checks
    rasterio = None  # type: ignore

try:  # pragma: no cover - exercised when torch is missing
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - fallback for tests
    class Dataset:  # type: ignore
        """Minimal stand-in for :class:`torch.utils.data.Dataset`."""

        def __getitem__(self, idx):  # pragma: no cover - interface definition
            raise NotImplementedError

        def __len__(self):  # pragma: no cover - interface definition
            raise NotImplementedError


def _require_numpy() -> "np":  # type: ignore[name-defined]
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for dataset operations. Install numpy to work with raster imagery."
        )
    return np  # type: ignore[return-value]


def _require_rasterio():
    if rasterio is None:
        raise ModuleNotFoundError(
            "rasterio is required for dataset operations. Install rasterio to read GeoTIFF inputs."
        )
    return rasterio


### band statistics: mean & std
# calculated from 50k data
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]




class S12Dataset(Dataset):
    def __init__(self, root, ):
        _require_numpy()
        _require_rasterio()

        self.root = root
        self.num_locations = None
        self.length = None
        self.s1_paths = []
        self.s2_paths = []


        s1_dir = os.path.join(self.root, 's1')
        s2_dir = os.path.join(self.root, 's2c')


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

        logging.debug('Done loading data.')


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        ### get the sample corresponding to idx
        np_module = _require_numpy()
        location_idx, season_idx = self.int_to_filepath(idx)

        location_folder = self.locations[location_idx]

        path_to_s1 = os.path.join(self.root, 's1', location_folder)
        path_to_s2 = os.path.join(self.root, 's2c', location_folder)

        s1_season_folders = sorted(os.listdir(path_to_s1))
        s2_season_folders = sorted(os.listdir(path_to_s2))

        path_to_s1_season = os.path.join(path_to_s1, s1_season_folders[season_idx])
        path_to_s2_season = os.path.join(path_to_s2, s2_season_folders[season_idx])

        print(f'Path to s1: {path_to_s1_season}')
        print(f'Path to s2: {path_to_s2_season}')


        ### load and stack s1 images
        vh_path = os.path.join(path_to_s1_season, 'VH.tif')
        vv_path = os.path.join(path_to_s1_season, 'VV.tif')
        vh_image, _ = self.read_raster_image(vh_path)
        vv_image, _ = self.read_raster_image(vv_path)

        # Normalize the VH and VV bands
        # vh_image = self.normalize_image(vh_image)
        # vv_image = self.normalize_image(vv_image)
        vv_image = self.normalize(vv_image, S1_MEAN[0], S1_STD[0])
        vh_image = self.normalize(vh_image, S1_MEAN[1], S1_STD[1])
        

                
        # Create an RGB composite using VH, VV, and their average
        s1_composite_image = np_module.stack((vh_image, vv_image, (vh_image + vv_image) / 2), axis=-1)
        # s1_composite_image = np.stack((vh_image, vv_image, vv_image / vh_image), axis=-1)
        

        ### load and stack s2 images
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
        # band_images = [self.normalize_image(band_image) for band_image in band_images]
        band_images = [self.normalize(img, mean, std) for img, mean, std in zip(band_images, S2C_MEAN, S2C_STD)]

        s2_composite_image = np_module.stack(band_images, axis=-1)  # Create an RGB composite
        

        # img1 = self.transforms(Image.open(str(self.img1_paths[idx])))
        # img2 = self.transforms(Image.open(str(self.img2_paths[idx])))
        return (s1_composite_image, s2_composite_image)


    def int_to_filepath(self, i):
        if i < 0 or i >= self.length:
            raise ValueError("Integer i must be in the range [0, n)")
        
        location_idx = i // 4
        season_idx = i % 4
        
        return location_idx, season_idx

    def read_raster_image(self, image_path):
        rasterio_module = _require_rasterio()
        with rasterio_module.open(image_path) as src:
            return src.read(1), src.profile

    #### our custom normalization function for image-wise norm
    # def normalize_image(self, image):
    #     """Normalize image data to the range [0, 1]"""
    #     image_min, image_max = np.min(image), np.max(image)
    #     return (image - image_min) / (image_max - image_min)
    

    #### band-wise normalization based on mean and std calculate from 50k data
    ## taken from: https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py#L36
    # normalize: standardize + percentile
    def normalize(img, mean, std):
        np_module = _require_numpy()
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value) * 255.0
        img = np_module.clip(img, 0, 255).astype(np_module.uint8)
        return img


# taken from: https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py#L127
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


def generate_splits(dataset: Dataset, val_frac: float, seed: int | None = None):
    """Split a dataset into train and validation subsets.

    The implementation only relies on the Python standard library so it works in
    lightweight environments where optional dependencies such as NumPy are not
    available.
    """

    if not 0 <= val_frac <= 1:
        raise ValueError("val_frac must be between 0 and 1 inclusive")

    total_examples = len(dataset)
    if total_examples == 0:
        return Subset(dataset, []), Subset(dataset, [])

    indices = list(range(total_examples))
    validation_count = int(val_frac * total_examples)

    rng = random.Random(seed)
    validation_indices = rng.sample(indices, validation_count)

    validation_set = set(validation_indices)
    training_indices = [index for index in indices if index not in validation_set]

    return Subset(dataset, training_indices), Subset(dataset, validation_indices)

