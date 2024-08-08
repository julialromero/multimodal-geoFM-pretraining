# code for pytorch dataset class

import logging
import os
import numpy as np
import rasterio
from torch.utils.data import Dataset

class S12Dataset(Dataset):
    def __init__(self, root):
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


        # create path lists
        s1_location_paths = [os.path.join(s1_dir, location_id) for location_id in self.locations]
        s2_location_paths = [os.path.join(s2_dir, location_id) for location_id in self.locations]


        logging.debug('Done loading data.')


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        ### get the sample corresponding to idx
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
        vh_image = self.normalize_image(vh_image)
        vv_image = self.normalize_image(vv_image)
                
        # Create an RGB composite using VH, VV, and their average
        # composite_image = np.stack((vh_image, vv_image, (vh_image + vv_image) / 2), axis=-1)
        s1_composite_image = np.stack((vh_image, vv_image, vv_image / vh_image), axis=-1)
        

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

    def read_raster_image(self, image_path):
        with rasterio.open(image_path) as src:
            return src.read(1), src.profile

    def normalize_image(self, image):
        """Normalize image data to the range [0, 1]"""
        image_min, image_max = np.min(image), np.max(image)
        return (image - image_min) / (image_max - image_min)
