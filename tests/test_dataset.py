# import sys
# sys.path.append('../')


from ciip.dataset import S12Dataset
from torch.utils.data import DataLoader
# import torch

root = '~/Documents/research/geohai/ssl4eo-s12_100patches/'

dataset = S12Dataset(root)

print(f'Dataset length={dataset.__len__()}')

print(f'Sample 50: {dataset.__getitem__(50)}')
print(f'Sample 80: {dataset.__getitem__(80)}')
print(f'Sample 3: {dataset.__getitem__(3)}')
print(f'Sample 25: {dataset.__getitem__(25)}')
