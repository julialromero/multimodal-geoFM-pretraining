import os, sys
sys.path.append(os.path.abspath(os.path.join('..', 'ciip/ciip')))
print(os.path.abspath(os.path.join('..', 'ciip/ciip')))



from ciip.dataset import S12Dataset
from torch.utils.data import DataLoader
# import torch

root = '/home/juro4948/ciip/'

dataset = S12Dataset(root)

print(f'Dataset length={dataset.__len__()}')

print(f'Sample 0: {dataset.__getitem__(0)}')
print(f'Sample 1: {dataset.__getitem__(1)}')
print(f'Sample 4: {dataset.__getitem__(4)}')
print(f'Sample 6: {dataset.__getitem__(6)}')
