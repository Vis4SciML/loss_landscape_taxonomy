import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

from noise import Noise

# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/econ/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from autoencoder_datamodule import AutoEncoderDataModule



class NoisyDataset(Dataset):
    '''
    Class used to wrap a dataloader and add noise to it's input.
    '''
    def __init__(self, original_dataloader, percentage=5, noise_type='random'):
        self.original_dataloader = original_dataloader
        self.percentage = percentage
        
        # select the function to add noise
        self.noise_adder = None
        if noise_type == 'random':
            self.noise_adder = Noise.add_random_perturbation
        elif noise_type == 'gaussian':
            self.noise_adder = Noise.add_gaussian_noise
        elif noise_type == 'salt_pepper':
            self.noise_adder = Noise.add_salt_and_pepper_noise
        else:
            warnings.warn("Warn: not valid noise type.")
            self.noise_adder = Noise.add_random_perturbation
            
    
    def __len__(self):
        return len(self.original_dataloader.dataset)
    
    
    def __getitem__(self, index):
        original_batch, target = self.original_dataloader.dataset[index]
        # add the noise
        original_batch = torch.Tensor.numpy(original_batch)
        noisy_batch = self.noise_adder(original_batch, self.percentage)
        noisy_batch = np.float32(noisy_batch)
        noisy_batch = torch.from_numpy(noisy_batch)
        return noisy_batch, target
    
    
# test 
if __name__ == "__main__":
    DATASET_DIR = '../../../data/ECON/Elegun'
    DATASET_FILE = 'nELinks5.npy'
    # get the datamodule
    data_module = AutoEncoderDataModule(
        data_dir=DATASET_DIR,
        data_file=os.path.join(DATASET_DIR, DATASET_FILE),
        batch_size=1,
        num_workers=4)
    
    # check if we have processed the data
    if not os.path.exists(os.path.join(DATASET_DIR, DATASET_FILE)):
        print('Processing the data...')
        data_module.process_data(save=True)

    data_module.setup(0)
    
    noisy_dataloader = NoisyDataset(data_module.dataloaders()[1], 10, 'random')
    
    print("Without noise")
    for batch, target in data_module.dataloaders()[1]:
        print('batch:', batch)
        print('target:', target)
        break
    
    print("With noise")
    for batch, target in noisy_dataloader:
        print('noisy batch:', batch)
        print('target:', target)
        break
