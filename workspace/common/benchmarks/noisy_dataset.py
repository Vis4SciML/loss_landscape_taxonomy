import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

from noise import Noise

# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import rn08


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


    def add_noise(self, sample):
        noisy_sample = self.noise_adder(sample, self.percentage)
        noisy_sample = np.float32(noisy_sample)
        return torch.from_numpy(noisy_sample)

    
    def __len__(self):
        return len(self.original_dataloader.dataset)
    
    
    def __getitem__(self, index):
        # check if it is a tuple
        sample = self.original_dataloader.dataset[index]
        
        if isinstance(sample, tuple):
            original_batch, target = sample
            noisy_batch = self.add_noise(original_batch)
            return noisy_batch, target
        
        noisy_sample = self.add_noise(sample)
        # add the noise
        
        return noisy_sample
    
    
# test 
if __name__ == "__main__":
    train_dataloader, _, _ = rn08.get_cifar10_loaders("../../../data/RN08", 1)
    
    print("test tuple")
    noisy_dataset = NoisyDataset(train_dataloader, 5, 'gaussian')
    noisy_dataloader = DataLoader(noisy_dataset, 1, False)
    print("Without noise")
    for batch, target in train_dataloader:
        print('batch:', batch.shape)
        print('target:', target.shape)
        break
    
    print("With noise")
    for batch, target in noisy_dataloader:
        print('noisy batch:', batch.shape)
        print('target:', target.shape)
        break
    
    # print("test single")
    # noisy_dataset = NoisyDataset(data_module.val_dataloader(), 0, 'gaussian')
    # dataloader = DataLoader(noisy_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    print("Without noise")
    for batch in train_dataloader:
        print('batch:', batch)
        break
    
    print("With noise")
    for batch in noisy_dataloader:
        print('noisy batch:', batch)
        break
