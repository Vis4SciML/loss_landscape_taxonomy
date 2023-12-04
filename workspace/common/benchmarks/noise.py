import numpy as np
import torch

class Noise:
    '''
    Class with static methods to add different kind of noise
    to the input data
    '''
    
    @staticmethod
    def add_random_perturbation(data, percentage, perturbation_range=0.1):
        perturbation = np.random.uniform(-perturbation_range, perturbation_range, data.shape)
        noisy_data = data + (percentage / 100) * perturbation
        return noisy_data
    
    
    @staticmethod
    def add_gaussian_noise(data, percentage=5, mean=0, std=1):
        noise = np.random.normal(mean, std, data.shape)
        noisy_data = data + (percentage / 100) * noise
        return noisy_data
    
    
    @staticmethod
    def add_salt_and_pepper_noise(data, percentage=5):
        noisy_data = data.clone()
        
        # salt noise
        salt_mask = np.random.rand(*data.shape) < (percentage / 200)
        noisy_data[[salt_mask]] = 1.0
        
        # pepper noise
        pepper_mask = np.random.rand(*data.shape) < (percentage / 200)
        noisy_data[[pepper_mask]] = 0.0
        
        return noisy_data
    
    

# test 
if __name__ == "__main__":
    original_data = torch.Tensor([[[1, 2, 3, 4, 5]]])
    
    noisy_data = Noise.add_gaussian_noise(original_data, 5)
    print('Gaussin noise:', noisy_data)
    
    noisy_data = Noise.add_random_perturbation(original_data, 5)
    print('Random noise:', noisy_data)
    
    noisy_data = Noise.add_salt_and_pepper_noise(original_data, 20)
    print('Salt & Pepper noise:', noisy_data)