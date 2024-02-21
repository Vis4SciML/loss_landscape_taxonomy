import numpy as np
import torch

class Noise:
    '''
    Class with static methods to add different kind of noise
    to the input data
    '''
    
    @staticmethod
    def add_random_perturbation(data, percentage, perturbation_range=0.1):
        if isinstance(data, torch.Tensor):
            data = data.clone()
        else:
            data = data.copy()
        
        perturbation = np.random.uniform(-perturbation_range, perturbation_range, data.shape)
        noisy_data = data + (percentage / 100) * perturbation
        return noisy_data
    
    
    @staticmethod
    def add_gaussian_noise(data, percentage=5, mean=0, std=1):
        if isinstance(data, torch.Tensor):
            data = data.clone()
        else:
            data = data.copy()
        
        noise = np.random.normal(mean, std, data.shape)
        noisy_data = data + (percentage / 100) * noise
        return noisy_data
    
    
    @staticmethod
    def add_salt_and_pepper_noise(data, percentage=5):
        if isinstance(data, torch.Tensor):
            data = data.clone().cpu().numpy()
        elif isinstance(data, np.ndarray):
            data = data.copy()
        else:
            raise ValueError("Unsupported data type. Only PyTorch tensors and NumPy ndarrays are supported.")
        
        # Calculate the number of pixels to corrupt
        num_pixels = data.size * (percentage / 100)
        
        # Add salt noise
        salt_coords = [np.random.randint(0, i, int(num_pixels)) for i in data.shape]
        data[tuple(salt_coords)] = 1.0
        
        # Add pepper noise
        pepper_coords = [np.random.randint(0, i, int(num_pixels)) for i in data.shape]
        data[tuple(pepper_coords)] = 0.0
        
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return torch.from_numpy(data).to(device=data.device)
    


# test 
if __name__ == "__main__":
    original_data = torch.Tensor([[[1, 2, 3, 4, 5]]])
    
    noisy_data = Noise.add_gaussian_noise(original_data, 5)
    print('Gaussin noise:', noisy_data)
    
    noisy_data = Noise.add_random_perturbation(original_data, 5)
    print('Random noise:', noisy_data)
    
    noisy_data = Noise.add_salt_and_pepper_noise(original_data, 20)
    print('Salt & Pepper noise:', noisy_data)