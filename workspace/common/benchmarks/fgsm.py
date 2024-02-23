import torch
from torchvision import transforms

import numpy as np
import os
import sys

# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import rn08


class FSGM:
    def __init__(self, model, dataset, normalize, denormalize):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.normalize = normalize
        self.denormalize = denormalize
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'
        
    @staticmethod
    def fsgm_attack(input, epsilon, data_grad, min=0, max=1):
        sign = data_grad.sign()
        noisy_input = input + epsilon * sign 
        
        noisy_input = torch.clamp(noisy_input, min, max)
        
        return noisy_input
    
    def test(self, epsilon):
        correct = 0
        adv_example = []
        
        for input, target in self.dataset:
            input, target = input.to(self.device), target.to(self.device)
            
            input.require_grad = True
            
            output = self.model(input)
            init_pred =output.max(1, keepdim=True)[1]
            
            # if the model already miss predict continue
            if init_pred.item() != target.item():
                continue
            
            loss = self.model.loss(output, target)
            
            self.model.zero_grad()
            loss.backward()

            input_grad = input.grad.data
            
            input_denorm = self.denormalize(input)
            
            perturbed_input = FSGM.fsgm_attack(input_denorm, epsilon, input_grad)
            
            perturbed_input_norm = self.normalize(perturbed_input)
            
            output = self.model(perturbed_input_norm)
            
            final_pred = output.max(1, keepdim=True)[1]
            
            if final_pred.item() == target.item():
                correct += 1
                
            adv_ex = perturbed_input.squeeze().detach().cpu().numpy()
            adv_example.append(adv_ex)

        final_acc = correct/float(len(self.dataset))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc}")
        
        return final_acc, adv_example
    
    
DATA_PATH = "/data/tbaldi/work/checkpoint"
DATA_FILE = "../../../data/RN08"

def denorm(batch, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (tuple): Mean used for normalization.
        std (tuple): Standard deviation used for normalization.

    Returns:
        torch.Tensor: Batch of tensors without normalization applied to them.
    """
    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1)

    return batch * std_tensor + mean_tensor

if __name__ == "__main__":
    
    model, acc = rn08.get_model_and_accuracy(DATA_PATH, 128, 0.00625, 8)
    _, _, datalaoder = rn08.get_cifar10_loaders(DATA_FILE, 1)
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    
    normalize = transforms.Normalize(mean=mean, std=std)

    
    denormalize = transforms.Normalize(
                    mean=(-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                    std=(1 / std[0], 1 / std[1], 1 / std[2])
                )


    
    for image, _ in datalaoder:
        print(image.shape)
        denorm_image = denorm(image)
        norm_image = normalize(denorm_image)
        
        image_np = image.numpy()
        denorm_image_np = denorm_image.numpy()
        norm_image_np = norm_image.numpy()
        # Print the tensor shapes
        print("Original Image Shape:", image_np.shape)
        print("Denormalized Image Shape:", denorm_image_np.shape)
        print("Normalized Image Shape:", norm_image_np.shape)

        # Print the range of pixel values for each image
        print("\nOriginal Image Pixel Range:", np.min(image_np), np.max(image_np))
        print("Denormalized Image Pixel Range:", np.min(denorm_image_np), np.max(denorm_image_np))
        print("Normalized Image Pixel Range:", np.min(norm_image_np), np.max(norm_image_np))
        break
    
    
    