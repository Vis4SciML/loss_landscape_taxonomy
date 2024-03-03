import torch
from torchvision import transforms

import numpy as np
import os
import sys

# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import rn08

        
@staticmethod
def fgsm_attack(input, epsilon, data_grad, min=0, max=1):
    sign_data_grad = data_grad.sign()
    perturbed_input = input + epsilon*sign_data_grad
    perturbed_input = torch.clamp(perturbed_input, min, max)
    return perturbed_input        
    
   