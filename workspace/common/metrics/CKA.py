
"""The following code is adapted from:
DO WIDE AND DEEP NETWORKS LEARN THE SAME
THINGS? UNCOVERING HOW NEURAL NETWORK
REPRESENTATIONS VARY WITH WIDTH AND DEPTH
Thao Nguyen, AI Resident, Google Research
https://blog.research.google/2021/05/do-wide-and-deep-networks-learn-same.html
"""
from __future__ import print_function
import os 
import sys
import torch
import math
from statistics import mean
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import warnings
from metric import Metric
from utils.feature_extractor import FeatureExtractor

module_path = os.path.abspath(os.path.join('../../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from noisy_dataset import NoisyDataset

# ---------------------------------------------------------------------------- #
#                                CKA similarity                                #
# ---------------------------------------------------------------------------- #
        
class CKA(Metric):
    
    def __init__(self, model, data_loader, name="CKA_similarity", layers=[], max_batches=100):
        super().__init__(model, data_loader, name)
        self.layers = layers
        self.max_batches = max_batches
        self.results = {}   # there will be different values
        self.device = 'cpu'
        # CKA is measured on perturbed training set comprised of mixup samples
        noisy_dataset = NoisyDataset(data_loader, 2, 'gaussian')
        self.data_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=True)
        
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'

    
    @staticmethod
    def gram_matrix(X):
        '''
        Generate Gram matrix and preprocess to compute unbiased HSIC.

        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.

        Args:
        x: A [num_examples, num_features] matrix.

        Returns:
        A [num_examples ** 2] vector.
        '''
        X = X.reshape(X.shape[0], -1)
        gram_X = X @ X.T
        
        if not torch.allclose(gram_X, gram_X.t()):
            raise ValueError("Gram matrix should be symmetric!")
        
        means = torch.mean(gram_X, 0)
        means -= torch.mean(means) / 2
        gram_X -= means[:, None]
        gram_X -= means[None, :]
        
        # n = gram_X.shape[0]
        # gram_X.fill_diagonal_(0)
        # means = torch.sum(gram_X, axis=0) / (n - 2)
        # means -= torch.sum(means) / 2 * (n - 1)
        # gram_X -= means[:, None]
        # gram_X -= means[None, :]
        # gram_X.fill_diagonal_(0)
        return gram_X.reshape((-1,))
    
    
    @staticmethod
    def update_state(hsic_accumulator, activations):
        layers_gram = []
        for x in activations.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram.append(CKA.gram_matrix(x))
        layers_gram = torch.stack(layers_gram, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram, layers_gram.T)
    
    
    @staticmethod
    def update_state_across_models(hsic_accumulator,
                                   hsic_accumulator1, 
                                   activations1, 
                                   hsic_accumulator2, 
                                   activations2):
        # dimension test
        torch.testing.assert_close(hsic_accumulator1.shape[0], len(activations1))
        torch.testing.assert_close(hsic_accumulator2.shape[0], len(activations2))
        # activation 1
        layers_gram1 = []
        for x in activations1.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram1.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram1.append(CKA.gram_matrix(x))
        layers_gram1 = torch.stack(layers_gram1, axis=0)
        # activation 2
        layers_gram2 = []
        for x in activations2.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram2.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram2.append(CKA.gram_matrix(x))
        layers_gram2 = torch.stack(layers_gram2, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram1, layers_gram2.T), \
                hsic_accumulator1 + torch.einsum('ij,ij->i', layers_gram1, layers_gram1), \
                hsic_accumulator2 + torch.einsum('ij,ij->i', layers_gram2, layers_gram2)
    
    
    def compare(self, model, layers=None):
        '''
        Compare the CKA similarity between the layers of two models
        '''
        # second model can have different layers
        layers1 = self.layers
        layers2 = self.layers
        if layers is not None:
            layers2 = layers
        num_layers1 = len(layers1)
        num_layers2 = len(layers2)
        
        hsic_accumulator = torch.zeros((num_layers1, num_layers2), device=self.device, dtype=torch.float32)
        hsic_accumulator1 = torch.zeros((num_layers1,), device=self.device, dtype=torch.float32)
        hsic_accumulator2 = torch.zeros((num_layers2,), device=self.device, dtype=torch.float32)
        
        self.model.eval()
        model.eval()
        
        model1 = FeatureExtractor(self.model, layers1)
        model2 = FeatureExtractor(model, layers2)
        
        model2.to(self.device)
        
        count = 0
        for batch in self.data_loader:
            count += 1
            # remove the label from the tuple
            if isinstance(batch, list):
                batch = batch[0]
                            
            activations1 = model1.forward(batch)
            activations2 = model2.forward(batch)
            hsic_accumulator, hsic_accumulator1, hsic_accumulator2 = \
                                            CKA.update_state_across_models(hsic_accumulator,
                                                                           hsic_accumulator1,
                                                                           activations1,
                                                                           hsic_accumulator2,
                                                                           activations2)
            if count == self.max_batches:
                break
            
        mean_hsic = hsic_accumulator
        normalization1 = torch.sqrt(hsic_accumulator1)
        normalization2 = torch.sqrt(hsic_accumulator2)
        mean_hsic /= normalization1[:, None]
        mean_hsic /= normalization2[None, :]
                
        self.results['cka_dist'] = 1 - torch.mean(torch.diagonal(mean_hsic)).item()
        self.results['compared_cka'] = mean_hsic
        return self.results
    
    def compare_output(self, model, num_outputs=10, num_runs=5):
        '''
        Compare the CKA similarity between the outputs of two models
        '''
        
        cka_similarity = []
        for _ in range(num_runs):
            # output of models
            F1 = []
            F2 = []
            for i, batch in enumerate(self.data_loader, 1):
                # stop condition
                if i > num_outputs:
                    break
                # remove the label from the tuple
                if isinstance(batch, list):
                    batch = batch[0]
                
                F1.append(self.model(batch))
                F2.append(model(batch))
            
            F1 = torch.cat(F1)
            F2 = torch.cat(F2)
            
            gram_x = CKA.gram_matrix(F1)
            gram_y = CKA.gram_matrix(F2)
            
            scaled_hsic = torch.ravel(gram_x) @ torch.ravel(gram_y)
            normalization_x = torch.linalg.norm(gram_x)
            normalization_y = torch.linalg.norm(gram_y)
            s = scaled_hsic / (normalization_x * normalization_y)
            if torch.isnan(s):
                cka_similarity.append(0.0)
            else:
                cka_similarity.append(s.item())
        avg_s = mean(cka_similarity)

        if math.isnan(avg_s) or avg_s < 0 or math.isinf(avg_s):
            avg_s = 0
            
        return avg_s
    
    def compute(self):
        '''
        Compare the CKA similarity among the layers of a model.
        '''
        num_layers = len(self.layers)
        hsic_accumulator = torch.zeros((num_layers, num_layers), device=self.device, dtype=torch.float32)
        # bind a hook to the outputs of the models' layers
        model = FeatureExtractor(self.model, self.layers)
        model.eval()

        # iterate over the dataset
        count = 0
        for batch in self.data_loader:
            count += 1
            # remove the label from the tuple
            if isinstance(batch, list):
                batch = batch[0]
                
            # compare the layers one against the other
            activations = model.forward(batch)
            hsic_accumulator = CKA.update_state(hsic_accumulator, activations)
            if count == self.max_batches:
                break
                
        mean_hsic = hsic_accumulator
        normalization = torch.sqrt(torch.diagonal(hsic_accumulator))
        mean_hsic = mean_hsic / normalization[:, None]
        mean_hsic = mean_hsic / normalization[None, :]
            
        self.results['internal_cka'] = mean_hsic
        return self.results
            

# test
# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
# sys.path.insert(0, module_path)
# import rn08
# DATA_PATH = "/home/jovyan/checkpoint/"
# DATASET_PATH = "../../../data/RN08"
    
# if __name__ == "__main__":
#     model, acc = rn08.get_model_and_accuracy(DATA_PATH, 1024, 0.1, 11)
#     dataloader = rn08.get_dataloader(DATASET_PATH, 1)
#     print(f'accuracy: {acc}')
#     layers = [
#         'model.conv1', 
#         'model.QBlocks.0.conv1', 
#         'model.QBlocks.0.conv2', 
#         'model.QBlocks.1.conv1', 
#         'model.QBlocks.1.conv2',  
#         'model.QBlocks.2.conv1', 
#         'model.QBlocks.2.conv2',
#         'model.QBlocks.2.shortcut',
#         'model.QBlocks.3.conv1', 
#         'model.QBlocks.3.conv2', 
#         'model.QBlocks.4.conv1', 
#         'model.QBlocks.4.conv2',
#         'model.QBlocks.4.shortcut',
#         'model.QBlocks.5.conv1', 
#         'model.QBlocks.5.conv2', 
#         'model.linear'
#     ]
    
#     cka = CKA(model, dataloader, layers=layers, max_batches=50)
#     # print(cka.compute())
#     batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
#     for bs in batch_sizes:
#         model2, acc = rn08.get_model_and_accuracy(DATA_PATH, bs, 0.1, 11)
#         print(f'accuracy {bs}: {acc}')
#         print(cka.compare_output(model2, 10))