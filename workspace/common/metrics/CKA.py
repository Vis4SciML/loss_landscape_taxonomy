
"""The following code is adapted from:
DO WIDE AND DEEP NETWORKS LEARN THE SAME
THINGS? UNCOVERING HOW NEURAL NETWORK
REPRESENTATIONS VARY WITH WIDTH AND DEPTH
Thao Nguyen, AI Resident, Google Research
https://blog.research.google/2021/05/do-wide-and-deep-networks-learn-same.html
"""
from __future__ import print_function

import torch
import warnings
from metric import Metric
from utils.feature_extractor import FeatureExtractor


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
        n = gram_X.shape[0]
        gram_X.fill_diagonal_(0)
        means = torch.sum(gram_X, axis=0) / (n - 2)
        means -= torch.sum(means) / 2 * (n - 1)
        gram_X -= means[:, None]
        gram_X -= means[None, :]
        gram_X.fill_diagonal_(0)
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
            

    
    