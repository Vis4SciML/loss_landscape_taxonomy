
"""The following code is adapted from 
Similarity of Neural Network Representations Revisited
Simon Kornblith, Mohammad Norouzi, Honglak Lee and Geoffrey Hinton
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
"""
from __future__ import print_function

import warnings
from metric import Metric
from utils.feature_extractor import FeatureExtractor

import numpy as np


# ---------------------------------------------------------------------------- #
#                                CKA similarity                                #
# ---------------------------------------------------------------------------- #
        
class CKA(Metric):
    def __init__(self, model=None, data_loader=None, name="CKA_similarity", activation_layers=[]):
        super().__init__(model, data_loader, name)
        self.activation_layers = activation_layers
        self.results = {}   # there will be different values
    
    
    """Compute Gram (kernel) matrix for a linear kernel.
    Args:
        x: A num_examples x num_features matrix of features.
    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    def _gram_linear(self, x):
        return x.dot(x.T)
        
    """Compute Gram (kernel) matrix for a linear kernel.
    Args:
        x: A num_examples x num_features matrix of features.
    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    def _gram_linear(self, x):
        return x.dot(x.T)
        
    def lin_cka_dist(self, A, B):
        """
        Computes Linear CKA distance bewteen representations A and B
        """
        print(type(A), A.shape)
        print(type(B), B.shape)
        # center each row
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        

        # normalize each representation
        A = A / np.linalg.norm(A, ord="fro")
        B = B / np.linalg.norm(B, ord="fro")
        
        print(A)
        
        similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
        normalization = np.linalg.norm(A @ A.T, ord="fro") * \
                        np.linalg.norm(B @ B.T, ord="fro")
        return 1 - similarity / normalization
    
    def lin_cka_prime_dist(self, A, B):
        """
        Computes Linear CKA prime distance bewteen representations A and B
        The version here is suited to a, b >> n
        """
        # center each row
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)

        # normalize each representation
        A = A / np.linalg.norm(A, ord="fro")
        B = B / np.linalg.norm(B, ord="fro")

        if A.shape[0] > A.shape[1]:
            At_A = A.T @ A  # O(n * n * a)
            Bt_B = B.T @ B  # O(n * n * a)
            numerator = np.sum((At_A - Bt_B) ** 2)
            denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
            return numerator / denominator
        else:
            similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
            denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
            return 1 - 2 * similarity / denominator

    
    def _extract_features_from_model(self, target_model, data_loader, activations_layers):
        '''
        Utility methods used to extract the feature from each layer of the model.
        those features will be used to compute the CKA similarity temperature map.
        '''
        # do not train the network
        target_model.eval()
        # wrap the model with the feature extractor class
        feature_extractor = FeatureExtractor(target_model, activations_layers)
        # iterate over the the samples of the data_loader
        for batch in data_loader:
            features = feature_extractor.forward(batch)

        # remove possible tuples and none among the layers
        for name, feature in features.items():
            print(name)
            if feature is None:
                warnings.warn(f"Attention: the layer {name} has None features!")
            elif isinstance(feature, tuple):
                features[name] = feature[0].detach().numpy()
            else:
                features[name] = feature.detach().numpy()
        return features
        
    
    def compute(self):
        '''
        Compute the CKA similarity among the layers of the same model.
        '''
        print("Computing the CKA similarity...")
        # get the features of each layer
        features_per_layer = self._extract_features_from_model(self.model, 
                                                               self.data_loader,
                                                               self.activation_layers)
        cka_matrix = np.zeros((len(features_per_layer), len(features_per_layer)))
        
        # layers must be flatted
        for row, X in enumerate(features_per_layer.values()):
            X = X.reshape(X.shape[0], -1)
            for col, Y in enumerate(features_per_layer.values()):
                Y = Y.reshape(Y.shape[0], -1)
                cka_matrix[row, col] = self.lin_cka_prime_dist(X, Y)
                
        self.results = {'cka_dist': cka_matrix}
        
        return self.results
    
    '''
    Compare two models with the CKA similarity'''
    def compare(self, model, data_loader, activation_layers):
        # get the features of each layer
        features_per_layer1 = self._extract_features_from_model(self.model, 
                                                                self.data_loader,
                                                                self.activation_layers)
        
        features_per_layer2 = self._extract_features_from_model(model, 
                                                                data_loader,
                                                                activation_layers)
        
        cka_matrix = np.zeros((len(features_per_layer1), len(features_per_layer2)))
        # layers must be flatted
        for row, X in enumerate(features_per_layer1.values()):
            X = X.reshape(X.shape[0], -1)
            for col, Y in enumerate(features_per_layer2.values()):
                Y = Y.reshape(Y.shape[0], -1)
                cka_matrix[row, col] = self.lin_cka_prime_dist(X, Y)
                
        self.results = {'cka_dist': cka_matrix}
        
        return self.results
