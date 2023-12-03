
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
import pandas as pd


# ---------------------------------------------------------------------------- #
#                                CKA similarity                                #
# ---------------------------------------------------------------------------- #
        
class CKA(Metric):
    def __init__(self, model=None, name="CKA_similarity", layers=[]):
        super().__init__(model, None, name)
        self.layers = layers
        self.results = {}   # there will be different values

    @staticmethod
    def extract_features_from_model(target_model, layers):
        '''
        Utility methods used to extract the feature from each layer of the model.
        those features will be used to compute the CKA similarity temperature map.
        '''
            
        structure = {}
        # remove possible tuples and none among the layers
        for name, params in target_model.named_parameters():
            if 'weight' in name:
                name = name.replace('.weight', '')
                if name in layers:
                    structure[name] = params.detach().numpy()
                    
        return structure
    

    @staticmethod
    def lin_cka_dist(A, B):
        """
        Computes Linear CKA distance between representations A and B
        """

        # center each row
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        # normalize each representation
        A = A / np.linalg.norm(A)
        B = B / np.linalg.norm(B)
                
        
        similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
        normalization = np.linalg.norm(A @ A.T, ord="fro") * \
                        np.linalg.norm(B @ B.T, ord="fro")
        return 1 - similarity / normalization


    @staticmethod
    def lin_cka_prime_dist(A, B):
        """
        Computes Linear CKA prime distance between representations A and B
        The version here is suited to a, b >> n
        """
        # check if the matrix contains only zeros
        if np.any(A):
            # center each row
            A = A - A.mean(axis=1, keepdims=True)
            # normalize each representation
            A = A / np.linalg.norm(A, ord="fro")
        else:
            warnings.warn("Warning: matrix A contains only zeros!")
            
        if np.any(B):
            B = B - B.mean(axis=1, keepdims=True)
            B = B / np.linalg.norm(B, ord="fro")
        else:
            warnings.warn("Warning: matrix B contains only zeros!")
        
        
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
    
    
    def compare(self, model, layers):
        '''
        Compare two models with the CKA similarity
        '''
        # get the features of each layer
        features_per_layer1 = CKA.extract_features_from_model(self.model, self.layers)
        features_per_layer1 = {'[1] ' + key: value for key, value in features_per_layer1.items()}
        
        features_per_layer2 = CKA.extract_features_from_model(model, layers)
        features_per_layer2 = {'[2] ' + key: value for key, value in features_per_layer2.items()}
        
        # init the matrix
        cka_matrix = np.zeros((len(features_per_layer1), len(features_per_layer2)))
        avg_dist = 0.0
        
        # layers must be flatted
        for row, X in enumerate(features_per_layer1.values()):
            for col, Y in enumerate(features_per_layer2.values()):
                if len(X.shape) != len(Y.shape):
                    # structural difference (max distance)
                    cka_matrix[row, col] = 1
                    continue
                
                X = X.reshape(X.shape[0], -1)
                Y = Y.reshape(Y.shape[0], -1)
                
                # TODO:  interpolation?
                if X.shape != Y.shape:
                    cka_matrix[row, col] = 1
                    continue
                
                #compute the CKA
                cka_dist = CKA.lin_cka_dist(X, Y)
                cka_matrix[row, col] = cka_dist
                
                if row == col:
                    avg_dist += cka_dist
            
        cka_matrix = pd.DataFrame(cka_matrix, 
                                  index=features_per_layer1.keys(), 
                                  columns=features_per_layer2.keys())
        # compute the avg distance of all the cells on the diagonal, 
        # higher is the value different are the information learned by 
        # the two architectures (only if the matrix is square)
        if len(features_per_layer1.values()) == len(features_per_layer2.values()):
            avg_dist = avg_dist / len(features_per_layer1.values())
        else:
            avg_dist = None
                
        self.results = {
            'cka_dist': cka_matrix,
            'avg_cka': avg_dist
        }
        
        return self.results
