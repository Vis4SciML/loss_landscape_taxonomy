
"""The following code is adapted from 
Similarity of Neural Network Representations Revisited
Simon Kornblith, Mohammad Norouzi, Honglak Lee and Geoffrey Hinton
https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
"""
from __future__ import print_function

import os 
import sys
from metric import Metric

import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

from workspace.common.metrics.utils.CKA_utils import *

# Get data
train_loader, test_loader = get_loader(args)

# init the structure to store the performances
representation_similarity = {}
classification_similarity = {}
cos = nn.CosineSimilarity(dim=0)

# we do it 3 times
for exp_id1 in range(3):
    
    representation_similarity[exp_id1] = {}
    classification_similarity[exp_id1] = {}
    
    for exp_id2 in range(3):
        # get the file name of two checkpoints 
        file_name1, file_name2 = return_file_name(args, exp_id1, exp_id2)
        
        # load the models
        model1 = load_checkpoint(args, file_name1)
        model2 = load_checkpoint(args, file_name2)
        
        # pick the dataset on which evaluate the CKA metric
        if args.train_or_test == "train":
            eval_loader = train_loader
        elif args.train_or_test == "test":
            eval_loader = test_loader
        else:
            raise ValueError('Invalid input.')
        
        if not args.compare_classification:
            # do not compare the classification
            
            cka_from_features_average = []
            
            # as many times as designed for CKA
            for CKA_repeat_runs in range(args.CKA_repeat_runs):

                cka_from_features = []

                latent_all_1, latent_all_2 = all_latent(model1, model2, eval_loader, num_batches=args.CKA_batches, args=args)

                for name in latent_all_1.keys():

                    print(name)

                    if args.flattenHW:
                        cka_from_features.append(feature_space_linear_cka(latent_all_1[name], latent_all_2[name]))
                    else:
                        cka_from_features.append(cka_compute(gram_linear(latent_all_1[name]), gram_linear(latent_all_2[name])))
                        
                cka_from_features_average.append(cka_from_features)
                
            # compute the average of n CKA computations
            cka_from_features_average = np.mean(np.array(cka_from_features_average), axis=0)
            
            print('cka_from_features shape')
            print(cka_from_features_average.shape)

            representation_similarity[exp_id1][exp_id2] = cka_from_features_average
        
        else:
            # compare the classification
            classification_similarity[exp_id1][exp_id2] = compare_classification(model1, model2, eval_loader, args=args, cos=cos)
        
        temp_results = {'representation_similarity': representation_similarity, 'classification_similarity': classification_similarity}
        
        # save the results on file
        f = open(args.result_location, "wb")
        pickle.dump(temp_results, f)
        f.close()
        
        
class CKA(Metric):
    def __init__(self, model=None, data_loader=None, name="CKA_similarity", loss=None):
        super().__init__(model, data_loader, name)
        self.loss = loss
        self.results = {}   # there will be different values
    
    """Compute Gram (kernel) matrix for a linear kernel.
    Args:
        x: A num_examples x num_features matrix of features.
    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    def _gram_linear(self, x):
        return x.dot(x.T)

    """Compute Gram (kernel) matrix for an RBF kernel.
    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)
    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    def _gram_rbf(self, x, threshold=1.0):
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        
        return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

    """Center a symmetric Gram matrix.
    This is equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.
    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.
    Returns:
        A symmetric matrix with centered columns and rows.
    """
    def _center_gram(self, gram, unbiased=False):
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram
    
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    def _debiased_dot_product_similarity_helper(xty, 
                                                sum_squared_rows_x, 
                                                sum_squared_rows_y, 
                                                squared_norm_x, 
                                                squared_norm_y,
    n):
  
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

        
    """
    Compute CKA.
    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    def cka(self, gram_X, gram_Y, debiased=True):
        gram_x = center_gram(gram_x, unbiased=debiased)
        gram_y = center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)
        
    def feature_space_linear_cka(self, gram_X, gram_Y, debiased=True):
        
        
    def compute(self):
        print("Computing the CKA similarity...")
        
        
        return self.results
