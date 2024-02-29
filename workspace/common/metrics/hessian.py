"""The following code is adapted from 
PyHessian: Neural Networks Through the Lens of the Hessian
Z. Yao, A. Gholami, K Keutzer, M. Mahoney
https://github.com/amirgholami/PyHessian
"""

from __future__ import print_function
import os
import sys
import ast
from statistics import mean
import scipy
import torch
import torch.nn as nn

from pyhessian import hessian
from metric import Metric


# ---------------------------------------------------------------------------- #
#                                Hessian metrics                               #
# ---------------------------------------------------------------------------- #

class Hessian(Metric):
    def __init__(self, model=None, data_loader=None, name="hessian"):
        super().__init__(model, data_loader, name)
        self.results = {}   # there will be different values
        
    def compute(self):
        print("Computing the hessian metrics...")
        
        hessian_comp = None

        hessian_comp = hessian(self.model,
                            criterion=self.model.loss,
                            #data=batch,
                            dataloader=self.data_loader,
                            cuda=torch.cuda.is_available())
            
        trace = hessian_comp.trace(maxIter=100, tol=1e-6)
        
        eigenvalues, _ = hessian_comp.eigenvalues(maxIter=100, tol=1e-6, top_n=5)
        
        
        print('trace:', sum(trace))
        print('eigenvalue:', eigenvalues[0])
        #print('eigenvector:', eigenvector)
        
        self.results = {
            "trace": trace,
            "eigenvalues": eigenvalues,
        }
        
        return self.results
    
    
    
# DATA_PATH = '/home/jovyan/checkpoint/'
# DATASET_DIR = '../../../data/RN08'



# if __name__ == "__main__":
#     # get the datamodule
#     model1, _ = rn08.get_model_and_accuracy(DATA_PATH, 16, 0.00625, 8)
#     model2, _ = rn08.get_model_and_accuracy(DATA_PATH, 1024, 0.1, 8)
#     dataloader1 = rn08.get_dataloader(DATASET_DIR, 16)
#     dataloader2 = rn08.get_dataloader(DATASET_DIR, 1024)
#     metric = Hessian(model1, dataloader1)
#     metric.compute()
#     metric = Hessian(model2, dataloader2)
#     metric.compute()
    
