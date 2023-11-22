"""The following code is adapted from 
PyHessian: Neural Networks Through the Lens of the Hessian
Z. Yao, A. Gholami, K Keutzer, M. Mahoney
https://github.com/amirgholami/PyHessian
"""

from __future__ import print_function
import torch.nn as nn
from pyhessian import hessian
from metric import Metric


# ---------------------------------------------------------------------------- #
#                                Hessian metrics                               #
# ---------------------------------------------------------------------------- #

class Hessian(Metric):
    def __init__(self, model=None, data_loader=None, name="hessian", loss=nn.CrossEntropyLoss()):
        super().__init__(model, data_loader, name)
        self.loss = loss
        self.results = {}   # there will be different values
        
    def compute(self):
        print("Computing the hessian metrics...")
        
        # compute the hessian component
        hessian_comp = hessian(self.model,
                               criterion=self.loss,
                               dataloader=self.data_loader)
        
        # compute the hessian eigenvalues and the trace
        #top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(maxIter=200, tol=1e-6)
        trace = hessian_comp.trace(maxIter=200, tol=1e-6)
        
        print('trace:', trace)
        
        return self.results