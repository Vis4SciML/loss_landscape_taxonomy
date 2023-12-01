from __future__ import print_function

import warnings
import torch
from metric import Metric
from utils.feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------- #
#                                 Fisher metric                                #
# ---------------------------------------------------------------------------- #

class Fisher(Metric):
    
    def __init__(self, 
                 model=None, 
                 data_loader=None, 
                 name="fisher_trace", 
                 optimizer=None, 
                 target_layers=[]):
        
        super().__init__(model, data_loader, name)
        self.results = {}   # there will be different values
        self.optimizer = optimizer
        self.target_layers = target_layers
        # model in evaluation mode
        self.model.eval()
        
        
    def compute(self):
        print("Computing the Fisher estimated trace...")
        
        # dictionary to store the trace of the targets layer
        ef_trace = {}
        for name, _ in self.model.named_parameters():
            # select only the weights
            if 'weight' in name:
                name = name.replace(".weight", "")
                # select only the target layers
                if name in self.target_layers:
                    ef_trace[name] = 0.0
                    
        # get the target layers
        for batch, target in self.data_loader:
            # forward pass
            output = self.model.forward(batch)
            # compute the loss
            loss = self.model.loss(output, target)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # update model parameters
            self.optimizer.step()
            
            for name, param in self.model.named_parameters():
                name = name.replace(".weight", "")
                if name in ef_trace.keys():
                    ef_trace[name] += torch.square(torch.linalg.vector_norm(param.grad))
                    
        
        for name, value in ef_trace.items():
            ef_trace[name] = value.item() / len(self.data_loader)
            
        self.results = {
            "EF_trace": ef_trace
        }
        
        return self.results

            