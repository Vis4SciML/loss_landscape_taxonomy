from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

from metric import Metric
# ---------------------------------------------------------------------------- #
#                                   Gradient                                   #
# ---------------------------------------------------------------------------- #

class Gradient(Metric):
    def __init__(self, model=None, data_loader=None, name="gradient", loss=nn.CrossEntropyLoss()):
        super().__init__(model, data_loader, name)
        self.loss = loss
        self.results = []
        
    def _get_batch_gradients(self):
        gradient_traces = []
        for _, param in self.model.encoder.named_parameters():
            if param.requires_grad and type(param.grad) == torch.Tensor:
                # print(param.grad.mean().item, type(param.grad))
                gradient_traces.append(param.grad.mean().cpu().item())
        return np.array(gradient_traces).mean()
        
    def compute(self):
        print("Computing the gradients...")
        gradient_traces = []
        for batch, target in self.data_loader:
            outputs = self.model(batch)
            target = target.to(outputs.device)    # both tensors on the same device
            loss = self.loss(outputs, target.float(), self.model.device) 
            loss.backward()
            gradient_traces.append(self._get_batch_gradients())
        self.results.append(gradient_traces)
        
        return self.results
        
        