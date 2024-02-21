from __future__ import print_function

import numpy as np
import warnings
import math
import torch
from utils.feature_extractor import FeatureExtractor
from metric import Metric
from scipy.stats import entropy


# ---------------------------------------------------------------------------- #
#                               Neural efficiency                              #
# ---------------------------------------------------------------------------- #

class NeuralEfficiency(Metric):
    def __init__(self, 
                 model=None, 
                 data_loader=None, 
                 name="neural_efficiency", 
                 target_layers=[],
                 performance=None, 
                 max_batches=None):
        super().__init__(model, data_loader, name)
        self.performance = performance  # used to compute the aIQ
        self.results = {}   # there will be different values
        self.max_batches = max_batches
        self.target_layers = target_layers
        if max_batches is None:
            self.max_batches = len(self.data_loader)
    
    
    def quantize_activation(self, x):
        '''
        We are interested only if the neuron fired (>0) or not (=0)
        '''
        return torch.where(x > 0, 1, 0)
    
    
    def get_outputs(self, activation):
        '''
        the convolutional layer will produce more outputs per input (one per channel),
        so we will iterate over each channel and return an array with all the outputs.
        '''
        assert activation.shape[0] == 1, \
            f'The batch size should be 1 instead of {activation.shape[0]}!'
            
        outputs = []
        # removing the batch size
        activation = torch.squeeze(activation, dim=0)
        if len(activation.shape) == 3:
            # 2dconv
            for channel_idx in range(activation.shape[0]):
                channel_output = activation[channel_idx, :, :]
                outputs.append(channel_output)
        elif len(activation.shape) == 1:
            # dense (or similar)
            outputs.append(activation)
        else:
            warnings.warn("Warning: activation dimension not handled yet!")
            
        return outputs
        
    
    def entropy_per_layer(self, layers):
        # do not train the network
        self.model.eval()
        # wrap the model with the feature extractor class
        feature_extractor = FeatureExtractor(self.model, layers)
        
        #iterate over the batches to compute the probability of each state
        state_space = {}
        counter = 0
        for batch in self.data_loader:
            counter += 1
            
            # print status 
            if counter % 10 == 0:
                print(f"Analysis status: {counter}/{min(len(self.data_loader), self.max_batches)}")
                
            # check if there are also the label
            if isinstance(batch, list):
                batch = batch[0]
                
            # get the activations of each layer
            features_per_layer = feature_extractor.forward(batch)
            for name, activations in features_per_layer.items():
                # handle possible problems with the tensors
                if activations is None:
                    warnings.warn(f"Attention: the layer " + name + " has None features!")
                    continue
                elif isinstance(activations, tuple):
                    # special case where I just consider the first tensor of the tuple (HAWQ problem)
                    warnings.warn(f"Attention: the layer " + name + " has tuple as features!")
                    activations = activations[0]
                else:
                    activations = activations
                                        
                # the convolutional layer will produce more outputs per input (one per channel),
                # so I will iterate over each channel
                outputs = self.get_outputs(activations)
                # each output state must be quantized and we record its frequency
                for output_state in outputs:
                    # dictionary to record the probabilities
                    # quantize the activations (convert to bytes to use it as key)
                    quant_activation = self.quantize_activation(output_state).detach().cpu().numpy().tobytes()
                    # record the probabilities for each layer
                    if name not in state_space:
                        state_space[name] = {'num_neurons': torch.numel(output_state)}
                        
                    if quant_activation not in state_space[name]:
                        state_space[name][quant_activation] = 1
                    else:
                        state_space[name][quant_activation] += 1
            if counter >= self.max_batches:
                break
                
        for name, state_freq in state_space.items():
            # compute the probabilities for each output state
            probabilities = []
            num_outputs = sum(state_freq.values())
            for freq in state_freq.values():
                probabilities.append(freq / num_outputs)
            
            # compute the entropy of the layer
            probabilities = np.array(probabilities)
            layer_entropy = entropy(probabilities, base=2)
            state_space[name]['entropy'] = layer_entropy
                
        return state_space
        
    def compute(self, beta=2):
        '''
        Compute the neural efficiency metrics.
        '''
        # compute the entropy for each layer
        entropies = self.entropy_per_layer(self.target_layers)
        
        # compute neuron efficiency for each layer
        layers_efficiency = {}
        for name, layer in entropies.items():
            layers_efficiency[name] = layer['entropy'] / layer['num_neurons']
            
        # print('layers neural efficiency:\n', layers_efficiency)
        
        # compute network efficiency, which is the geometric mean of the efficiency 
        # of all the layers
        network_efficiency = 1
        for efficiency in layers_efficiency.values():
            network_efficiency *= efficiency
            
        network_efficiency = network_efficiency ** (1/len(layers_efficiency.items()))
        
        # print('network neural efficiency:\n', network_efficiency)
        
        #compute the aIQ, which is a combination between neural network efficiency and model performance
        aIQ = None
        if self.performance is not None:
            aIQ = ((self.performance ** beta) * network_efficiency) ** (1 / (beta + 1))
        else:
            warnings.warn("Warning: you cannot compute the aIQ without the performance of the model (accuracy, EMD, MSE, ...).")
            
        # print('aIQ\n', aIQ)
        
        self.results = {
            'layers_efficiency': layers_efficiency,
            'network_efficiency': network_efficiency,
            'aIQ': aIQ
        }
        
        return self.results
    
    

# test
import os
import sys
module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import rn08

DATA_PATH = "/home/jovyan/checkpoint/"
DATASET_PATH = "../../../data/RN08"
    
if __name__ == "__main__":
    model, acc = rn08.get_model_and_accuracy(DATA_PATH, 64, 0.0015625, 11)
    dataloader = rn08.get_dataloader(DATASET_PATH, 1)
    layers = [
        'model.conv1', 
        'model.QBlocks.0.conv1', 
        'model.QBlocks.0.conv2', 
        'model.QBlocks.1.conv1', 
        'model.QBlocks.1.conv2',  
        'model.QBlocks.2.conv1', 
        'model.QBlocks.2.conv2',
        'model.QBlocks.2.shortcut',
        'model.QBlocks.3.conv1', 
        'model.QBlocks.3.conv2', 
        'model.QBlocks.4.conv1', 
        'model.QBlocks.4.conv2',
        'model.QBlocks.4.shortcut',
        'model.QBlocks.5.conv1', 
        'model.QBlocks.5.conv2', 
        'model.linear'
    ]
    
    ne = NeuralEfficiency(model, dataloader, target_layers=layers, max_batches=10000, performance=acc)
    print(ne.compute())