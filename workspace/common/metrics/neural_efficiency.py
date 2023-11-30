from __future__ import print_function

import numpy as np
import warnings
import math
import torch
from utils.feature_extractor import FeatureExtractor
from metric import Metric
from utils import *


# ---------------------------------------------------------------------------- #
#                               Neural efficiency                              #
# ---------------------------------------------------------------------------- #

class NeuralEfficiency(Metric):
    def __init__(self, model=None, data_loader=None, name="neural_efficiency", performance=None):
        super().__init__(model, data_loader, name)
        self.performance = performance  # used to compute the aIQ
        self.results = {}   # there will be different values
        
        
    def entropy(self, probabilities):
        """
        Compute Shannon's entropy for a given probability distribution.

        Parameters:
        - probabilities: List of probabilities for each outcome.

        Returns:
        - entropy: Shannon's entropy.
        """
        return -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    
    def quantize_activation(self, x):
        '''
        We are interested only if the neuron fired (>0) or not (=0)
        '''
        return np.where(x > 0, 1, 0)
    
    
    def get_outputs(self, activation):
        '''
        the convolutional layer will produce more outputs per input (one per channel),
        so we will iterate over each channel and return an array with all the outputs.
        '''
        outputs = []
        if len(activation.shape) == 4:
            # 2dconv
            for channel_idx in range(activation.shape[1]):
                channel_output = activation[:, channel_idx, :, :]
                outputs.append(channel_output)
        elif len(activation.shape) == 2:
            # dense (or similar)
            outputs.append(activation)
        else:
            warnings.warn("Warning: activation dimension not handled yet!")
            
        return outputs
        
    
    def neurons_per_layer(self):
        '''
        Get the number of neurons for each 
        '''
        # Get the number of neurons per layer using named_modules
        nodes_per_layer = {}
        for name, layer in self.model.named_parameters():
            if 'weight' in name:
                nodes_per_layer[name.replace(".weight", "")] = torch.numel(layer)
                
        return nodes_per_layer
    
    
    def entropy_per_layer(self, layers):
        # do not train the network
        self.model.eval()
        # wrap the model with the feature extractor class
        feature_extractor = FeatureExtractor(self.model, layers)
        
        #iterate over the batches to compute the probability of each state
        state_space = {}
        num_batches = 0
        for batch in self.data_loader:
            # need the number of batches as denominator
            num_batches += 1   
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
                    activations = activations[0].detach().numpy()
                else:
                    activations = activations.detach().numpy()
                                        
                # the convolutional layer will produce more outputs per input (one per channel),
                # so I will iterate over each channel
                outputs = self.get_outputs(activations)
                # each output state must be quantized and we record its frequency
                for output_state in outputs:
                    # dictionary to record the probabilities
                    # quantize the activations (convert to bytes to use it as key)
                    quant_activation = self.quantize_activation(output_state).tobytes()
                    # record the probabilities for each layer
                    if name not in state_space:
                        state_space[name] = {}
                        
                    if quant_activation not in state_space[name]:
                        state_space[name][quant_activation] = 1
                    else:
                        state_space[name][quant_activation] += 1
                
        for name, state_freq in state_space.items():
            # compute the probabilities for each output state
            probabilities = []
            for freq in state_freq.values():
                probabilities.append(freq / num_batches)
            
            # compute the entropy of the layer
            layer_entropy = self.entropy(probabilities)
            state_space[name] = layer_entropy
                
        return state_space
        
    def compute(self, beta=2):
        '''
        Compute the neural efficiency metrics.
        '''
        print("Computing the Neural efficiency...")
        
        # get the number of neurons for each layer
        neurons_per_layer = self.neurons_per_layer()
        # compute the entropy for each layer
        entropies = self.entropy_per_layer(neurons_per_layer.keys())
        
        # compute neuron efficiency for each layer
        layers_efficiency = {}
        for name, entropy in entropies.items():
            layers_efficiency[name] = entropy / neurons_per_layer[name]
            
        print('layers neural efficiency:\n', layers_efficiency)
        
        # compute network efficiency, which is the geometric mean of the efficiency 
        # of all the layers
        network_efficiency = 1
        for efficiency in layers_efficiency.values():
            network_efficiency *= efficiency
            
        network_efficiency = network_efficiency ** (1/len(layers_efficiency.items()))
        
        print('network neural efficiency:\n', network_efficiency)
        
        #compute the aIQ, which is a combination between neural network efficiency and model performance
        aIQ = None
        if self.performance is not None:
            aIQ = ((self.performance ** beta) * network_efficiency) ** (1 / (beta + 1))
        else:
            warnings.warn("Warning: you cannot compute the aIQ without the performance of the model (accuracy, EMD, MSE, ...).")
            
        print('aIQ\n', aIQ)
        
        self.results = {
            'layers_efficiency': layers_efficiency,
            'network_efficiency': network_efficiency,
            'aIQ': aIQ
        }
        
        return self.results