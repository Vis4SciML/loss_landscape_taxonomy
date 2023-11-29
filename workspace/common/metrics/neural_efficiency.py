from __future__ import print_function

import numpy as np
import warnings
from utils.feature_extractor import FeatureExtractor
from metric import Metric
from utils import *


##################################################################

# def get_attr(module, name):
#     names = name.split('.')
#     for name in names:
#         module = getattr(module, name)
#     return module 


# ##################################################################
# def hook(model, input, output):
#     if len(output) == 2:
#         model._activations = torch.cat((model._activations, output[0].detach().float()), 1)
#     else:
#         model._activations = torch.cat((model._activations, output.detach().float()), 1)


# def register_hooks(model):
#     for name, layer in model.named_modules():
#         layer.__name__ = name
#         layer._activations = torch.tensor([])
#         layer.register_forward_hook(hook)


# def cleanup(model):
#     for _name, layer in model.named_modules():
#         layer._activations = torch.tensor([])


# ##################################################################
# def compute_neural_efficiency(model, data_loader):
#     batch_eff = []
#     register_hooks(model)

#     for idx, (inputs, _) in enumerate(data_loader):
#         if idx+1 == len(data_loader):
#             break
#         _ = model(inputs)
#         activations = get_activations(model, layer_types=('Linear', 'Conv2d','QuantLinear', 'QuantConv2d'))
#         b_eff = layer_entropy(model, activations, len(inputs))
#         batch_eff.append(b_eff)

#     neural_eff = np.mean(np.array(batch_eff))
#     print(f"[Model] Neural Efficiency: {neural_eff}")

#     return neural_eff


# def num_nodes(module):
#     class_name = module.__class__.__name__
#     if 'Linear' in class_name and hasattr(module, 'in_features'):
#         return module.out_features
#     elif 'Conv' in class_name and hasattr(module, 'out_channels'):
#         return np.prod(module._activations.shape[1:])
#     elif 'BatchNorm' in class_name and hasattr(module, 'num_features'):
#         return module.num_features


# def layer_entropy(model, activations, width):
#     entropy = []
#     for layer_name, acts in activations.items():
#         if len(acts.shape) == 1:
#             continue
#         layer = get_attr(model, layer_name)
#         # layer = getattr(model, layer_name)
#         # layer neural_eff = entropy/num_nodes
#         # print(f'Layer {layer_name} non-zero count: {np.count_nonzero(acts.numpy(), axis=0)}')
#         entropy.append(
#             ((acts > 0).type(torch.int8).sum(axis=0) / width).sum() / num_nodes(layer)
#             # (np.count_nonzero(acts.numpy(), axis=0) / width).sum() / num_nodes(layer)
#         )
#     cleanup(model)
#     return np.sqrt(np.sqrt(np.prod(entropy)))


# ##################################################################
# param_layers = ['Linear', 'Bilinear', 'Conv1d', 'Conv2d', 'Conv3d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'QuantLinear', 'QuantConv2d','QuantBnConv2d']
# act_layers = ['ReLU', 'LeakyReLU', 'ReLU6', 'PReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'Tanh', 'QuantAct', 'QuantMaxPool2d', 'QuantAveragePool2d']
# layers = param_layers + act_layers


# def get_activations(model, layer_types=(), layer_names=['encoder.conv', 'encoder.enc_dense']):
#     if not layer_types:
#         layer_types = layers
#     layer_a = {}
#     if layer_names:
#         for layer_name in layer_names:
#             module = get_attr(model, layer_name)
#             layer_a[module.__name__] = module._activations
#         return layer_a
#     else:
#         for _, module in model.named_modules():
#             if module.__class__.__name__ in layer_types:
#                 layer_a[module.__name__] = module._activations
#         return layer_a


# ##################################################################
# parser = get_parser(code_type='neural_eff')
# args = parser.parse_args()

# model_arch = args.arch.split('_')[0]
# print('Importing code for', model_arch)
# if model_arch == 'JT':
#     sys.path.append(os.path.join(sys.path[0], "../jets/code")) 
# elif model_arch == 'ECON':
#     sys.path.append(os.path.join(sys.path[0], "../econ/code")) 

# from model import load_checkpoint 
# from data import get_loader

# ##################################################################
# print(os.getcwd())
# train_loader, test_loader = get_loader(args)

# if args.train_or_test == 'train':
#     eval_loader = train_loader
# elif args.train_or_test == 'test':
#     eval_loader = test_loader

# eval_loader = test_loader

# model_eff = []
# for exp_id1 in range(3):
#     file_name1 = get_filename(args, exp_id1)
#     model = load_checkpoint(args, file_name1)    
#     print(f'Computing Neural Efficiency for {file_name1}')
#     model_eff.append(compute_neural_efficiency(model, eval_loader))

# f = open(args.result_location, "wb")
# pickle.dump(model_eff, f)
# f.close()


# ---------------------------------------------------------------------------- #
#                               Neural efficiency                              #
# ---------------------------------------------------------------------------- #

class NeuralEfficiency(Metric):
    def __init__(self, model=None, data_loader=None, name="neural_efficiency", layers=[]):
        super().__init__(model, data_loader, name)
        self.layers = layers
        self.results = {}   # there will be different values
        
    def entropy(self, probability):
        pass
    
    def quantize_activation(self, x):
        '''
        We are interested only if the neuron fired (>0) or not (=0)
        '''
        return np.where(x > 0, 1, 0)
    
    def entropy_per_layer(self):
        # do not train the network
        self.model.eval()
        # wrap the model with the feature extractor class
        feature_extractor = FeatureExtractor(self.model, self.layers)
        
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
                    warnings.warn(f"Attention: the layer {name} has None features!")
                    continue
                elif isinstance(activations, tuple):
                    # special case where I just consider the first tensor of the tuple
                    warnings.warn(f"Attention: the layer {name} has tuple as features!")
                    activations = activations[0].detach().numpy()
                else:
                    activations = activations.detach().numpy()
                
                # dictionary to record the probabilities
                # quantize the activations (convert to bytes to use it as key)
                quant_activation = self.quantize_activation(activations).tobytes()
                # record the probabilities for each layer
                if name not in state_space:
                    state_space[name] = {}
                    
                if quant_activation not in state_space[name]:
                    state_space[name][quant_activation] = 1
                else:
                    state_space[name][quant_activation] += 1
        
        for name, state_freq in state_space.items():
            probabilities = []
            for freq in state_freq.values():
                probabilities.append(freq / num_batches)
                
            print(probabilities)
                
                
            
        
    def compute(self):
        '''
        Compute the neural efficiency of the layers of the model.
        '''
        print("Computing the Neural efficiency...")
        self.entropy_per_layer()
