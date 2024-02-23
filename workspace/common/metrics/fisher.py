from __future__ import print_function

import os
import sys
import ast
from statistics import mean
import warnings
import torch
import torch.nn as nn
import numpy as np
import scipy
from metric import Metric
from utils.feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------- #
#                                 Fisher metric                                #
# ---------------------------------------------------------------------------- #
class FIT(Metric):
    def __init__(self, model, data_loader, name="fisher", target_layers=[], input_spec = (16,16), layer_filter=None):
        ''' Class for computing FIT
        Args:
            model
            device
            input_spec
            layer_filter - str - layers to ignore 
        '''
        super().__init__(model, data_loader, name)
        # select the device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'
            
        self.target_layers = target_layers
        self.results = {}
        names, param_nums, params = self.layer_accumulator(model, layer_filter)
        param_sizes = [p.size() for p in params]
        self.hook_layers(model, layer_filter)
        _ = model(torch.randn(input_spec).to(self.device))
        act_sizes = []
        act_nums = []
        for name, module in model.named_modules():    
            if module.act_quant:
                act_sizes.append(module.act_in[0].size())
                act_nums.append(np.prod(np.array(module.act_in[0].size())[1:]))
                
        self.names = names
        self.param_nums = param_nums
        self.params = params
        self.param_sizes = param_sizes
        self.act_sizes = act_sizes
        self.act_nums = act_nums
        
    
    def layer_accumulator(self, model, layer_filter=None):
        ''' Accumulates the required parameter information,
        Args:
            model
            layer_filter
        Returns:
            names - accumulated layer names
            param_nums - accumulated parameter numbers
            params - accumulated parameter values
        '''
        names = []
        param_nums = []
        params = []
        
        for name, module in model.named_modules():
            if name in self.target_layers:
                for n, p in module.named_parameters():
                    if n.endswith('weight'):
                        
                        if name in names:
                            continue
                        names.append(name)
                        param_nums.append(p.numel())
                        params.append(p)
                        p.collect = True  # Assuming you want to set collect flag for gradient computation
                    else:
                        p.collect = False
            else:
                for p in module.parameters():
                    p.collect = False


        for i, (n, p) in enumerate(zip(names, params)):
            p.collect = True


        return names, np.array(param_nums), params
    
        
    def hook_layers(self, model, layer_filter=None):
        ''' Hooks the required activation information, which can be collected on network pass
        Args:
            model
            layer_filter
        '''

        def hook_inp(m, inp, out):
            m.act_in = inp

        def layer_filt(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True

        for name, module in model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(hook_inp)
                module.act_quant = True
            else:
                module.act_quant = False
        
    
    def EF(self, 
           tol=1e-3, 
           min_iterations=100, 
           max_iterations=100):
        ''' Computes the EF
        Args:
            tol - tolerance used for early stopping
            min_iterations - minimum number of iteration to include
            max_iterations - maximum number of iterations after which to break
        Returns:
            vFv_param_c - EF for the parameters
            vFv_act_c - EF for the activations
            F_param_acc - EF estimator accumulation for the parameters
            F_act_acc - EF estimator accumulation for the activations
            ranges_param_acc - parameter range accumulation
            ranges_act_acc - activation range accumulation
        '''
        
        self.model.eval()
        F_act_acc = []
        F_param_acc = []
        param_estimator_accumulation = []
        act_estimator_accumulation = []

        F_flag = False
        NaN_flag = False

        total_batches = 0.
        
        TFv_act = [torch.zeros(ps).to(self.device) for ps in self.act_sizes[1:]]  # accumulate result
        TFv_param = [torch.zeros(ps).to(self.device) for ps in self.param_sizes]  # accumulate result

        ranges_param_acc = []
        ranges_act_acc = []
        
        while(total_batches < max_iterations and not F_flag):
            
            for batch, label in self.data_loader:
                self.model.zero_grad()
                
                inputs, labels = batch.to(self.device), label.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                loss = self.model.loss(outputs, labels)
                
                ranges_act = []
                actsH = []
                for name, module in self.model.named_modules():
                    if module.act_quant:
                        actsH.append(module.act_in[0])
                        ranges_act.append((torch.max(module.act_in[0]) - torch.min(module.act_in[0])).detach().cpu().numpy())
                        
                ranges_param = []
                paramsH = []
                for paramH in self.model.parameters():
                    if not paramH.collect:
                        continue
                    paramsH.append(paramH)
                    ranges_param.append((torch.max(paramH.data) - torch.min(paramH.data)).detach().cpu().numpy())
                    
                G = torch.autograd.grad(loss, [*paramsH, *actsH[1:]])
                
                                
                G2 = []
                for g in G:
                    G2.append(batch_size*g*g)
                                                        
                indiv_param = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[:len(TFv_param)]])
                indiv_act = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[len(TFv_param):]])
                param_estimator_accumulation.append(indiv_param)
                act_estimator_accumulation.append(indiv_act)
                
                TFv_param = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_param, G2[:len(TFv_param)])]
                ranges_param_acc.append(ranges_param)
                TFv_act = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_act, G2[len(TFv_param):])]
                ranges_act_acc.append(ranges_act)
                                
                total_batches += 1
                
                TFv_act_normed = [TFv_ / float(total_batches) for TFv_ in TFv_act]
                vFv_act = [torch.sum(x) for x in TFv_act_normed]
                vFv_act_c = np.array([i.detach().cpu().numpy() for i in vFv_act])

                TFv_param_normed = [TFv_ / float(total_batches) for TFv_ in TFv_param]
                vFv_param = [torch.sum(x) for x in TFv_param_normed]
                vFv_param_c = np.array([i.detach().cpu().numpy() for i in vFv_param])
                
                if total_batches >= 2:

                    param_var = np.var((param_estimator_accumulation - vFv_param_c)/vFv_param_c)/total_batches
                    act_var= np.var((act_estimator_accumulation - vFv_act_c)/vFv_act_c)/total_batches
                    
                    # print(f'Iteration {total_batches}, Estimator variance: W:{param_var} / A:{act_var}')
                
                    if act_var < tol and param_var < tol and total_batches > min_iterations:
                        F_flag = True
                
                if F_flag or total_batches >= max_iterations:
                    break
        
        self.EFw = vFv_param_c
        self.EFa = vFv_act_c
        # TODO: check possible usage
        self.FAw = F_param_acc
        self.FAa = F_act_acc
        self.Rw = ranges_param_acc
        self.Ra = ranges_act_acc
        
        self.results = {
            "EF_trace_w": self.EFw,
            "EF_trace_a": self.EFa,
            "avg_EF": scipy.stats.mstats.gmean(self.EFw),
            #"error_nan": NaN_flag
        }
        
        return self.results
    
    # compute FIT:
    def noise_model(self, ranges, precision):
        ''' Uniform noise model
        Args:
            ranges - data ranges
            config - bit configuration
        Returns:
            noise power
        '''
        return (ranges/(2**precision - 1))**2


    def FIT(self, w_precision, act_precision):
        ''' computes FIT 
        Args:
            config - bit configuration for weights and activations interlaced: [w1,a1,w2,a2,...]
        Returns:
            FIT value
        '''
        pert_acts = self.noise_model(np.mean(self.Ra, axis=0)[1:], act_precision)
        pert_params = self.noise_model(np.mean(self.Rw, axis=0), w_precision)

        f_acts_T = pert_acts * self.EFa
        f_params_T = pert_params * self.EFw
        pert_T = np.sum(f_acts_T) + np.sum(f_params_T)
        return pert_T

DATA_PATH = '/home/jovyan/checkpoint/'
DATASET_DIR = '../../../data/RN08'

# test
# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../../workspace/models/rn08/code/')) # or the path to your source code
# sys.path.insert(0, module_path)
# import rn08
# DATA_PATH = "/home/jovyan/checkpoint/"
# DATASET_PATH = "../../../data/RN08"

# if __name__ == "__main__":
#     # get the datamodule
#     model, acc = rn08.get_model_and_accuracy(DATA_PATH, 1024, 0.0015625, 8)
#     dataloader = rn08.get_dataloader(DATASET_PATH, 1024)
#     layers = [
#         'model.conv1', 
#         'model.QBlocks.0.conv1', 
#         'model.QBlocks.0.conv2', 
#         'model.QBlocks.1.conv1', 
#         'model.QBlocks.1.conv2',  
#         'model.QBlocks.2.conv1', 
#         'model.QBlocks.2.conv2',
#         'model.QBlocks.2.shortcut',
#         'model.QBlocks.3.conv1', 
#         'model.QBlocks.3.conv2', 
#         'model.QBlocks.4.conv1', 
#         'model.QBlocks.4.conv2',
#         'model.QBlocks.4.shortcut',
#         'model.QBlocks.5.conv1', 
#         'model.QBlocks.5.conv2', 
#         'model.linear'
#     ]
#     fit_computer = FIT(model, dataloader, 
#                        target_layers=layers, 
#                        input_spec=(1024 ,3, 32, 32))
    
#     result = fit_computer.EF(tol=1e-2, 
#                              min_iterations=20,
#                              max_iterations=1000)
    
#     print(result)