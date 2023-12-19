from __future__ import print_function

import os
import sys
import ast
from statistics import mean
import warnings
import torch
import torch.nn as nn
import numpy as np
from metric import Metric
from utils.feature_extractor import FeatureExtractor

module_path = os.path.abspath(os.path.join('../../../workspace/models/jets/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from model import JetTagger
from jet_datamodule import JetDataModule

# ---------------------------------------------------------------------------- #
#                                 Fisher metric                                #
# ---------------------------------------------------------------------------- #
class FIT(Metric):
    def __init__(self, model, target_layer, input_spec = (16,16), layer_filter=None):
        ''' Class for computing FIT
        Args:
            model
            device
            input_spec
            layer_filter - str - layers to ignore 
        '''
        
        # select the device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = 'cuda'
            
            
        names, param_nums, params = self.layer_accumulator(model, target_layer, layer_filter)
        param_sizes = [p.size() for p in params]
        self.hook_layers(model, layer_filter)
        _ = model(torch.randn(input_spec)[None, ...].to(self.device))
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
        
    
    def layer_accumulator(self, model, target_layer, layer_filter=None):
        ''' Accumulates the required parameter information,
        Args:
            model
            layer_filter
        Returns:
            names - accumulated layer names
            param_nums - accumulated parameter numbers
            params - accumulated parameter values
        '''
        
        def layer_filter(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True
            
            
        layers = []
        names = []
        param_nums = []
        params = []
        for name, module in model.named_modules():
            if name in target_layer:
                print(name, module)
                for n, p in list(module.named_parameters()):
                    if n.endswith('weight'):
                        names.append(name)
                        p.collect = True
                        layers.append(module)
                        param_nums.append(p.numel())
                        params.append(p)
                    else:
                        p.collect = False
                continue
            for p in list(module.parameters()):
                if p.requires_grad:
                    p.collect = False
        print(len(layers))
        print(np.sum(param_nums))
        for i, (n, p) in enumerate(zip(names, param_nums)):
            print(i, n, p)

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
            if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))) and (layer_filt(name)):

                module.register_forward_hook(hook_inp)
                module.act_quant = True
            else:
                module.act_quant = False
        

DATA_PATH = '/home/jovyan/checkpoint/'
DATASET_DIR = '../../../data/JTAG/'
DATASET_FILE = 'processed_dataset.h5'

def get_model_index_and_relative_accuracy(batch_size, learning_rate, precision, num_tests=5):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    performances = []
    max_acc = 0
    max_acc_index = 0
    for i in range (1, num_tests+1):
        file_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/' \
                    f'JTAG_{precision}b/accuracy_{i}.txt'
        try:
            jtag_file = open(file_path)
            jtag_text = jtag_file.read()
            accuracy = ast.literal_eval(jtag_text)
            accuracy = accuracy[0]['test_acc']
            performances.append(accuracy)
            if accuracy >= max_acc:
                max_acc = accuracy
                max_acc_index = i
            jtag_file.close()
        except Exception as e:
            # warnings.warn("Warning: " + file_path + " not found!")
            continue
        
    if len(performances) == 0:
        # warnings.warn(f"Attention: There is no accuracy value for the model: " \
        #               f"bs{batch_size}_lr{learning_rate}/JTAG_{precision}b")
        #TODO: I may compute if the model is there
        return
    
    return mean(performances), max_acc_index


def load_model(batch_size, learning_rate, precision):
    '''
    Method used to get the model and the relative accuracy
    '''
    accuracy, idx = get_model_index_and_relative_accuracy(batch_size, learning_rate, precision)
    model_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/JTAG_{precision}b/net_{idx}_best.pkl'
    
    # load the model
    model = JetTagger(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model(torch.randn((16, 16)))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    return model, accuracy


if __name__ == "__main__":
    # get the datamodule
    data_module = JetDataModule(
        data_dir=DATASET_DIR,
        data_file=os.path.join(DATASET_DIR, DATASET_FILE),
        batch_size=16,
        num_workers=4)
    
    # check if we have processed the data
    if not os.path.exists(os.path.join(DATASET_DIR, DATASET_FILE)):
        data_module.process_data(save=True)

    data_module.setup(0)
    model, _ = load_model(16, 0.05, 8)
    model.eval()
    
    data_loader = data_module.val_dataloader()
    fit_computer = FIT(model, ['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4'], input_spec=(16, 16))