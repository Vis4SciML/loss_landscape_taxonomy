
"""The following code is adapted from:
DO WIDE AND DEEP NETWORKS LEARN THE SAME
THINGS? UNCOVERING HOW NEURAL NETWORK
REPRESENTATIONS VARY WITH WIDTH AND DEPTH
Thao Nguyen, AI Resident, Google Research
https://blog.research.google/2021/05/do-wide-and-deep-networks-learn-same.html
"""
from __future__ import print_function
import os
import sys
import ast
import torch
import warnings
from statistics import mean
from metric import Metric
from utils.feature_extractor import FeatureExtractor
import time

import pandas as pd

# test
module_path = os.path.abspath(os.path.join('../../../workspace/models/econ/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from autoencoder_datamodule import AutoEncoderDataModule
from q_autoencoder import AutoEncoder


# ---------------------------------------------------------------------------- #
#                                CKA similarity                                #
# ---------------------------------------------------------------------------- #
        
class CKA(Metric):
    
    def __init__(self, model, data_loader, name="CKA_similarity", layers=[], max_batches=100):
        super().__init__(model, data_loader, name)
        self.layers = layers
        self.max_batches = max_batches
        self.results = {}   # there will be different values
        self.device = 'cpu'
        if torch.cuda.is_available():
            print('CUDA available!')
            self.model.cuda()
            self.device = 'cuda'

    
    @staticmethod
    def gram_matrix(X):
        '''
        Generate Gram matrix and preprocess to compute unbiased HSIC.

        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.

        Args:
        x: A [num_examples, num_features] matrix.

        Returns:
        A [num_examples ** 2] vector.
        '''
        X = X.reshape(X.shape[0], -1)
        gram_X = X @ X.T
        n = gram_X.shape[0]
        gram_X.fill_diagonal_(0)
        means = torch.sum(gram_X, axis=0) / (n - 2)
        means -= torch.sum(means) / 2 * (n - 1)
        gram_X -= means[:, None]
        gram_X -= means[None, :]
        gram_X.fill_diagonal_(0)
        return gram_X.reshape((-1,))
    
    
    @staticmethod
    def update_state(hsic_accumulator, activations):
        layers_gram = []
        for x in activations.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram.append(CKA.gram_matrix(x))
        layers_gram = torch.stack(layers_gram, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram, layers_gram.T)
    
    
    @staticmethod
    def update_state_across_models(hsic_accumulator,
                                   hsic_accumulator1, 
                                   activations1, 
                                   hsic_accumulator2, 
                                   activations2):
        # dimension test
        torch.testing.assert_close(hsic_accumulator1.shape[0], len(activations1))
        torch.testing.assert_close(hsic_accumulator2.shape[0], len(activations2))
        device = hsic_accumulator.device
        # activation 1
        layers_gram1 = []
        for x in activations1.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram1.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram1.append(CKA.gram_matrix(x))
        layers_gram1 = torch.stack(layers_gram1, axis=0)
        # activation 2
        layers_gram2 = []
        for x in activations2.values():
            if x is None:
                continue
            elif isinstance(x, tuple):
                layers_gram2.append(CKA.gram_matrix(x[0]))    # HAWQ nesting problem
            else:
                layers_gram2.append(CKA.gram_matrix(x))
        layers_gram2 = torch.stack(layers_gram2, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram1, layers_gram2.T), \
                hsic_accumulator1 + torch.einsum('ij,ij->i', layers_gram1, layers_gram1), \
                hsic_accumulator2 + torch.einsum('ij,ij->i', layers_gram2, layers_gram2)
    
    
    def compare(self, model, layers=None):
        '''
        Compare the CKA similarity between the layers of two models
        '''
        # second model can have different layers
        layers1 = self.layers
        layers2 = self.layers
        if layers is not None:
            layers2 = layers
        num_layers1 = len(layers1)
        num_layers2 = len(layers2)
        
        hsic_accumulator = torch.zeros((num_layers1, num_layers2), device=self.device, dtype=torch.float32)
        hsic_accumulator1 = torch.zeros((num_layers1,), device=self.device, dtype=torch.float32)
        hsic_accumulator2 = torch.zeros((num_layers2,), device=self.device, dtype=torch.float32)
        
        self.model.eval()
        model.eval()
        
        model1 = FeatureExtractor(self.model, layers1)
        model2 = FeatureExtractor(model, layers2)
        
        model2.to(self.device)
        
        count = 0
        for batch in self.data_loader:
            count += 1
            
            activations1 = model1.forward(batch)
            activations2 = model2.forward(batch)
            hsic_accumulator, hsic_accumulator1, hsic_accumulator2 = \
                                            CKA.update_state_across_models(hsic_accumulator,
                                                                           hsic_accumulator1,
                                                                           activations1,
                                                                           hsic_accumulator2,
                                                                           activations2)
            if count == self.max_batches:
                break
            
        mean_hsic = hsic_accumulator
        normalization1 = torch.sqrt(hsic_accumulator1)
        normalization2 = torch.sqrt(hsic_accumulator2)
        mean_hsic /= normalization1[:, None]
        mean_hsic /= normalization2[None, :]
        
        # cka_matrix = pd.DataFrame(mean_hsic, 
        #                           index=layers1, 
        #                           columns=layers2)
        
        
        self.results['cka_dist'] = 1 - torch.mean(torch.diagonal(mean_hsic)).item()
        self.results['compared_cka'] = mean_hsic
        return self.results
    
    
    def compute(self):
        '''
        Compare the CKA similarity among the layers of a model.
        '''
        num_layers = len(self.layers)
        hsic_accumulator = torch.zeros((num_layers, num_layers), device=self.device, dtype=torch.float32)
        # bind a hook to the outputs of the models' layers
        model = FeatureExtractor(self.model, self.layers)
        model.eval()

        # iterate over the dataset
        count = 0
        for batch in self.data_loader:
            count += 1
            # compare the layers one against the other
            activations = model.forward(batch)
            hsic_accumulator = CKA.update_state(hsic_accumulator, activations)
            if count == self.max_batches:
                break
                
        mean_hsic = hsic_accumulator
        normalization = torch.sqrt(torch.diagonal(hsic_accumulator))
        mean_hsic = mean_hsic / normalization[:, None]
        mean_hsic = mean_hsic / normalization[None, :]
            
        # cka_matrix = pd.DataFrame(mean_hsic, 
        #                           index=self.layers, 
        #                           columns=self.layers)
        
        self.results['internal_cka'] = mean_hsic
        return self.results
            

# test 
DATA_PATH = '/home/jovyan/checkpoint/'

def get_model_index_and_relative_EMD(batch_size, learning_rate, precision, size, num_tests=3):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    EMDs = []
    min_emd = 1000
    min_emd_index = 0
    for i in range (1, num_tests+1):
        file_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/' \
                    f'ECON_{precision}b/{size}/{size}_emd_{i}.txt'
        try:
            emd_file = open(file_path)
            emd_text = emd_file.read()
            emd = ast.literal_eval(emd_text)
            emd = emd[0]['AVG_EMD']
            EMDs.append(emd)
            if min_emd >= emd:
                min_emd = emd
                min_emd_index = i
            emd_file.close()
        except Exception as e:
            warnings.warn("Warning: " + file_path + " not found!")
            continue
        
    if len(EMDs) == 0:
        warnings.warn(f"Attention: There is no EMD value for the model: " \
                      f"bs{batch_size}_lr{learning_rate}/ECON_{precision}b/{size}")
        #TODO: I may compute if the model is there
        return
    
    return mean(EMDs), min_emd_index


def load_model(batch_size, learning_rate, precision, size):
    '''
    Method used to get the model and the relative EMD value
    '''
    emd, idx = get_model_index_and_relative_EMD(batch_size, learning_rate, precision, size)
    model_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/ECON_{precision}b/{size}/net_{idx}_best.pkl'
    
    # load the model
    model = AutoEncoder(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
        econ_type=size
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model(torch.randn((1, 1, 8, 8)))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    return model, emd


if __name__ == "__main__":
    DATASET_DIR = '../../../data/ECON/Elegun'
    DATASET_FILE = 'nELinks5.npy'
    # get the datamodule
    data_module = AutoEncoderDataModule(
        data_dir=DATASET_DIR,
        data_file=os.path.join(DATASET_DIR, DATASET_FILE),
        batch_size=1024,
        num_workers=4)
    
    # check if we have processed the data
    if not os.path.exists(os.path.join(DATASET_DIR, DATASET_FILE)):
        print('Processing the data...')
        data_module.process_data(save=True)

    data_module.setup(0)
    
    model, _ = load_model(16, 0.0015625, 2, 'baseline')
    model2, _ = load_model(256, 0.00625, 10, 'baseline')
    cka = CKA(model, 
              data_module.test_dataloader(), 
              layers=['encoder.conv', 'encoder.enc_dense'],
              max_batches=10000)
    start = time.perf_counter()
    result = cka.compare(model2)
    end = time.perf_counter()
    print('time:', end - start)
    print(result)
    
    