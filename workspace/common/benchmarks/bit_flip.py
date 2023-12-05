import os
import sys
import warnings
import ast
import torch
from statistics import mean
import numpy as np

module_path = os.path.abspath(os.path.join('../../../workspace/models/econ/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from q_autoencoder import AutoEncoder
from autoencoder_datamodule import AutoEncoderDataModule




class BitFlip:
    '''
    Class used to simulate the bit-flip stress test on
    models quantized with HAWQ.
    '''
    
    def __init__(self, model, precision, layers=[]):
        self.model = model
        self.precision = precision
        self.quant_weights = {}
        self.bits_number = 0
        for name, layer in self.model.named_modules():
            if name in layers:
                self.quant_weights[name] = layer.weight_integer.detach().numpy()
                self.bits_number += self.quant_weights[name].size * self.precision
                        
        
    def flip_bits(self, percentage):
        num = int(self.bits_number * percentage)
        flip_indexes = np.random.randint(0, self.bits_number, size=num)
        for flip_index in flip_indexes:
            weight_index = flip_index // self.precision
            bit_index = flip_index % self.precision
            acc = 0
            for weights in self.quant_weights.values():
                if weight_index < weights.size + acc:
                    # target layer
                    weight_index -= acc # align to the layer
                    flat_weights = weights.reshape(-1)
                    target_weight = flat_weights[weight_index].astype(np.int32)
                    bit_mask = 1 << bit_index
                    target_weight ^= bit_mask
                    flat_weights[weight_index] = target_weight
                else:
                    acc += weights.size


# DATA_PATH = '/data/tbaldi/checkpoint/'
# DATASET_DIR = '../../../data/ECON/Elegun'
# DATASET_FILE = 'nELinks5.npy'

# def get_model_index_and_relative_EMD(batch_size, learning_rate, precision, size, num_tests=3):
#     '''
#     Return the average EMDs achieved by the model and the index of best experiment
#     '''
#     EMDs = []
#     min_emd = 1000
#     min_emd_index = 0
#     for i in range (1, num_tests+1):
#         file_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/' \
#                     f'ECON_{precision}b/{size}/{size}_emd_{i}.txt'
#         try:
#             emd_file = open(file_path)
#             emd_text = emd_file.read()
#             emd = ast.literal_eval(emd_text)
#             emd = emd[0]['AVG_EMD']
#             EMDs.append(emd)
#             if min_emd >= emd:
#                 min_emd = emd
#                 min_emd_index = i
#             emd_file.close()
#         except Exception as e:
#             warnings.warn("Warning: " + file_path + " not found!")
#             continue
        
#     if len(EMDs) == 0:
#         warnings.warn(f"Attention: There is no EMD value for the model: " \
#                       f"bs{batch_size}_lr{learning_rate}/ECON_{precision}b/{size}")
#         #TODO: I may compute if the model is there
#         return
    
#     return mean(EMDs), min_emd_index


# def load_model(batch_size, learning_rate, precision, size):
#     '''
#     Method used to get the model and the relative EMD value
#     '''
#     emd, idx = get_model_index_and_relative_EMD(batch_size, learning_rate, precision, size)
#     model_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/ECON_{precision}b/{size}/net_{idx}_best.pkl'
    
#     # load the model
#     model = AutoEncoder(
#         quantize=(precision < 32),
#         precision=[
#             precision,
#             precision,
#             precision+3
#         ],
#         learning_rate=learning_rate,
#         econ_type=size
#     )
    
#     # to set the map location
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model(torch.randn((1, 1, 8, 8)))  # Update tensor shapes 
#     model_param = torch.load(model_path, map_location=device)
#     model.load_state_dict(model_param['state_dict'])
    
#     return model, emd


# if __name__ == "__main__":
#     # get the datamodule
#     data_module = AutoEncoderDataModule(
#         data_dir=DATASET_DIR,
#         data_file=os.path.join(DATASET_DIR, DATASET_FILE),
#         batch_size=16,
#         num_workers=4)
    
#     # check if we have processed the data
#     if not os.path.exists(os.path.join(DATASET_DIR, DATASET_FILE)):
#         print('Processing the data...')
#         data_module.process_data(save=True)

#     data_module.setup(0)
    
#     model, _ = load_model(16, 0.05, 8, 'baseline')
    
    
    
#     model.eval()
    
#     data_loader = data_module.val_dataloader()
#     flat1 = model.encoder.enc_dense.weight_integer.detach().numpy().flatten()
#     print(flat1)
#     # bit flip
#     bit_flip = BitFlip(model, 8, ['encoder.enc_dense', 'encoder.conv'])
#     bit_flip.flip_bits(0.01)
#     count = 0
#     for batch in data_loader:
#         count += 1
#         outout = model(batch)
        
#         flat2 = model.encoder.enc_dense.weight_integer.detach().numpy().flatten()
#         print(flat2)
#         print(np.setdiff1d(flat2, flat1))
        
#         if count == 3:
#             break
    # noisy_dataset = NoisyDataset(data_module.dataloaders()[1], 10, 'gaussian')
    # dataloader = DataLoader(noisy_dataset, batch_size=16, shuffle=True, num_workers=4)
    # model, _ = load_model(16, 0.05, 8, 'baseline')
    # bit_flip = BitFlip(model, 8, ['encoder.enc_dense'])
    # bit_flip.flip_bits(4)
