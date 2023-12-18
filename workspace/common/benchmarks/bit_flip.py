import os
import sys
import warnings
import ast
import torch
from statistics import mean
import numpy as np

module_path = os.path.abspath(os.path.join('../../../workspace/models/jets/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from model import JetTagger
from jet_datamodule import JetDataModule




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
                        
        
    def flip_bits(self, percentage=0, number=0):
        '''
        Method used to flip a certain number of bits in the target layers of a neural network.
        '''
        num = int(number + self.bits_number * percentage)
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
                    break
                else:
                    acc += weights.size


# DATA_PATH = '/home/jovyan/checkpoint/'
# DATASET_DIR = '../../../data/JTAG/'
# DATASET_FILE = 'processed_dataset.h5'

# def get_model_index_and_relative_accuracy(batch_size, learning_rate, precision, num_tests=5):
#     '''
#     Return the average EMDs achieved by the model and the index of best experiment
#     '''
#     performances = []
#     max_acc = 0
#     max_acc_index = 0
#     for i in range (1, num_tests+1):
#         file_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/' \
#                     f'JTAG_{precision}b/accuracy_{i}.txt'
#         try:
#             jtag_file = open(file_path)
#             jtag_text = jtag_file.read()
#             accuracy = ast.literal_eval(jtag_text)
#             accuracy = accuracy[0]['test_acc']
#             performances.append(accuracy)
#             if accuracy >= max_acc:
#                 max_acc = accuracy
#                 max_acc_index = i
#             jtag_file.close()
#         except Exception as e:
#             # warnings.warn("Warning: " + file_path + " not found!")
#             continue
        
#     if len(performances) == 0:
#         # warnings.warn(f"Attention: There is no accuracy value for the model: " \
#         #               f"bs{batch_size}_lr{learning_rate}/JTAG_{precision}b")
#         #TODO: I may compute if the model is there
#         return
    
#     return mean(performances), max_acc_index


# def load_model(batch_size, learning_rate, precision):
#     '''
#     Method used to get the model and the relative accuracy
#     '''
#     accuracy, idx = get_model_index_and_relative_accuracy(batch_size, learning_rate, precision)
#     model_path = DATA_PATH + f'bs{batch_size}_lr{learning_rate}/JTAG_{precision}b/net_{idx}_best.pkl'
    
#     # load the model
#     model = JetTagger(
#         quantize=(precision < 32),
#         precision=[
#             precision,
#             precision,
#             precision+3
#         ],
#         learning_rate=learning_rate,
#     )
    
#     # to set the map location
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model(torch.randn((16, 16)))  # Update tensor shapes 
#     model_param = torch.load(model_path, map_location=device)
#     model.load_state_dict(model_param['state_dict'])
    
#     return model, accuracy


# if __name__ == "__main__":
#     # get the datamodule
#     data_module = JetDataModule(
#         data_dir=DATASET_DIR,
#         data_file=os.path.join(DATASET_DIR, DATASET_FILE),
#         batch_size=16,
#         num_workers=4)
    
#     # check if we have processed the data
#     if not os.path.exists(os.path.join(DATASET_DIR, DATASET_FILE)):
#         print('Processing the data...')
#         data_module.process_data(save=True)

#     data_module.setup(0)
#     model, _ = load_model(16, 0.05, 8)
#     model.eval()
    
#     data_loader = data_module.val_dataloader()
#     flat1 = model.model.dense_1.weight_integer.detach().numpy().flatten()
#     print(flat1)
#     # bit flip
#     bit_flip = BitFlip(model, 8, ['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4'])
#     bit_flip.flip_bits(number=10)
#     count = 0
#     for batch, label in data_loader:
#         count += 1
#         outout = model(batch)
        
#         flat2 = model.model.dense_1.weight_integer.detach().numpy().flatten()
#         print(flat2)
#         print(np.setdiff1d(flat2, flat1))
        
#         if count == 3:
#             break

#     model, _ = load_model(16, 0.05, 8)
#     bit_flip = BitFlip(model, 8, ['encoder.enc_dense'])
#     bit_flip.flip_bits(10)
