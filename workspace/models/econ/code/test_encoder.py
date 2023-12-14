from argparse import ArgumentParser
import ast
import os
import sys
from statistics import mean
import warnings
from utils_pt import unnormalize, emd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
import numpy as np
import multiprocessing
from q_autoencoder import AutoEncoder
from autoencoder_datamodule import AutoEncoderDataModule

module_path = os.path.abspath(os.path.join('../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from noisy_dataset import NoisyDataset
from bit_flip import BitFlip

module_path = os.path.abspath(os.path.join('../../common/metrics')) 
sys.path.insert(0, module_path)
from CKA import CKA
from neural_efficiency import NeuralEfficiency

def get_model_index_and_relative_EMD(path, batch_size, learning_rate, precision, size, num_tests=3):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    EMDs = []
    min_emd = 1000
    min_emd_index = 0
    for i in range (1, num_tests+1):
        file_path = path + f'bs{batch_size}_lr{learning_rate}/' \
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


def main(args):
    
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
        
    # ---------------------------------------------------------------------------- #
    #                                  DATA MODULE                                 #
    # ---------------------------------------------------------------------------- #
    #load the datamodule
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    # process the dataset if required
    if not os.path.exists(args.data_file):
        print("Processing data...")
        data_module.process_data()
        
    
    # ---------------------------------------------------------------------------- #
    #                                     MODEL                                    #
    # ---------------------------------------------------------------------------- #
    original_emd, idx = get_model_index_and_relative_EMD(args.saving_folder, 
                                                         args.batch_size, 
                                                         args.learning_rate, 
                                                         args.precision, 
                                                         args.size)
    model_path = args.saving_folder + f'bs{args.batch_size}_lr{args.learning_rate}' \
                f'/ECON_{args.precision}b/{args.size}/net_{idx}_best.pkl'

    model = AutoEncoder(
        quantize=(args.precision < 32),
        precision=[
            args.precision,
            args.precision,
            args.precision+3
        ],
        learning_rate=args.learning_rate,
        econ_type=args.size
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model(torch.randn((1, 1, 8, 8)))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    # eval the model
    _, val_sum = data_module.get_val_max_and_sum()
    model.set_val_sum(val_sum)
    data_module.setup("test")
    _, dataloader = data_module.dataloaders()
    
    # # ---------------------------------------------------------------------------- #
    # #                                   FLIP BIT                                   #
    # # ---------------------------------------------------------------------------- #
    # if args.bit_flip > 0:
    #     print('-'*80)
    #     print(f'Noise type: {args.noise_type}')
    #     print(f'Flipped bits: {args.bit_flip}')
    #     bit_flip = BitFlip(model, 
    #                        args.precision, 
    #                        ['encoder.conv', 'encoder.enc_dense'])
    #     bit_flip.flip_bits(number=args.bit_flip)    # we are using the same model, so I flip just one time per iteration
            
    
    # # ---------------------------------------------------------------------------- #
    # #                                   ADD NOISE                                  #
    # # ---------------------------------------------------------------------------- #
    
    # if args.percentage > 0:
    #     print('-'*80)
    #     print(f'Radiation test')
    #     # prepare noisy dataloader
    #     print(f'Noise percentage: {args.percentage}%')
    #     noisy_dataset = NoisyDataset(dataloader, 
    #                                  args.percentage, 
    #                                  args.noise_type)
    #     dataloader = DataLoader(noisy_dataset, 
    #                             args.batch_size, 
    #                             shuffle=False,
    #                             num_workers=args.num_workers)
        
    # # ---------------------------------------------------------------------------- #
    # #                                   BENCHMARK                                  #
    # # ---------------------------------------------------------------------------- #
    # trainer = pl.Trainer(
    #     accelerator='auto',
    #     devices=-1,
    # )
        
    # test_results = trainer.test(model=model, dataloaders=dataloader)
    # print(f'Original EMD:\t{original_emd}\nBenchmark EMD:\t{test_results}')
    
    saving_path = os.path.join(args.saving_folder, f'bs{args.batch_size}_lr{args.learning_rate}' \
                f'/ECON_{args.precision}b/{args.size}/')
    
    cka = CKA(model, 
              dataloader, 
              layers=['encoder.conv', 'encoder.enc_dense'],
              max_batches=50000)
    cka.compute()
    cka.save_on_file(path=saving_path)
    
    
    data_module.batch_size = 1
    
    metric = NeuralEfficiency(model, 
                              data_module.test_dataloader(), 
                              performance=original_emd, 
                              max_batches=50000,
                              target_layers=['encoder.conv', 'encoder.enc_dense'])
    metric.compute(beta=0.5)
    metric.save_on_file(path=saving_path)
    
    # # save the results on file
    # file_name = args.size + f"_emd"
    # if args.percentage > 0:
    #     file_name += f"_{args.noise_type}_{args.percentage}"
    # elif args.bit_flip > 0:
    #     file_name += f"_bitflip_{args.bit_flip}"
    # file_name += ".txt"
    
    # test_results_log = os.path.join(
    #     args.saving_folder, f'bs{args.batch_size}_lr{args.learning_rate}/ECON_{args.precision}b/{args.size}', file_name
    # )
    
    # print('Result stored in: ' + test_results_log)
    # with open(test_results_log, "w") as f:
    #     f.write(str(test_results))
    #     f.close()
    
    print('Test over!')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--percentage", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0015625)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--bit_flip", type=int, default=0)

    parser = AutoEncoderDataModule.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    main(args)
    
    