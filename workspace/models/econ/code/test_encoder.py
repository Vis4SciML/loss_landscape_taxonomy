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
import q_autoencoder as econ
from autoencoder_datamodule import AutoEncoderDataModule

module_path = os.path.abspath(os.path.join('../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from noisy_dataset import NoisyDataset
from bit_flip import BitFlip

module_path = os.path.abspath(os.path.join('../../common/metrics')) 
sys.path.insert(0, module_path)
from CKA import CKA
from neural_efficiency import NeuralEfficiency
from fisher import FIT
from plot import Plot

ECON_layers = ['encoder.conv', 'encoder.enc_dense']
PRECISIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
LEARNING_RATES = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]


def main(args):
    
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
        
    saving_path = os.path.join(
        args.saving_folder, 
        f'bs{args.batch_size}_lr{args.learning_rate}/ECON_{args.precision}b/{args.size}/'
    )
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
    model, original_emd = econ.load_model(args.saving_folder, 
                                          args.batch_size, 
                                          args.learning_rate, 
                                          args.precision, 
                                          args.size)
    
    # eval the model
    _, val_sum = data_module.get_val_max_and_sum()
    model.set_val_sum(val_sum)
    data_module.setup("test")
    _, dataloader = data_module.dataloaders()
    
    print('-'*80)
    print(f"Metric:\t{args.metric}")
    if args.metric == 'noise':
        # ---------------------------------------------------------------------------- #
        #                                     NOISE                                    #
        # ---------------------------------------------------------------------------- #
        print(f'Noise type:\t{args.noise_type}')
        print(f'Noise percentage:\t{args.percentage}%')
        print('-'*80)
        noisy_dataset = NoisyDataset(dataloader, 
                                     args.percentage, 
                                     args.noise_type)
        dataloader = DataLoader(noisy_dataset, 
                                args.batch_size, 
                                shuffle=False,
                                num_workers=args.num_workers)
        trainer = pl.Trainer(accelerator='auto', devices='auto')
        test_results = trainer.test(model=model, dataloaders=dataloader)
        print(f'Original EMD:\t{original_emd}\n' \
              f'Benchmark EMD:\t{test_results}')
        file_name = f"emd_{args.noise_type}_{args.percentage}.txt"
        test_results_log = os.path.join(saving_path, file_name)
        print('Result stored in: ' + test_results_log)
        with open(test_results_log, "w") as f:
            f.write(str(test_results))
            f.close()       
    elif args.metric == 'bitflip':
        # ---------------------------------------------------------------------------- #
        #                                   BIT FLIP                                   #
        # ---------------------------------------------------------------------------- #
        print(f'Flipped bits: {args.bit_flip}')
        print('-'*80)
        bit_flip = BitFlip(model, args.precision, ECON_layers)
        bit_flip.flip_bits(number=args.bit_flip) 
        trainer = pl.Trainer(accelerator='auto', devices='auto')
        test_results = trainer.test(model=model, dataloaders=dataloader)
        print(f'Original EMD:\t{original_emd}\n' \
              f'Benchmark EMD:\t{test_results}')
        file_name = f"emd_bitflip_{args.bit_flip}.txt"
        test_results_log = os.path.join(saving_path, file_name)
        print('Result stored in: ' + test_results_log)
        with open(test_results_log, "w") as f:
            f.write(str(test_results))
            f.close()
    elif args.metric == 'CKA':
        # ---------------------------------------------------------------------------- #
        #                                      CKA                                     #
        # ---------------------------------------------------------------------------- #
        cka = CKA(model, dataloader, layers=ECON_layers, max_batches=args.num_batches)
        cka_list = []
        for p in PRECISIONS:
            for bs in BATCH_SIZES:
                for lr in LEARNING_RATES:
                    if bs != args.batch_size and lr != args.learning_rate and p != args.precision:
                        target_model, _ = econ.load_model(args.saving_folder, 
                                                          bs, 
                                                          lr, 
                                                          p, 
                                                          args.size)
                        s = cka.compare_output(target_model, 10)
                        cka_list.append(s)
        cka.results['CKA_similarity'] = mean(cka_list)
        cka.save_on_file(path=saving_path)
        print(mean(cka_list))
    elif args.metric == 'neural_efficiency':
        # ---------------------------------------------------------------------------- #
        #                               Neural Efficiency                              #
        # ---------------------------------------------------------------------------- #
        # we have to pass one input per time
        data_module.batch_size = 1
        dataloader = data_module.test_dataloader()
        metric = NeuralEfficiency(model, dataloader, 
                                  performance=original_emd, 
                                  max_batches=args.num_batches,
                                  target_layers=ECON_layers)
        metric.compute(beta=0.5)
        metric.save_on_file(path=saving_path)
    elif args.metric == 'fisher':
        # ---------------------------------------------------------------------------- #
        #                                    Fisher                                    #
        # ---------------------------------------------------------------------------- #
        fisher = FIT(model, 
                     dataloader, 
                     target_layers=ECON_layers, 
                     input_spec=(args.batch_size, 1, 8, 8))
        fisher.EF(min_iterations=100, max_iterations=1000)
        fisher.save_on_file(path=saving_path)
        
    elif args.metric == 'plot':
        # ---------------------------------------------------------------------------- #
        #                                     Plot                                     #
        # ---------------------------------------------------------------------------- #
        plot = Plot(model, dataloader)
        plot.compute(steps=args.steps, 
                     distance=args.distance, 
                     normalization=args.normalization)
        plot.save_on_file(path=saving_path)
    # ADD NEW METRICS HERE
    else:
        print("Metric not supported yet!")
        
    print('Test over!')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--metric", type=str)
    # model
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0015625)
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    # noise
    parser.add_argument("--percentage", type=int, default=0)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    # bit flip
    parser.add_argument("--bit_flip", type=int, default=0)
    # metrics
    parser.add_argument("--num_batches", type=int, default=None)
    # plot
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--distance", type=int, default=100)
    parser.add_argument("--normalization", type=str, default="filter")
    
    args = parser.parse_args()
    
    main(args)
    
    