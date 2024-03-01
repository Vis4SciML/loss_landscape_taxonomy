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
from hessian import Hessian

ECON_layers = ['encoder.conv', 'encoder.enc_dense']


def load_model(path, batch_size, learning_rate, precision, size, index=1):
    '''
    Method used to get the model and the relative EMD value
    '''
    lr = "{:.10f}".format(float(learning_rate)).rstrip('0')
    model_path = path + f'bs{batch_size}_lr{lr}/ECON_{precision}b/{size}/net_{index}_best.pkl'
    
    # load the model
    model = econ.AutoEncoder(
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
    model.to(device)
    model(torch.randn((1, 1, 8, 8)).to(device))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    return model


def main(args):
    lr = "{:.10f}".format(float(args.learning_rate)).rstrip('0')
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
        
    saving_path = os.path.join(
        args.saving_folder, 
        f'bs{args.batch_size}_lr{lr}/ECON_{args.precision}b/{args.size}/'
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
        _, val_sum = data_module.get_val_max_and_sum()
        model.set_val_sum(val_sum)
    
    # ---------------------------------------------------------------------------- #
    #                                     MODEL                                    #
    # ---------------------------------------------------------------------------- #
    model, original_emd = econ.load_model(args.saving_folder, 
                                          args.batch_size, 
                                          args.learning_rate, 
                                          args.precision, 
                                          args.size)
    
    # eval the model
    
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
        # CKA is measured on perturbed training set comprised of mixup samples
        noisy_dataset = NoisyDataset(dataloader, 2, 'gaussian')
        dataloader = DataLoader(noisy_dataset, batch_size=1, shuffle=True)
        
        cka_list = []
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    continue
                model1 = load_model(args.saving_folder, 
                                    args.batch_size, 
                                    args.learning_rate, 
                                    args.precision, 
                                    args.size,
                                    i)
                model2 = load_model(args.saving_folder, 
                                    args.batch_size, 
                                    args.learning_rate, 
                                    args.precision, 
                                    args.size,
                                    j)
                cka = CKA(model1, dataloader, layers=ECON_layers, max_batches=args.num_batches)
                s = cka.compare_output(model2, 10, 3)
                print(s)
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
        data_module.batch_size = 256
        dataloader = data_module.test_dataloader()
        fisher = FIT(model, 
                     dataloader, 
                     target_layers=ECON_layers, 
                     input_spec=(256, 1, 8, 8))
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
    elif args.metric == 'hessian':
        # ---------------------------------------------------------------------------- #
        #                                    Hessian                                   #
        # ---------------------------------------------------------------------------- #
        data_module.batch_size = 256
        dataloader = data_module.test_dataloader()
        hessian = Hessian(model, dataloader, name=f"hessian_{args.trial}")
        hessian.compute()
        hessian.save_on_file(path=saving_path)
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
    parser.add_argument("--trial", type=int, default=0)
    # plot
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--distance", type=int, default=100)
    parser.add_argument("--normalization", type=str, default="filter")
    
    args = parser.parse_args()
    
    # for arg in vars(args):
    #     print(f"{arg}: {getattr(args, arg)}")
    
    main(args)
    
    