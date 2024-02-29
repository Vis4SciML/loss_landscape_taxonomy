from argparse import ArgumentParser
import ast
import os
import sys
import gc
from statistics import mean
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl 
import warnings
import torch

import rn08

module_path = os.path.abspath(os.path.join('../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from noisy_dataset import NoisyDataset

module_path = os.path.abspath(os.path.join('../../common/metrics')) 
sys.path.insert(0, module_path)
from CKA import CKA
from neural_efficiency import NeuralEfficiency
from fisher import FIT
from plot import Plot
from hessian import Hessian


RN08_layers = layers = [
        'model.conv1', 
        'model.QBlocks.0.conv1', 
        'model.QBlocks.0.conv2', 
        'model.QBlocks.1.conv1', 
        'model.QBlocks.1.conv2',  
        'model.QBlocks.2.conv1', 
        'model.QBlocks.2.conv2',
        'model.QBlocks.2.shortcut',
        'model.QBlocks.3.conv1', 
        'model.QBlocks.3.conv2', 
        'model.QBlocks.4.conv1', 
        'model.QBlocks.4.conv2',
        'model.QBlocks.4.shortcut',
        'model.QBlocks.5.conv1', 
        'model.QBlocks.5.conv2', 
        'model.linear'
    ]


def load_model(path, batch_size, learning_rate, precision, index):
    '''
    Return the model and its accuracy
    '''
    folder_path = os.path.join(
        path,
        f'bs{batch_size}_lr{learning_rate}/RN08_{precision}b/'
    )
    
    # get the model
    model_file = os.path.join(folder_path, f"net_{index}_best.pkl")
    model = rn08.RN08(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model(torch.randn((1,3,32,32)).to(device))  # Update tensor shapes 
    try:
        model_param = torch.load(model_file, map_location=device)
        model.load_state_dict(model_param['state_dict'])
    except:
        print(f"File not found! ({model_file})")
        return None
    
    return model


def main(args):
    
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
        
    saving_path = os.path.join(
        args.saving_folder, 
        f'bs{args.batch_size}_lr{args.learning_rate}/RN08_{args.precision}b/'
    )
    # ---------------------------------------------------------------------------- #
    #                                  DATA MODULE                                 #
    # ---------------------------------------------------------------------------- #
    _, _, dataloader = rn08.get_cifar10_loaders(args.data_dir, args.batch_size)
        
    # ---------------------------------------------------------------------------- #
    #                                     MODEL                                    #
    # ---------------------------------------------------------------------------- #
    model, accuracy = rn08.get_model_and_accuracy(args.saving_folder, 
                                                  args.batch_size, 
                                                  args.learning_rate, 
                                                  args.precision)
    
    print('-'*80)
    print(f"Metric:\t{args.metric}")
    if args.metric == 'noise':
        # ---------------------------------------------------------------------------- #
        #                                     NOISE                                    #
        # ---------------------------------------------------------------------------- #
        print(f'Noise type:\t{args.noise_type}')
        print('-'*80)
        corruptions = [f'{args.noise_type}']
        x_test, y_test = load_cifar10c(n_examples=args.num_batches, corruptions=corruptions, severity=5)
        noisy_acc = clean_accuracy(model, x_test, y_test)
        print(f'Standard accuracy: {accuracy}\nCIFAR-10-C accuracy: {noisy_acc:.1%}')
        
        file_name = f"accuracy_{args.noise_type}.txt"
        test_results_log = os.path.join(saving_path, file_name)
        print('Result stored in: ' + test_results_log)
        with open(test_results_log, "w") as f:
            f.write(str(noisy_acc))
            f.close()
        
    elif args.metric == 'CKA':
        # ---------------------------------------------------------------------------- #
        #                                      CKA                                     #
        # ---------------------------------------------------------------------------- #
        # CKA is measured on perturbed training set comprised of mixup samples
        noisy_dataset = NoisyDataset(dataloader, 2, 'gaussian')
        dataloader = DataLoader(noisy_dataset, batch_size=1, shuffle=True)
        
        cka_list = []
        # compute the average over all the couples
        for i in range(1, 4):
            for j in range(1, 4):
                if i == j:
                    continue
                model1 = load_model(args.saving_folder, 
                                    args.batch_size, 
                                    args.learning_rate, 
                                    args.precision, 
                                    i)
                model2 = load_model(args.saving_folder, 
                                    args.batch_size, 
                                    args.learning_rate, 
                                    args.precision, 
                                    j)
                cka = CKA(model1, dataloader, layers=RN08_layers, max_batches=args.num_batches)
                s = cka.compare_output(model2, 10, 3)
                print(s)
                cka_list.append(s)
        # store the result
        cka.results['CKA_similarity'] = mean(cka_list)
        cka.save_on_file(path=saving_path)
        print(mean(cka_list))
    elif args.metric == 'neural_efficiency':
        # ---------------------------------------------------------------------------- #
        #                               Neural Efficiency                              #
        # ---------------------------------------------------------------------------- #
        # we have to pass one input per time
        _, _, dataloader = rn08.get_cifar10_loaders(args.data_dir, 1)
        metric = NeuralEfficiency(model, 
                                  dataloader, 
                                  performance=accuracy, 
                                  max_batches=args.num_batches,
                                  target_layers=RN08_layers)
        metric.compute(beta=2)
        metric.save_on_file(path=saving_path)
    elif args.metric == 'fisher':
        # ---------------------------------------------------------------------------- #
        #                                    Fisher                                    #
        # ---------------------------------------------------------------------------- #
        _, _, dataloader = rn08.get_cifar10_loaders(args.data_dir, 256)
        fisher = FIT(model, 
                     dataloader, 
                     target_layers=RN08_layers,
                     input_spec=(256 ,3, 32, 32))
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
        _, _, dataloader = rn08.get_cifar10_loaders(args.data_dir, 256)
        hessian = Hessian(model, dataloader)
        hessian.compute()
        hessian.save_on_file(path=saving_path)
        
    # ADD NEW METRICS HERE
    else:
        print("Metric not supported yet!")
        
    print('Test over!')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--data_dir", type=str, default="../../data/RN08")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metric", type=str)
    # model
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0015625)
    parser.add_argument("--batch_size", type=int, default=1024)
    # noise
    parser.add_argument("--noise_type", type=str, default="pixelate")
    # metrics
    parser.add_argument("--num_batches", type=int, default=1000)
    # plot
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--distance", type=int, default=100)
    parser.add_argument("--normalization", type=str, default="filter")

    args = parser.parse_args()
    
    main(args)
    
    