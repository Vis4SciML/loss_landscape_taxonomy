from argparse import ArgumentParser
import ast
import os
import sys
from statistics import mean
from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl 

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
PRECISIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
LEARNING_RATES = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]

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
        
        cka = CKA(model, dataloader, layers=RN08_layers, max_batches=args.num_batches)
        cka_list = []
        # compute the average over all the couples
        for bs in BATCH_SIZES:
            for lr in LEARNING_RATES:
                if bs == args.batch_size and lr == args.learning_rate:
                    continue
                target_model, _ = rn08.get_model_and_accuracy(args.saving_folder, 
                                                              bs, 
                                                              lr, 
                                                              args.precision)
                s = cka.compare_output(target_model, 10, 3)
                cka_list.append(s)
                
                # print status
                if len(cka_list) % 10 == 0:
                    print(f"Analysis status:\t{len(cka_list)}/{len(BATCH_SIZES) * len(LEARNING_RATES)}")
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
        fisher = FIT(model, 
                     dataloader, 
                     target_layers=RN08_layers,
                     input_spec=(args.batch_size ,3, 32, 32))
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
    
    