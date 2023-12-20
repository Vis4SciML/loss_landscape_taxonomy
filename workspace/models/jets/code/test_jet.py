from argparse import ArgumentParser
import ast
import os
import sys
from statistics import mean
import warnings
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 

from model import JetTagger
from jet_datamodule import JetDataModule

module_path = os.path.abspath(os.path.join('../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from noisy_dataset import NoisyDataset
from bit_flip import BitFlip

module_path = os.path.abspath(os.path.join('../../common/metrics')) 
sys.path.insert(0, module_path)
from CKA import CKA
from neural_efficiency import NeuralEfficiency
from fisher import FIT


JTAG_layers = ['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4']


def get_model_index_and_relative_accuracy(path, batch_size, learning_rate, precision, num_tests=5):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    performances = []
    max_acc = 0
    max_acc_index = 0
    for i in range (1, num_tests+1):
        file_path = path + f'bs{batch_size}_lr{learning_rate}/' \
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
            warnings.warn("Warning: " + file_path + " not found!")
            continue
        
    if len(performances) == 0:
        warnings.warn(f"Attention: There is no accuracy value for the model: " \
                      f"bs{batch_size}_lr{learning_rate}/JTAG_{precision}b")
        #TODO: I may compute if the model is there
        return
    
    return mean(performances), max_acc_index


def main(args):
    
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
        
    saving_path = os.path.join(
        args.saving_folder, 
        f'bs{args.batch_size}_lr{args.learning_rate}/JTAG_{args.precision}b/'
    )
    # ---------------------------------------------------------------------------- #
    #                                  DATA MODULE                                 #
    # ---------------------------------------------------------------------------- #
    #load the datamodule
    data_module = JetDataModule.from_argparse_args(args)
    # process the dataset if required
    if not os.path.exists(args.data_file):
        print("Processing data...")
        data_module.process_data()
        
    # ---------------------------------------------------------------------------- #
    #                                     MODEL                                    #
    # ---------------------------------------------------------------------------- #
    original_accuracy, idx = get_model_index_and_relative_accuracy(args.saving_folder, 
                                                                   args.batch_size, 
                                                                   args.learning_rate, 
                                                                   args.precision)
    model_path = saving_path + f'net_{idx}_best.pkl'

    model = JetTagger(
        quantize=(args.precision < 32),
        precision=[
            args.precision,
            args.precision,
            args.precision+3
        ],
        learning_rate=args.learning_rate    
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model(torch.randn((16, 16)))  
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    data_module.setup("test")
    dataloader = data_module.test_dataloader()
    
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
        print(f'Original accuracy:\t{original_accuracy}\n' \
              f'Benchmark accuracy:\t{test_results}')
        file_name = f"accuracy_{args.noise_type}_{args.percentage}.txt"
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
        bit_flip = BitFlip(model, args.precision, JTAG_layers)
        bit_flip.flip_bits(number=args.bit_flip) 
        trainer = pl.Trainer(accelerator='auto', devices='auto')
        test_results = trainer.test(model=model, dataloaders=dataloader)
        print(f'Original accuracy:\t{original_accuracy}\n' \
              f'Benchmark accuracy:\t{test_results}')
        file_name = f"accuracy_bitflip_{args.bit_flip}.txt"
        test_results_log = os.path.join(saving_path, file_name)
        print('Result stored in: ' + test_results_log)
        with open(test_results_log, "w") as f:
            f.write(str(test_results))
            f.close()
            
    elif args.metric == 'CKA':
        # ---------------------------------------------------------------------------- #
        #                                      CKA                                     #
        # ---------------------------------------------------------------------------- #
        cka = CKA(model, dataloader, layers=JTAG_layers, max_batches=args.num_batches)
        cka.compute()
        cka.save_on_file(path=saving_path)
        # TODO: compute the distance among models
    elif args.metric == 'neural_efficiency':
        # ---------------------------------------------------------------------------- #
        #                               Neural Efficiency                              #
        # ---------------------------------------------------------------------------- #
        # we have to pass one input per time
        data_module.batch_size = 1
        dataloader = data_module.test_dataloader()
        metric = NeuralEfficiency(model, dataloader, 
                                  performance=original_accuracy, 
                                  max_batches=args.num_batches,
                                  target_layers=JTAG_layers)
        metric.compute(beta=2)
        metric.save_on_file(path=saving_path)
    elif args.metric == 'fisher':
        # ---------------------------------------------------------------------------- #
        #                                    Fisher                                    #
        # ---------------------------------------------------------------------------- #
        fisher = FIT(model, 
                     dataloader, 
                     target_layers=JTAG_layers, 
                     input_spec=(args.batch_size, 16))
        fisher.EF(min_iterations=100, max_iterations=1000)
        fisher.save_on_file(path=saving_path)
    # ADD NEW METRICS HERE
    else:
        print("Metric not supported yet!")
        
    print('Test over!')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--metric", type=str)
    # model
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0015625)
    parser = JetDataModule.add_argparse_args(parser)
    # noise
    parser.add_argument("--percentage", type=int, default=0)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    # bit flip
    parser.add_argument("--bit_flip", type=int, default=0)
    # metrics
    parser.add_argument("--num_batches", type=int, default=None)

    args = parser.parse_args()
    
    main(args)
    
    