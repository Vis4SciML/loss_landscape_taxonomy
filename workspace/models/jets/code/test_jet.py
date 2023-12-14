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
    model_path = args.saving_folder + f'bs{args.batch_size}_lr{args.learning_rate}' \
                f'/JTAG_{args.precision}b/net_{idx}_best.pkl'

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
    
    print(model_path)
    model(torch.randn((16, 16)))  # ASK Javi
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    data_module.setup("test")
    dataloader = data_module.test_dataloader()
    
    # # ---------------------------------------------------------------------------- #
    # #                                   FLIP BIT                                   #
    # # ---------------------------------------------------------------------------- #
    # if args.bit_flip > 0:
    #     print('-'*80)
    #     print(f'Noise type: {args.noise_type}')
    #     print(f'Flipped bits: {args.bit_flip}')
    #     bit_flip = BitFlip(model, 
    #                        args.precision, 
    #                        ['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4'])
    #     bit_flip.flip_bits(number=args.bit_flip)    # we are using the same model, so I flip just one time per iteration
            
    
    # # ---------------------------------------------------------------------------- #
    # #                                   ADD NOISE                                  #
    # # ---------------------------------------------------------------------------- #
    
    # if args.percentage > 0:
    #     print('-'*80)
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
    # print('-'*80)
    
    # trainer = pl.Trainer(
    #     accelerator='auto',
    #     devices=-1,
    # )
        
    # test_results = trainer.test(model=model, dataloaders=dataloader)
    # print(f'Original accuracy:\t{original_accuracy}\n' \
    #       f'Benchmark accuracy:\t{test_results}')
    
    saving_path = os.path.join(args.saving_folder, f'bs{args.batch_size}_lr{args.learning_rate}' \
                f'/JTAG_{args.precision}b/')
    
    cka = CKA(model, 
              dataloader, 
              layers=['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4'],
              max_batches=5)
    cka.compute()
    cka.save_on_file(path=saving_path)
    
    
    data_module.batch_size = 1
    
    metric = NeuralEfficiency(model, 
                              data_module.test_dataloader(), 
                              performance=original_accuracy, 
                              max_batches=5,
                              target_layers=['model.dense_1', 'model.dense_2', 'model.dense_3', 'model.dense_4'])
    metric.compute(beta=2)
    metric.save_on_file(path=saving_path)
    
    # # save the results on file
    # file_name = "accuracy"
    # if args.percentage > 0:
    #     file_name += f"_{args.noise_type}_{args.percentage}"
    # elif args.bit_flip > 0:
    #     file_name += f"_bitflip_{args.bit_flip}"
    # file_name += ".txt"
    
    # test_results_log = os.path.join(
    #     args.saving_folder, f'bs{args.batch_size}_lr{args.learning_rate}/JTAG_{args.precision}b', file_name
    # )
    
    # print('Result stored in: ' + test_results_log)
    # with open(test_results_log, "w") as f:
    #     f.write(str(test_results))
    #     f.close()
    
    print('Test over!')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--percentage", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0015625)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--bit_flip", type=int, default=0)

    parser = JetDataModule.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    main(args)
    
    