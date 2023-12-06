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


def test_model(model, test_loader, max_batches):
    """
    Our own testing loop instead of using the trainer.test() method so that we
    can multithread EMD computation on the CPU
    """
    model.eval()
    input_calQ_list = []
    output_calQ_list = []
    with torch.no_grad():
        count_batches = 0
        max_batches = min(max_batches, len(test_loader))
        for batch, original_batch in test_loader:
            count_batches += 1
            # move to the right device
            batch = batch.to(model.device)
            original_batch = original_batch.to(model.device)
            # compute the encoded outputs
            output = model(batch)
            input_calQ = model.map_to_calq(original_batch)
            output_calQ_fr = model.map_to_calq(output)
            input_calQ = torch.stack(
                [input_calQ[i] * model.val_sum[i] for i in range(len(input_calQ))]
            )  # shape = (batch_size, 48)
            output_calQ = unnormalize(
                torch.clone(output_calQ_fr), model.val_sum
            )  # ae_out
            input_calQ_list.append(input_calQ)
            output_calQ_list.append(output_calQ)
            # terminate the test
            print(f"Benchmark: {count_batches}/{max_batches}")
            if count_batches == max_batches:
                break
            
    input_calQ = np.concatenate([i_calQ.cpu() for i_calQ in input_calQ_list], axis=0)
    output_calQ = np.concatenate([o_calQ.cpu() for o_calQ in output_calQ_list], axis=0)

    with multiprocessing.Pool() as pool:
        emd_list = pool.starmap(emd, zip(input_calQ, output_calQ))

    average_emd = np.mean(np.array(emd_list))
    return average_emd



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
                                                args.lr, 
                                                args.precision, 
                                                args.size)
    model_path = args.saving_folder + f'bs{args.batch_size}_lr{args.lr}' \
                f'/ECON_{args.precision}b/{args.size}/net_{idx}_best.pkl'

    model = AutoEncoder(
        quantize=(args.precision < 32),
        precision=[
            args.precision,
            args.precision,
            args.precision+3
        ],
        learning_rate=args.lr,
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
    
    # ---------------------------------------------------------------------------- #
    #                                   ADD NOISE                                  #
    # ---------------------------------------------------------------------------- #
    if args.percentage > 0:
        print('-'*80)
        print(f'Noise type: {args.noise_type} - Percentage: {args.percentage}%')
        # prepare noisy dataloader
        noisy_dataset = NoisyDataset(dataloader, 
                                     args.percentage, 
                                     args.noise_type)
        dataloader = DataLoader(noisy_dataset, 
                                args.batch_size, 
                                shuffle=False,
                                num_workers=4)
    
    # ---------------------------------------------------------------------------- #
    #                                   FLIP BIT                                   #
    # ---------------------------------------------------------------------------- #
    if args.bit_flip > 0:
        print('-'*80)
        print(f'Bit flipped: {args.bit_flip}%')
        bit_flip = BitFlip(model, args.precision, ['encoder.conv', 'encoder.enc_dense'])
        bit_flip.flip_bits(args.bit_flip)
        
    # ---------------------------------------------------------------------------- #
    #                                   BENCHMARK                                  #
    # ---------------------------------------------------------------------------- #
    # prepare the trainer
    print('-'*80)
    test_results = test_model(model, dataloader, args.num_batches)
    print(f'Original EMD:\t{original_emd}\nBenchmark EMD:\t{test_results}')
    
    # save the results on file
    file_name = args.size + f"_emd"
    if args.percentage > 0:
        file_name += f"_{args.noise_type}_{args.percentage}"
    elif args.bit_flip > 0:
        file_name += f"_bitflip_{args.bit_flip}"
    file_name += ".txt"
    
    test_results_log = os.path.join(
        args.saving_folder, f'bs{args.batch_size}_lr{args.lr}/ECON_{args.precision}b/{args.size}', file_name
    )
    
    print('Result stored in: ' + test_results_log)
    with open(test_results_log, "w") as f:
        f.write(str(test_results))
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--percentage", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0015625)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--bit_flip", type=int, default=0)
    
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    main(args)
    
    