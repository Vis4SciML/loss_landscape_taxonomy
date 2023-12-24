import argparse
import torch
import os
import sys


# import modules from JTAG model
module_path = os.path.abspath(os.path.join('../../workspace/models/jets/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import model
from jet_datamodule import JetDataModule

# import modules from ECON model
module_path = os.path.abspath(os.path.join('../../workspace/models/econ/code/')) # or the path to your source code
sys.path.insert(0, module_path)
import q_autoencoder
from autoencoder_datamodule import AutoEncoderDataModule

DATA_PATH = '/home/jovyan/checkpoint/'

if __name__ == '__main__':
    
    # ---------------------------------------------------------------------------- #
    #                                   Arguments                                  #
    # ---------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--size', default='baseline', help='size of the model')
    parser.add_argument('--model', default='ECON', help='model to be tested')
    parser.add_argument('--batch_size', default=512, help='batch size of the model')
    parser.add_argument('--learning_rate', default=0.05, help='learning rate of the model')
    parser.add_argument('--precision', default=32, type=int, help='parameter precision')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    args = parser.parse_args()
    # ---------------------------------------------------------------------------- #
    #                              Plotting resolution                             #
    # ---------------------------------------------------------------------------- #
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
    
    # ---------------------------------------------------------------------------- #
    #                                     Model                                    #
    # ---------------------------------------------------------------------------- #
    net = None
    if args.model == 'ECON':
        net, _ = q_autoencoder.load_model(DATA_PATH, 
                                          args.batch_size, 
                                          args.learning_rate, 
                                          args.precision, 
                                          args.size)
    elif args.model == 'JTAG':
        net, _ = model.load_model(DATA_PATH, 
                                  args.batch_size, 
                                  args.learning_rate, 
                                  args.precision)
    else:
        print(f"{args.model} not implemented yet!")
        exit
        
    # TODO: check net_plotter.py