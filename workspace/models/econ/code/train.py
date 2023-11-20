import os
import time
from tqdm import tqdm
import torch
# import qtorch
import torchinfo
import numpy as np
import multiprocessing
import pytorch_lightning as pl 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from q_autoencoder import AutoEncoder
from autoencoder_datamodule import AutoEncoderDataModule
from utils_pt import unnormalize, emd

# Deprecated, embed in the model
def test_model(model, test_loader, device):
    """
    Our own testing loop instead of using the trainer.test() method so that we
    can multithread EMD computation on the CPU
    """
    model.eval()
    input_calQ_list = []
    output_calQ_list = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            print("batch", x)
            output = model(x.to(device))
            input_calQ = model.map_to_calq(x)
            output_calQ_fr = model.map_to_calq(output)
            input_calQ = torch.stack(
                [input_calQ[i] * model.val_sum[i] for i in range(len(input_calQ))]
            )  # shape = (batch_size, 48)
            output_calQ = unnormalize(
                torch.clone(output_calQ_fr), model.val_sum
            )  # ae_out
            input_calQ_list.append(input_calQ)
            output_calQ_list.append(output_calQ)
    input_calQ = np.concatenate([i_calQ.cpu() for i_calQ in input_calQ_list], axis=0)
    output_calQ = np.concatenate([o_calQ.cpu() for o_calQ in output_calQ_list], axis=0)
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        emd_list = pool.starmap(emd, zip(input_calQ, output_calQ))
    print(f"EMD computation time: {time.time() - start_time} seconds")
    average_emd = np.mean(np.array(emd_list))
    print(f"Average EMD: {average_emd}")
    return average_emd


def main(args):
    # if the directory does not exist you create it
    if not os.path.exists(args.saving_folder):
        os.makedirs(args.saving_folder)
    # ------------------------
    # 0 PREPARE DATA
    # ------------------------
    # instantiate the data module
    data_module = AutoEncoderDataModule.from_argparse_args(args)
    # process the dataset if required
    if not os.path.exists(args.data_file):
        print("Processing data...")
        data_module.process_data()
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print(f'Loading with quantize: {(args.weight_precision < 32)}')
    model = AutoEncoder(
        quantize=(args.weight_precision < 32),
        precision=[
            args.weight_precision, 
            args.bias_precision, 
            args.act_precision
        ],
        learning_rate=args.lr,
        econ_type=args.size,
    )
    torchinfo.summary(model, input_size=(1, 1, 8, 8))  # (B, C, H, W)

    tb_logger = pl_loggers.TensorBoardLogger(args.saving_folder, name=args.size)

    # Stop training when model converges
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="min"
    )

    # Save top-3 checkpoints based on Val/Loss
    top_checkpoint_callback = ModelCheckpoint(
        save_top_k=args.top_models,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=os.path.join(args.saving_folder, args.size),
        filename=f'net_{args.experiment}_best',
        auto_insert_metric_name=False,
    )
    top_checkpoint_callback.FILE_EXTENSION = '.pkl'
    print(f'Saving to dir: {os.path.join(args.saving_folder, args.size)}')
    print(f'Running experiment: {args.experiment}')

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        logger=tb_logger,
        callbacks=[top_checkpoint_callback, early_stop_callback],
        fast_dev_run=args.fast_dev_run,
    )
    print("strategy:", trainer.strategy)
    
    # multiply the batch size by the number of nodes and the number of GPUs
    print(f"Number of nodes: {trainer.num_nodes}")
    print(f"Number of GPUs: {trainer.num_devices}")
    data_module.batch_size = trainer.num_nodes * trainer.num_devices * data_module.batch_size
    print(f"New batch size: {data_module.batch_size}")
    # ------------------------
    # 3 TRAIN MODEL
    # ------------------------
    if not args.no_train:
        trainer.fit(model=model, datamodule=data_module)

    # ------------------------
    # 4 EVALUTE MODEL
    # ------------------------
    # load the model from file
    checkpoint_file = os.path.join(args.saving_folder, args.size, f'net_{args.experiment}_best.pkl')
    print('Loading checkpoint...', checkpoint_file)
    checkpoint = torch.load(checkpoint_file)  
    model.load_state_dict(checkpoint['state_dict'])
    # Need val_sum to compute EMD
    _, val_sum = data_module.get_val_max_and_sum()
    model.set_val_sum(val_sum)
    data_module.setup("test")
    # test_results = test_model(model, data_module.test_dataloader())
    test_results = trainer.test(model, dataloaders=data_module.test_dataloader())
    # save the results on file
    test_results_log = os.path.join(
        args.saving_folder, args.size, args.size + f"_emd_{args.experiment}.txt"
    )
    with open(test_results_log, "w") as f:
        f.write(str(test_results))
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--size", type=str, default="baseline")
    parser.add_argument("--weight_precision", type=int, default=8)
    parser.add_argument("--bias_precision", type=int, default=8)
    parser.add_argument("--act_precision", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.0015625)
    parser.add_argument("--top_models", type=int, default=3)
    parser.add_argument("--experiment", type=int, default=0)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu", "auto"], default="auto"
    )
    # Add dataset-specific args
    parser = AutoEncoderDataModule.add_argparse_args(parser)
    # NOTE: do not activate during real training, just for debugging
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    
    args = parser.parse_args()
    
    print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    main(args)