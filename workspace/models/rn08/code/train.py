import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
import rn08
import torchinfo

def main(args):
    # ---------------------------------------------------------------------------- #
    #                           Download and process data                          #
    # ---------------------------------------------------------------------------- #
    train_loader, val_loader, test_loader = rn08.get_cifar10_loaders(args.data_dir, 
                                                                     args.batch_size)
    # ---------------------------------------------------------------------------- #
    #                                Lightning model                               #
    # ---------------------------------------------------------------------------- #
    print(f'Loading with quantize: {(args.weight_precision < 32)}')
    model = rn08.RN08(
        quantize=(args.weight_precision < 32),
        precision=[
            args.weight_precision, 
            args.bias_precision, 
            args.act_precision
        ],
        learning_rate=args.lr
    )
    
    torchinfo.summary(model, input_size=(1, 3, 32, 32))
    
    tb_logger = pl_loggers.TensorBoardLogger(args.saving_folder)
    
    # stop training when model converges
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    
    # save top-3 checkpoints based on Val/Loss
    top_checkpoint_callback = ModelCheckpoint(
        save_top_k=args.top_models,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=args.saving_folder,
        filename=f'net_{args.experiment}_best',
        auto_insert_metric_name=False,
    )
    top_checkpoint_callback.FILE_EXTENSION = '.pkl'
    print(f'Saving to dir: {args.saving_folder}')
    print(f'Running experiment: {args.experiment}')
    
    # ---------------------------------------------------------------------------- #
    #                                    Trainer                                   #
    # ---------------------------------------------------------------------------- #
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices="auto",
        logger=tb_logger,
        callbacks=[top_checkpoint_callback, early_stop_callback],
        fast_dev_run=args.fast_dev_run,
    )

    # multiply the batch size by the number of nodes and the number of GPUs
    print(f"Number of nodes: {trainer.num_nodes}")
    print(f"Number of GPUs: {trainer.num_devices}")
    
    # ---------------------------------------------------------------------------- #
    #                                  Train model                                 #
    # ---------------------------------------------------------------------------- #
    if not args.no_train:
        trainer.fit(model=model, 
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)
        
    # ---------------------------------------------------------------------------- #
    #                                Evaluate model                                #
    # ---------------------------------------------------------------------------- #
    # load the model from file
    checkpoint_file = os.path.join(
            args.saving_folder, 
            f'net_{args.experiment}_best.pkl'
        )
    print('Loading checkpoint...', checkpoint_file)
    
    checkpoint = torch.load(checkpoint_file)  
    model.load_state_dict(checkpoint['state_dict'])
    test_results = trainer.test(model, dataloaders=test_loader)
    
    # save the results on file
    test_results_log = os.path.join(
        args.saving_folder, 
        f"accuracy_{args.experiment}.txt"
    )
    with open(test_results_log, "w") as f:
        f.write(str(test_results))
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saving_folder", type=str)
    parser.add_argument("--data_dir", type=str, default="../../data/RN08")
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--weight_precision", type=int, default=8)
    parser.add_argument("--bias_precision", type=int, default=8)
    parser.add_argument("--act_precision", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.0015625)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--top_models", type=int, default=3)
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, choices=["cpu", "gpu", "tpu", "auto"], 
        default="auto"
    )
    parser.add_argument("--num_workers", type=int, default=8)
    

    # NOTE: do not activate during real training, just for debugging
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    
    args = parser.parse_args()
    
    print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    main(args)