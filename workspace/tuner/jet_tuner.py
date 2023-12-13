import re 
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from hawq.utils import QuantAct, QuantLinear
import pytorch_lightning as pl 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
module_path = os.path.abspath(os.path.join('../models/jets/code/')) # or the path to your source code
sys.path.insert(0, module_path)
from jet_datamodule import JetDataModule



####################################################
# Base Model for Quantization 
####################################################
class QModel(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision

    def init_dense(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantLinear(
            weight_bit=self.weight_precision, bias_bit=self.bias_precision
        )
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)


####################################################
# MLP Jet-Tagger
####################################################
class ThreeLayer(nn.Module):
    def __init__(self):
        super(ThreeLayer, self).__init__()

        # half of the weights
        self.dense_1 = nn.Linear(16, 5)
        # self.dense_2 = nn.Linear(8, 8)
        # self.dense_3 = nn.Linear(16, 16)
        #self.dense_4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act(self.dense_1(x))
        # x = self.act(self.dense_2(x))
        # x = self.act(self.dense_3(x))
        # return self.softmax(self.dense_4(x))
        return self.softmax(x)


####################################################
# MLP Jet-Tagginer
####################################################
class QThreeLayer(QModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer, self).__init__(weight_precision, bias_precision)

        self.quant_input = QuantAct(act_precision)
        
        self.init_dense(model, "dense_1")
        self.quant_act_1 = QuantAct(act_precision)

        # self.init_dense(model, "dense_2")
        # self.quant_act_2 = QuantAct(act_precision)

        # self.init_dense(model, "dense_3")
        # self.quant_act_3 = QuantAct(act_precision)

        # self.init_dense(model, "dense_4")

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        
        x = self.act(self.dense_1(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)
        
        # x = self.act(self.dense_2(x, act_scaling_factor))
        # x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)
        
        # x = self.act(self.dense_3(x, act_scaling_factor))
        # x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)
        
        # x = self.dense_4(x, act_scaling_factor)
        x = self.softmax(x)
        return x


####################################################
# JetTagging Model 
####################################################
class JetTagger(pl.LightningModule):
    def __init__(self, quantize, precision, learning_rate, *args, **kwargs) -> None:
        super().__init__()

        self.input_shape = (2, 16) 
        self.quantize = quantize
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        # self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=5)

        self.model = ThreeLayer()
        if self.quantize:
            print('Loading quantized model with bitwidth', precision[0])
            self.model = QThreeLayer(self.model, precision[0], precision[1], precision[2])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        val_acc = self.accuracy(y_hat, torch.argmax(y, axis=1))
        self.log('val_acc', val_acc)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        test_acc = self.accuracy(y_hat, torch.argmax(y, axis=1))
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        return optimizer

SAVING_FOLDER = './experiment_models/'
DATA_FOLDER = '../../data/JTAG/'
DATA_FILE = '../../data/JTAG/processed_dataset.h5'


'''
robust example:
batch size      128
learning rate   0.0125
precision       4
'''

PRECISION = 4
LEARNING_RATE = 0.0125
BATCH_SIZE = 128

def main():
    # if the directory does not exist you create it
    if not os.path.exists(SAVING_FOLDER):
        os.makedirs(SAVING_FOLDER)
    # ------------------------
    # 0 PREPARE DATA
    # ------------------------
    # instantiate the data module
    data_module = JetDataModule(DATA_FILE, 
                                data_dir=DATA_FOLDER, 
                                batch_size=126, 
                                num_workers=8)
    # process the dataset if required
    if not os.path.exists(DATA_FILE):
        print("Processing data...")
        data_module.process_data()
        
    # split the dataset
    data_module.setup(0)
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print(f'Loading with quantize: {(PRECISION < 32)}')
    model = JetTagger(
        quantize=(PRECISION < 32),
        precision=[
            PRECISION, 
            PRECISION, 
            PRECISION + 3
        ],
        learning_rate=LEARNING_RATE,
    )
    # torchinfo.summary(model, input_size=(1, 16))

    tb_logger = pl_loggers.TensorBoardLogger(SAVING_FOLDER)

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
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=SAVING_FOLDER,
        filename='net_best',
        auto_insert_metric_name=False,
    )
    top_checkpoint_callback.FILE_EXTENSION = '.pkl'

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        logger=tb_logger,
        callbacks=[top_checkpoint_callback, early_stop_callback],
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
    trainer.fit(model=model, datamodule=data_module)

    # ------------------------
    # 4 EVALUTE MODEL
    # ------------------------
    # load the model from file
    checkpoint_file = os.path.join(SAVING_FOLDER, f'net_best.pkl')
    print('Loading checkpoint...', checkpoint_file)
    checkpoint = torch.load(checkpoint_file)  
    model.load_state_dict(checkpoint['state_dict'])
    
    test_results = trainer.test(model, dataloaders=data_module.test_dataloader())
    # save the results on file
    test_results_log = os.path.join(
        SAVING_FOLDER, f"accuracy.txt"
    )
    with open(test_results_log, "w") as f:
        f.write(str(test_results))
        f.close()


if __name__ == "__main__":
    main()