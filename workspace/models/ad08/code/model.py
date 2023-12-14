import os
import sys
from typing import Any, Optional

from pytorch_lightning.utilities.types import STEP_OUTPUT
sys.path.append(os.path.join(sys.path[0], "../common/")) 
sys.path.append(os.path.join(sys.path[0], "../../common/")) # For debugging 
import torchinfo
import re
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from hawq.utils.quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d, QuantBnConv2d


####################################################
# Encoder
####################################################
class Encoder(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.linear_0 = nn.Linear(64, 72)
    self.batchnorm_0 = nn.BatchNorm1d(72)
    self.relu_0 = nn.ReLU()

    self.linear_1 = nn.Linear(72, 72)
    self.batchnorm_1 = nn.BatchNorm1d(72)
    self.relu_1 = nn.ReLU()

    self.linear_2 = nn.Linear(72, 8)
    self.batchnorm_2 = nn.BatchNorm1d(8)
    self.relu_2 = nn.ReLU()

  def forward(self, x):
    # q_dense_batchnorm
    x = self.batchnorm_0(self.linear_0(x))
    x = self.relu_0(x)
    # q_dense_batchnorm_1
    x = self.batchnorm_1(self.linear_1(x))
    x = self.relu_1(x)
    # q_dense_batchnorm_2
    x = self.batchnorm_2(self.linear_2(x))
    x = self.relu_2(x)
    return x


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
        setattr(self, 'q_'+name, quant_layer)


####################################################
# Quantized Encoder
####################################################
class QEncoder(QModel):
  def __init__(self, model, weight_precision, bias_precision, act_precision, *args, **kwargs) -> None:
    super(QEncoder, self).__init__(weight_precision, bias_precision)

    self.quant_input = QuantAct(act_precision)

    self.init_dense(model, 'linear_0')
    self.batchnorm_0 = nn.BatchNorm1d(72)
    self.relu_0 = nn.ReLU()
    self.q_relu_0 = QuantAct(act_precision)

    self.init_dense(model, 'linear_1')
    self.batchnorm_1 = nn.BatchNorm1d(72)
    self.relu_1 = nn.ReLU()
    self.q_relu_1 = QuantAct(act_precision)

    self.init_dense(model, 'linear_2')
    self.batchnorm_2 = nn.BatchNorm1d(8)
    self.relu_2 = nn.ReLU()
    self.q_relu_2 = QuantAct(act_precision)

  def forward(self, x):
    x, p_sf = self.quant_input(x)
    
    # q_dense_batchnorm
    x = self.batchnorm_0(self.q_linear_0(x, p_sf))
    # q_activation 
    x, p_sf = self.q_relu_0(self.relu_0(x), p_sf)

    # q_dense_batchnorm_1
    x = self.batchnorm_1(self.q_linear_1(x, p_sf))
    # q_activation_1
    x, p_sf = self.q_relu_1(self.relu_1(x), p_sf)

    # q_dense_batchnorm_2
    x = self.batchnorm_2(self.q_linear_2(x, p_sf))
    # q_activation_2
    x, p_sf = self.q_relu_2(self.relu_2(x), p_sf)    
    return x


####################################################
# Decoder
####################################################
class Decoder(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
  
    self.linear_3 = nn.Linear(8, 72)
    self.batchnorm_3 = nn.BatchNorm1d(72)
    self.relu_3 = nn.ReLU()

    self.linear_4 = nn.Linear(72, 72)
    self.batchnorm_4 = nn.BatchNorm1d(72)
    self.relu_4 = nn.ReLU()

    self.linear_5 = nn.Linear(72, 64)
  
  def forward(self, x):
    # q_dense_batchnorm_3
    x = self.batchnorm_3(self.linear_3(x))
    x = self.relu_3(x)
    # q_dense_batchnorm_4
    x = self.batchnorm_4(self.linear_4(x))
    x = self.relu_4(x)
    # q_dense 
    x = self.linear_5(x)
    return x


####################################################
# Quantized Decoder
####################################################
class QDecoder(QModel):
  def __init__(self, model, weight_precision, bias_precision, act_precision, *args, **kwargs) -> None:
    super(QDecoder, self).__init__(weight_precision, bias_precision)

    self.quant_input = QuantAct(act_precision)

    self.init_dense(model, 'linear_3')
    self.batchnorm_3 = nn.BatchNorm1d(72)
    self.relu_3 = nn.ReLU()
    self.q_relu_3 = QuantAct(act_precision)

    self.init_dense(model, 'linear_4')
    self.batchnorm_4 = nn.BatchNorm1d(72)
    self.relu_4 = nn.ReLU()
    self.q_relu_4 = QuantAct(act_precision)

    self.init_dense(model, 'linear_5')

  def forward(self, x):
    x, p_sf = self.quant_input(x)
    
    # q_dense_batchnorm_3
    x = self.batchnorm_3(self.q_linear_3(x, p_sf))
    # q_activation 
    x, p_sf = self.q_relu_3(self.relu_3(x), p_sf)

    # q_dense_batchnorm_4
    x = self.batchnorm_4(self.q_linear_4(x, p_sf))
    # q_activation_1
    x, p_sf = self.q_relu_4(self.relu_4(x), p_sf)

    # q_dense_batchnorm_2
    x = self.q_linear_5(x, p_sf)
    return x


####################################################
# ADO8 Autoencoder 
####################################################
class AD08(pl.LightningModule):
  def __init__(self, precision=[], lr=1e-3) -> None:
    super().__init__()

    self.lr = lr

    self.encoder = Encoder()
    self.decoder = Decoder()

    if precision[0] < 32 or precision[1] < 32:
      # print('Loading quantized model with bitwidth', precision[0])
      self.encoder = QEncoder(self.encoder, precision[0], precision[1], precision[2])
      self.decoder = QDecoder(self.decoder, precision[0], precision[1], precision[2])

  def forward(self, x):
    return self.decoder(self.encoder(x))
  
  def training_step(self, batch, batch_idx):
    x, y, = batch 
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y, = batch 
    z = self.encoder(x)
    x_hat = self.decoder(z)
    val_loss = F.mse_loss(x_hat, x)
    self.log('val_loss', val_loss)

  def test_step(self, batch, batch_idx):
    x, y = batch
    z = self.encoder(x)
    x_hat = self.decoder(z)
    test_loss = F.mse_loss(x_hat, x)
    self.log('test_loss', test_loss)
  
  def configure_optimizers(self) -> Any:
    optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), self.lr)
    return optimizer


####################################################
# Helper functions
####################################################
def get_new_model(model_arch, bitwidth, args):
  model = AD08([bitwidth, bitwidth, int(bitwidth+3)])
  return model


def load_checkpoint(args, checkpoint_filename): 
   # Find model architecture and quantization scheme
   model_arch = args.arch.split('_')[0]
   bitwidth_str = re.search(f'{model_arch}_(.*)b', args.arch)
   bitwidth = int(bitwidth_str.group(1))

   model = get_new_model(model_arch, bitwidth, args)
   torchinfo.summary(model, (1, 64))
  #  model(torch.randn([2, 1, 64]))  # Update tensor shapes 
   
   # Load checkpoint 
   print('Loading checkpoint...', checkpoint_filename)
   checkpoint = torch.load(checkpoint_filename)
   model.load_state_dict(checkpoint['state_dict'])  # strict=False
   return model
