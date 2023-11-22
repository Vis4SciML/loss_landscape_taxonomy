"""
  todo: move QModel to common 
"""
import re 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchinfo
from hawq.utils import QuantAct, QuantLinear


"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
JetTagger                                [1, 5]                    --
├─ThreeLayer: 1-1                        [1, 5]                    --
│    └─Linear: 2-1                       [1, 64]                   1,088
│    └─ReLU: 2-2                         [1, 64]                   --
│    └─Linear: 2-3                       [1, 32]                   2,080
│    └─ReLU: 2-4                         [1, 32]                   --
│    └─Linear: 2-5                       [1, 32]                   1,056
│    └─ReLU: 2-6                         [1, 32]                   --
│    └─Linear: 2-7                       [1, 5]                    165
│    └─Softmax: 2-8                      [1, 5]                    --
==========================================================================================
Total params: 4,389
Trainable params: 4,389
Non-trainable params: 0
Total mult-adds (M): 0.00
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.02
Estimated Total Size (MB): 0.02
==========================================================================================
"""


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

        self.dense_1 = nn.Linear(16, 64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 32)
        self.dense_4 = nn.Linear(32, 5)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.act(self.dense_1(x))
        x = self.act(self.dense_2(x))
        x = self.act(self.dense_3(x))
        return self.softmax(self.dense_4(x))


####################################################
# MLP Jet-Tagginer
####################################################
class QThreeLayer(QModel):
    def __init__(self, model, weight_precision=6, bias_precision=6, act_precision=6):
        super(QThreeLayer, self).__init__(weight_precision, bias_precision)

        self.quant_input = QuantAct(act_precision)
        
        self.init_dense(model, "dense_1")
        self.quant_act_1 = QuantAct(act_precision)

        self.init_dense(model, "dense_2")
        self.quant_act_2 = QuantAct(act_precision)

        self.init_dense(model, "dense_3")
        self.quant_act_3 = QuantAct(act_precision)

        self.init_dense(model, "dense_4")

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)
        
        x = self.act(self.dense_1(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_1(x, act_scaling_factor)
        
        x = self.act(self.dense_2(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_2(x, act_scaling_factor)
        
        x = self.act(self.dense_3(x, act_scaling_factor))
        x, act_scaling_factor = self.quant_act_3(x, act_scaling_factor)
        
        x = self.dense_4(x, act_scaling_factor)
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


####################################################
# Helper functions
####################################################
def load_checkpoint(args, checkpoint_filename): 
   # quantization scheme 
   model_arch = args.arch.split('_')[0]
   bitwidth_str = re.search(f'{model_arch}_(.*)b', args.arch)
   bitwidth = int(bitwidth_str.group(1))

   model = JetTagger([bitwidth, bitwidth, int(bitwidth+3)])
   torchinfo.summary(model, (1, 16))
   
   # Load checkpoint 
   print('Loading checkpoint...', checkpoint_filename)
   checkpoint = torch.load(checkpoint_filename)
   model.load_state_dict(checkpoint['state_dict'])  # strict=False
   return model
