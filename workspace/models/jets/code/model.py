"""
  todo: move QModel to common 
"""
import ast
import os
from statistics import mean 
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
      # print('Loading quantized model with bitwidth', precision[0])
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


# ####################################################
# # Helper functions
# ####################################################
def get_model_index_and_relative_accuracy(path, batch_size, learning_rate, precision, num_tests=5):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    performances = []
    max_acc = 0
    max_acc_index = 0
    for i in range (1, num_tests+1):
        file_path = os.path.join(
                            path, 
                            f'bs{batch_size}_lr{learning_rate}/' \
                            f'JTAG_{precision}b/accuracy_{i}.txt'
                    )
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
            # warnings.warn("Warning: " + file_path + " not found!")
            continue
        
    if len(performances) == 0:
        # warnings.warn(f"Attention: There is no accuracy value for the model: " \
        #               f"bs{batch_size}_lr{learning_rate}/JTAG_{precision}b")
        return
    
    return mean(performances), max_acc_index
  
  
def load_model(path, batch_size, learning_rate, precision):
    '''
    Method used to get the model and the relative accuracy
    '''
    accuracy, idx = get_model_index_and_relative_accuracy(path, batch_size, learning_rate, precision)
    model_path = path + f'bs{batch_size}_lr{learning_rate}/JTAG_{precision}b/net_{idx}_best.pkl'
    
    # load the model
    model = JetTagger(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model(torch.randn((16, 16)))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    return model, accuracy


def get_accuracy_with_noise(path, batch_size, learning_rate, precision, noise_type, percentage):
    '''
    Return the accuracy achieved by the Model with a certain level of noise
    '''

    file_path = os.path.join(
                    path,
                    f'bs{batch_size}_lr{learning_rate}/' \
                    f'JTAG_{precision}b/accuracy_{noise_type}_{percentage}.txt'
                )
    noise_acc = -1
    try:
        acc_file = open(file_path)
        acc_text = acc_file.read()
        acc = ast.literal_eval(acc_text)
        noise_acc = acc[0]['test_acc']
        acc_file.close()
    except Exception as e:
        # warnings.warn("Warning: " + file_path + " not found!")
        return 0
    return noise_acc



