
import os
import sys
import ast
import warnings
from itertools import starmap
from statistics import mean
from utils_pt import unnormalize, emd 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hawq.utils import QuantAct, QuantLinear, QuantConv2d
from collections import OrderedDict
from telescope_pt import telescopeMSE8x8
from autoencoder_datamodule import ARRANGE, ARRANGE_MASK

# add Jacobian and AT
module_path = os.path.abspath(os.path.join('../../common/benchmarks/')) 
sys.path.insert(0, module_path)
from jacobian import JacobianReg as jReg


"""
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 8, 1)]         0         
                                                                 
 input_qa (QActivation)      (None, 8, 8, 1)           0         
                                                                 
 conv2d_0_m (FQConv2D)       (None, 4, 4, 8)           80        
                                                                 
 accum1_qa (QActivation)     (None, 4, 4, 8)           0         
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 encoded_vector (FQDense)    (None, 16)                2064      
                                                                 
 encod_qa (QActivation)      (None, 16)                0         
                                                                 
=================================================================
Total params: 2,144
Trainable params: 2,144
Non-trainable params: 0

_________________________________________________________________
Model: "decoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 decoder_input (InputLayer)  [(None, 16)]              0         
                                                                 
 de_dense_final (Dense)      (None, 128)               2176      
                                                                 
 de_reshape (Reshape)        (None, 4, 4, 8)           0         
                                                                 
 conv2D_t_0 (Conv2DTranspose  (None, 8, 8, 8)          584       
 )                                                               
                                                                 
 conv2d_t_final (Conv2DTrans  (None, 8, 8, 1)          73        
 pose)                                                           
                                                                 
 decoder_output (Activation)  (None, 8, 8, 1)          0         
                                                                 
=================================================================
Total params: 2,833
Trainable params: 2,833
Non-trainable params: 0
"""


CALQ_MASK = torch.tensor(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
)

encoder_type = {
    'baseline': (3, 8, 128),
    'large': (5, 32, 288),
    'small': (3, 1, 16),
}


class BaseEncoder(nn.Module):
    def __init__(self, econ_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoded_dim = 16
        self.shape = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None

        kernel_size, num_kernels, fc_input = encoder_type[econ_type]

        self.conv = nn.Conv2d(1, num_kernels, kernel_size=kernel_size, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.enc_dense = nn.Linear(fc_input, self.encoded_dim)
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.enc_dense(x))
        return x

class QuantizedEncoder(nn.Module):
    def __init__(self, model, weight_precision, bias_precision, act_precision) -> None:
        super().__init__()

        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        self.act_precision = act_precision
        self.quant_input = QuantAct(activation_bit=self.act_precision)
        self.quant_relu = QuantAct(activation_bit=self.act_precision)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        base_layer = getattr(model, 'conv')
        hawq_layer = QuantConv2d(self.weight_precision, self.bias_precision)
        hawq_layer.set_param(base_layer)
        setattr(self, 'conv', hawq_layer)

        base_layer = getattr(model, 'enc_dense')
        hawq_layer = QuantLinear(self.weight_precision, self.bias_precision)
        hawq_layer.set_param(base_layer)
        setattr(self, 'enc_dense', hawq_layer)
    
    def forward(self, x):
        x, p_sf = self.quant_input(x)
        x, w_sf = self.conv(x, p_sf)
        x = self.relu1(x)
        x, p_sf = self.quant_relu(x, p_sf, w_sf)

        x = self.flatten(x)
        x = self.relu2(self.enc_dense(x, p_sf))
        return x


class AutoEncoder(pl.LightningModule):
    def __init__(self, quantize, precision, learning_rate, econ_type, jacobian_reg=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoded_dim = 16
        self.shape = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None
        self.quantize = quantize
        self.learning_rate = learning_rate

        self.encoder = BaseEncoder(econ_type) 
        if self.quantize:
            self.encoder = QuantizedEncoder(self.encoder, precision[0], precision[1], precision[2])
        
        self.decoder = nn.Sequential(OrderedDict([
            ("dec_dense", nn.Linear(self.encoded_dim, 128)),
            ("relu1", nn.ReLU()),
            ("unflatten", nn.Unflatten(1, (8, 4, 4))),
            (
                "convtrans2d1",
                nn.ConvTranspose2d(
                    8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            ),
            ("relu2", nn.ReLU()),
            (
                "convtrans2d2",
                nn.ConvTranspose2d(
                    8,
                    self.shape[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
            ("sigmoid", nn.Sigmoid()),
        ]))
        
        self.loss = telescopeMSE8x8
        
        # Jacobian rgularization
        self.regularizer = jReg(n=1)
        self.lambda_JR = jacobian_reg
        if self.lambda_JR > 0:
            print('\n')
            print('*'*100)
            print(f'\t\tJACOBIAN REGULARIZATION ACTIVE (lambda={self.lambda_JR})')
            print('*'*100)
            print('\n')
        

    def invert_arrange(self):
        """
        Invert the arrange mask
        """
        remap = []
        hashmap = {}  # cell : index mapping
        found_duplicate_charge = len(ARRANGE[ARRANGE_MASK == 1]) > len(
            torch.unique(ARRANGE[ARRANGE_MASK == 1])
        )
        for i in range(len(ARRANGE)):
            if ARRANGE_MASK[i] == 1:
                if found_duplicate_charge:
                    if CALQ_MASK[i] == 1:
                        hashmap[int(ARRANGE[i])] = i
                else:
                    hashmap[int(ARRANGE[i])] = i
        for i in range(len(torch.unique(ARRANGE))):
            remap.append(hashmap[i])
        return torch.tensor(remap)

    def map_to_calq(self, x):
        """
        Map the input/output of the autoencoder into CALQs orders
        """
        remap = self.invert_arrange()
        image_size = self.shape[0] * self.shape[1] * self.shape[2]
        reshaped_x = torch.reshape(x, (len(x), image_size))
        reshaped_x[:, ARRANGE_MASK == 0] = 0
        return reshaped_x[:, remap]

    def set_val_sum(self, val_sum):
        self.val_sum = val_sum

    def predict(self, x):
        decoded_Q = self(x)
        encoded_Q = self.encoder(x)
        encoded_Q = torch.reshape(encoded_Q, (len(encoded_Q), self.encoded_dim, 1))
        return decoded_Q, encoded_Q

    # Pytorch Lightning specific methods
    def forward(self, x):
        # print(type(self.encoder(x.float())))
        return self.decoder(self.encoder(x.float()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # lr=1e-3
        return optimizer

    def training_step(self, batch, batch_idx):
        input, target = batch
        input.requires_grad = True # this is essential!
        input_hat = self(input)
        loss = self.loss(input_hat, target)
        J_loss = self.regularizer(input, input_hat)
        loss = loss + self.lambda_JR * J_loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        input_hat = self(input)
        loss = self.loss(input_hat, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        input_calQ = self.map_to_calq(target)
        output_calQ_fr = self.map_to_calq(output)
        input_calQ = torch.stack(
            [input_calQ[i] * self.val_sum[i] for i in range(len(input_calQ))]
        )  # shape = (batch_size, 48)
        output_calQ = unnormalize(
            torch.clone(output_calQ_fr), self.val_sum
        )  # ae_out
        return {'input_calQ': input_calQ, 'output_calQ': output_calQ}
    
    def test_epoch_end(self, outputs):
        # concatenate all the tensor coming from all the batches processed
        input_calQ = torch.cat(
            ([i_cal['input_calQ'] for i_cal in outputs]),
            dim=0
        )
        output_calQ = torch.cat(
            ([i_cal['output_calQ'] for i_cal in outputs]),
            dim=0
        )
        # compute the average EMD
        emd_list = list(starmap(emd, zip(input_calQ.tolist(), output_calQ.tolist())))
        average_emd = torch.mean(torch.Tensor(emd_list)).item()
        result = {'AVG_EMD': average_emd}
        self.log_dict(result)
        return result


# ---------------------------------------------------------------------------- #
#                                Utility methods                               #
# ---------------------------------------------------------------------------- #
def load_model(path, batch_size, learning_rate, precision, size, jreg=0.0, aug_percentage=0.0):
    '''
    Method used to get the model and the relative EMD value
    '''
    lr = "{:.10f}".format(float(learning_rate)).rstrip('0')
    emd, idx = get_model_index_and_relative_EMD(path, batch_size, learning_rate, precision, size, 3, jreg, aug_percentage)
    
    if aug_percentage > 0:
        model_path = path + f'bs{batch_size}_lr{lr}/ECON_AUG_{precision}b/{size}/net_{idx}_{aug_percentage}_best.pkl'
    elif jreg > 0:
        model_path = path + f'bs{batch_size}_lr{lr}/ECON_JREG_{precision}b/{size}/net_{idx}_{jreg}_best.pkl'
    else:
        model_path = path + f'bs{batch_size}_lr{lr}/ECON_{precision}b/{size}/net_{idx}_best.pkl'
    
    # load the model
    model = AutoEncoder(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
        econ_type=size
    )
    
    # to set the map location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model(torch.randn((1, 1, 8, 8)).to(device))  # Update tensor shapes 
    model_param = torch.load(model_path, map_location=device)
    model.load_state_dict(model_param['state_dict'])
    
    return model, emd


def get_emd_with_noise(path, batch_size, learning_rate, precision, size, noise_type, percentage):
    '''
    Return the EMD achieved by the Model with a certain level of noise
    '''
    lr = "{:.10f}".format(float(learning_rate)).rstrip('0')
    file_path = os.path.join(
                path,
                f'bs{batch_size}_lr{lr}/' \
                f'ECON_{precision}b/{size}/emd_{noise_type}_{percentage}.txt'
        ) 
    
    noise_emd = -1
    try:
        emd_file = open(file_path)
        emd_text = emd_file.read()
        if not emd_text.startswith('['):
            noise_emd = float(emd_text)
        else:
            emd = ast.literal_eval(emd_text)
            noise_emd = emd[0]['AVG_EMD']
        emd_file.close()
    except Exception as e:
        # warnings.warn("Warning: " + file_path + " not found!")
        return -1
    return noise_emd


def get_model_index_and_relative_EMD(path, batch_size, learning_rate, precision, size, num_tests=3, jreg=0.0, aug_percentage=0.0):
    '''
    Return the average EMDs achieved by the model and the index of best experiment
    '''
    EMDs = []
    min_emd = 1000
    min_emd_index = 0
    lr = "{:.10f}".format(float(learning_rate)).rstrip('0')
    for i in range (1, num_tests+1):
        if aug_percentage > 0:
            file_path = os.path.join(
                        path,
                        f'bs{batch_size}_lr{lr}/' \
                        f'ECON_AUG_{precision}b/{size}/{size}_emd_{i}_{aug_percentage}.txt'
                    )
        elif jreg > 0:
            file_path = os.path.join(
                        path,
                        f'bs{batch_size}_lr{lr}/' \
                        f'ECON_JREG_{precision}b/{size}/{size}_emd_{i}_{jreg}.txt'
                    )
        else:
            file_path = os.path.join(
                        path,
                        f'bs{batch_size}_lr{lr}/' \
                        f'ECON_{precision}b/{size}/{size}_emd_{i}.txt'
                    )
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
            # warnings.warn("Warning: " + file_path + " not found!")
            continue
    
    if len(EMDs) == 0:
        warnings.warn(f"Attention: There is no EMD value for the model: " \
                      f"bs{batch_size}_lr{lr}/ECON_{precision}b/{size}")
        return 0, -1
    
    return mean(EMDs), min_emd_index