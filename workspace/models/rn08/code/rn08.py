import torch
import torch.nn as nn
import torch.nn.functional as F
from hawq.utils import QuantAct, QuantLinear, QuantConv2d, QuantAveragePool2d, QuantBnConv2d

# ---------------------------------------------------------------------------- #
#                                  Tiny ResNet                                 #
# ---------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    '''
    Class that build the basic blocks of a ResNet
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # add
        out = F.relu(out)
        return out

class TinyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.stack1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        
        self.stack2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        
        self.stack3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # update the input channels wit the previous uotput
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) #flatten
        out = self.linear(out)
        return F.softmax(out, dim=1)


# ---------------------------------------------------------------------------- #
#                               Quantized ResNet                               #
# ---------------------------------------------------------------------------- #
class QModel(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        
    def init_dense(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantLinear(weight_bit=self.weight_precision, 
                                  bias_bit=self.bias_precision)
        
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)
        
    def init_conv2d(self, model, name):
        layer = getattr(model, name)

class QuantizedBlock(nn.Module):
    '''
    Class that build the basic blocks of a ResNet
    '''
    def __init__(self, in_channels, out_channels, stride=1, weight_precision=8, bias_precision=8):
        super(QuantizedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # add
        out = F.relu(out)
        return out

class RN08(nn.Module):
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        
        
        
# def resnet_eembc():
#     return ResNet(BasicBlock, [2, 2, 2])  # This creates a ResNet with 18 layers (2+2+2=6 blocks)
