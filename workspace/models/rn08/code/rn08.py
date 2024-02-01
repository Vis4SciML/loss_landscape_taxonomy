import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import pytorch_lightning as pl
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
        super(TinyResNet, self).__init__()
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
            # update the input channels wit the previous output
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
# TODO: check it with javi
class QModel(nn.Module):
    '''
    Class used to quantize the layers of the model
    '''
    def __init__(self, weight_precision, bias_precision):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        
        
    def init_dense(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantLinear(self.weight_precision, self.bias_precision)
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)
        
        
    def init_conv2d(self, model, name):
        layer = getattr(model, name)
        quant_layer = QuantConv2d(self.weight_precision, self.bias_precision)
        quant_layer.set_param(layer)
        setattr(self, name, quant_layer)
        
        
    def init_bn_conv2d(self, model, bn_name, conv_name):
        bn_layer = getattr(model, bn_name)
        conv_layer = getattr(model, conv_name)
        quant_layer = QuantBnConv2d(self.weight_precision, self.bias_precision)
        quant_layer.set_param(conv_layer, bn_layer)
        setattr(self, conv_name, quant_layer)


class QBlock(QModel):
    '''
    Class that build the basic blocks of a ResNet
    '''
    def __init__(self, block, weight_precision=8, bias_precision=8, act_precision=11):
        super(QBlock, self).__init__(weight_precision, bias_precision)
        
        # shortcut
        self.resize_identity = False
        if hasattr(block, "shortcut") and len(block.shortcut):
            self.init_conv2d(block.shortcut, "0")
            self.resize_identity = True

        
        # init the precisions
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        self.act_precision = act_precision
        
        self.quant_relu1 = QuantAct(self.act_precision)
        self.quant_relu2 = QuantAct(self.act_precision)
        self.quant_relu3 = QuantAct(self.act_precision)
        
        self.init_bn_conv2d(block, bn_name="bn1", conv_name="conv1")
        self.init_bn_conv2d(block, bn_name="bn2", conv_name="conv2")


    def forward(self, x, act_scale):
        # shortcut
        if self.resize_identity:
            short_x, short_w_scale = self.shortcut(x, act_scale)
        
        out, w_scale = self.conv1(x, act_scale)
        out = F.relu(out)
        out, act_scale = self.quant_relu1(out, act_scale, w_scale)
        
        out, w_scale = self.conv2(out, act_scale)
        out = F.relu(out)
        out, act_scale = self.quant_relu2(out, act_scale, w_scale)
        
        if self.resize_identity:
            out += short_x
            out = F.relu(out)
            out, act_scale = self.quant_relu3(out, act_scale, w_scale, short_x, short_w_scale)
        else:
            out += x
            out = F.relu(out)
            out, act_scale = self.quant_relu3(out, act_scale, w_scale)
        
        return out, act_scale


class QResNet(QModel):
    def __init__(self, model, num_blocks, weight_precision=8, bias_precision=8, act_precision=11):
        super(QResNet, self).__init__(weight_precision, bias_precision)
        
        # init the precisions
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        self.act_precision = act_precision
        
        self.quant_relu = QuantAct(self.act_precision)
        self.quant_input = QuantAct(self.act_precision)
        
        self.init_bn_conv2d(model, bn_name="bn1", conv_name="conv1")
        
        # get the basic blocks
        layers = []
        for index, block_per_layer in enumerate(num_blocks, 1):
            stack = getattr(model, f"stack{index}")
            for i in range(block_per_layer):
                block = QBlock(stack[i], weight_precision, bias_precision, act_precision)
                layers.append(block)
        self.QBlocks = nn.Sequential(*layers)
        
        self.avgpool = QuantAveragePool2d(8)
        self.init_dense(model, "linear")


    def forward(self, x):
        x, act_scale = self.quant_input(x)
        
        x, w_scale = self.conv1(x, act_scale)
        x = F.relu(x)
        x, act_scale = self.quant_relu(x, act_scale, w_scale)
        
        x, act_scale = self.QBlocks(x, act_scale)
        
        x, act_scale = self.avgpool(x, act_scale)
        x = x.view(x.size(0), -1) #flatten
        x = self.linear(x, act_scale)
        
        return F.softmax(x, dim=1)

# ---------------------------------------------------------------------------- #
#                           PyTorch lightning module                           #
# ---------------------------------------------------------------------------- #
class RN08(pl.LightningModule):
    def __init__(self, quantize, precision, learning_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate

        self.model = TinyResNet(BasicBlock, [2,2,2])
        if quantize:
            self.model = QResNet(self.model, [2,2,2], precision[0], precision[1], precision[2])
        
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)


    def predict(self, x):
        pass


    # Pytorch Lightning specific methods
    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # lr=1e-3
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        # test_acc = self.accuracy(y_hat, torch.argmax(y, axis=1))
        self.log('test_loss', test_loss)
        # self.log('test_acc', test_acc)
        
        
    def test_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        test_acc = self.accuracy(y_hat, torch.argmax(y, axis=1))
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_train = datasets.CIFAR10(root='../../../data/RN08', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='../../../data/RN08', train=False, download=True, transform=transform)

    train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(cifar10_test, batch_size=64, shuffle=False, num_workers=4)

    model = RN08(True, [8, 8, 11], 0.001)
    trainer = pl.Trainer(max_epochs=5)  # Adjust max_epochs and gpus according to your setup

    torchinfo.summary(model, input_size=(1, 3, 32, 32))

    trainer.fit(model, train_loader, test_loader)


# TinyResNet(
#   (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (stack1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (stack2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential(
#         (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2))
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (stack3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential(
#         (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2))
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (shortcut): Sequential()
#     )
#   )
#   (avgpool): AvgPool2d(kernel_size=8, stride=8, padding=0)
#   (linear): Linear(in_features=64, out_features=10, bias=True)
# )