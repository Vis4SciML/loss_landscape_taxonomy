import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
                nn.BatchNorm2d(out_channels)
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
    def __init__(self, weight_precision, bias_precision, per_channel):
        super().__init__()
        self.weight_precision = weight_precision
        self.bias_precision = bias_precision
        self.per_channel = per_channel
        
        
    def init_dense(self, model, name, rename=None):
        layer = getattr(model, name)
        quant_layer = QuantLinear(self.weight_precision, self.bias_precision)
        quant_layer.set_param(layer)
        if rename:
            setattr(self, rename, quant_layer)
        else:
            setattr(self, name, quant_layer)
        
        
    def init_conv2d(self, model, name, rename=None):
        layer = getattr(model, name)
        quant_layer = QuantConv2d(self.weight_precision, 
                                  self.bias_precision, 
                                  per_channel=self.per_channel)
        quant_layer.set_param(layer)
        if rename:
            setattr(self, rename, quant_layer)
        else:
            setattr(self, name, quant_layer)
        
        
    def init_bn_conv2d(self, model, bn_name, conv_name, rename=None):
        bn_layer = getattr(model, bn_name)
        conv_layer = getattr(model, conv_name)
        quant_layer = QuantBnConv2d(self.weight_precision, 
                                    self.bias_precision,
                                    per_channel=self.per_channel)
        quant_layer.set_param(conv_layer, bn_layer)
        if rename:
            setattr(self, rename, quant_layer)
        else:
            setattr(self, conv_name, quant_layer)


class QBlock(QModel):
    '''
    Class that build the basic blocks of a ResNet
    '''
    def __init__(self, block, weight_precision=8, bias_precision=8, act_precision=11):
        super(QBlock, self).__init__(weight_precision, bias_precision, True)
        
        # shortcut
        self.resize_identity = False
        if hasattr(block, "shortcut") and len(block.shortcut):
            self.init_bn_conv2d(block.shortcut, "1", "0", "shortcut")
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


    def forward(self, x):
        x, act_scale = x    #unpack the input coming from nn.Sequential
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
        
        out = (out, act_scale)
        
        return out


class QResNet(QModel):
    def __init__(self, model, num_blocks, weight_precision=8, bias_precision=8, act_precision=11):
        super(QResNet, self).__init__(weight_precision, bias_precision, True)
        
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
        
        x = (x, act_scale)  # pack
        x = self.QBlocks(x) 
        x, act_scale = x    # unpack
        
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
        self.train_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=10)


    # Pytorch Lightning specific methods
    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=1e-5)  # lr=1e-3
        return optimizer


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.train_accuracy(y_hat, y)
        
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # Update validation accuracy
        self.val_accuracy(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)
        
        
    def test_step(self, batch, batch_idx):
        x, y, = batch 
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        # update test accuracy
        self.test_accuracy(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)


# ---------------------------------------------------------------------------- #
#                               Utility functions                              #
# ---------------------------------------------------------------------------- #
def get_accuracy(path, batch_size, learning_rate, precision):
    folder_path = os.path.join(
        path,
        f'bs{batch_size}_lr{learning_rate}/RN08_{precision}b/'
    )
    
    # get the accuracy
    accuracy_file = os.path.join(folder_path, "accuracy_1.txt")
    try:
        f = open(accuracy_file)
        text = f.read()
        acc = ast.literal_eval(text)
        value = acc[0]['test_accuracy']
        f.close()
    except:
        print(f"File not found! ({accuracy_file})")
        return -1
    
    return value


def get_model_and_accuracy(path, batch_size, learning_rate, precision, name="net_1_best"):
    '''
    Return the model and its accuracy
    '''
    folder_path = os.path.join(
        path,
        f'bs{batch_size}_lr{learning_rate}/RN08_{precision}b/'
    )
    
    # get the model
    model_file = os.path.join(folder_path, f"{name}.pkl")
    model = RN08(
        quantize=(precision < 32),
        precision=[
            precision,
            precision,
            precision+3
        ],
        learning_rate=learning_rate,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model(torch.randn((1,3,32,32)).to(device))  # Update tensor shapes 
    try:
        model_param = torch.load(model_file, map_location=device)
        model.load_state_dict(model_param['state_dict'])
    except:
        print(f"File not found! ({model_file})")
        return None, -1
    # get the accuracy
    value = get_accuracy(path, batch_size, learning_rate, precision)
    
    return model, value

def get_dataloader(path, batch_size):
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_dataset = datasets.CIFAR10(root=path, 
                                    train=False, 
                                    download=True, 
                                    transform=transform)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=0,
                            drop_last=True)
    
    return test_loader

def get_cifar10_loaders(path, batch_size, noise=False, percentage=0):
    '''
    Utility function to retrieve the train, validation and test
    dataloaders.
    '''
    train_ds = datasets.CIFAR10(
            root=path,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.247, 0.243, 0.261)),
                ])
            )
    test_ds = datasets.CIFAR10(
            root=path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.247, 0.243, 0.261)),
                ])
            )
    
    # perturbation to the labels 
    if noise:
        print("Adding noise to the training data...")
        train_ds.targets = randomize_labels(torch.tensor(train_ds.targets), portion=percentage)
    
    total_size = len(test_ds)
    val_size = int(0.6 * total_size)
    test_size = total_size - val_size
    
    test_ds, validation_ds = random_split(test_ds, [test_size, val_size])
    
    
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=0,
                              drop_last=True)
    validation_loader = DataLoader(validation_ds, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=0,
                             drop_last=True)
    test_loader = DataLoader(test_ds, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=0,
                             drop_last=True)
    
    print("Dataset Size:")
    print(len(train_loader), len(validation_loader), len(test_loader))
    
    return train_loader, validation_loader, test_loader


def randomize_labels(target, portion=0.2):
    '''
    Stress test randomizing labels
    '''
    num_samples = target.size(0)
    num_to_randomize = int(num_samples * portion)
    indices_to_randomize = torch.randperm(num_samples)[:num_to_randomize]
    randomized_labels = torch.randint_like(target[indices_to_randomize], 0, 10)
    target[indices_to_randomize] = randomized_labels
    return target


def get_accuracy_with_noise(path, batch_size, learning_rate, precision, noise_type):
    '''
    Return the accuracy of the model computed with noisy dataset
    '''
    
    file_path = os.path.join(
                path,
                f'bs{batch_size}_lr{learning_rate}/' \
                f'RN08_{precision}b/accuracy_{noise_type}.txt'
        )
    
    acc = -1
    try:
        acc_file = open(file_path)
        acc_text = acc_file.read()
        acc = float(acc_text)
        acc_file.close()
    except Exception as e:
        print(f"{file_path} not found!")
        return -1
    
    return acc 


if __name__ == "__main__":
    
    # model = get_model_and_accuracy("/data/tbaldi/work/checkpoint/", 16, 0.003125, 2)
    # print(model)
    train_loader, val_loader, test_loader = get_cifar10_loaders('../../../data/RN08', 1024)
    
    model = RN08(False, [4, 4, 7], 0.0015625)
    # print(model)
    torchinfo.summary(model, input_size=(1, 3, 32, 32))
    
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
        save_top_k=3,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath='./',
        filename=f'net_full_precision_best',
        auto_insert_metric_name=False,
    )
    
    tb_logger = pl_loggers.TensorBoardLogger('./')
    
    trainer = pl.Trainer(
        max_epochs=100,
        logger=tb_logger,
        callbacks=[top_checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model=model, 
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
