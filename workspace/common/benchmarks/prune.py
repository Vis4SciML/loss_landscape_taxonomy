"""
QPrune developed by Javier Campos (Fermilab)
HAWQIterativePruning developed by Tommaso Baldi (INFN)

TODO:
    - messages are not showing properly
"""
import os
import logging
from typing import Dict, Callable, Tuple, Optional
import numpy as np
from typing import Any
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl 
import torch 
import torch.nn as nn
from torch.nn.utils import prune
from hawq.utils.quantization_utils.quant_modules import QuantConv2d, QuantLinear, QuantBnConv2d
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning.callbacks import Checkpoint


log = logging.getLogger(__name__)
mode_dict = {"min": torch.lt, "max": torch.gt}


class QPrune():
    def __init__(self, method) -> None:
        self.method = method
        self.pruned_layers = []

    
    def __str__(self) -> str:
        pass


    def prune_layers(self, model, amount, module_name=''):
        """Recursively prune all HAWQ modules"""
        for name, module in model.named_children():
            #######################
            # Prune HAWQ layers
            #######################
            if isinstance(module, QuantLinear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                self._register_layer(module_name+"."+name)
                prune.remove(module, 'weight')  # remove all those values in the global pruned model
            elif isinstance(module, QuantBnConv2d):
                prune.l1_unstructured(module.conv, name='weight', amount=amount)
                # prune.l1_unstructured(module.bn, name='weight', amount=self.amount)
                self._register_layer(module_name+"."+name)
                prune.remove(module.conv, 'weight')  # remove all those values in the global pruned model
            elif isinstance(module, QuantConv2d):
                prune.l1_unstructured(module.conv, name='weight', amount=amount)
                self.pruned_layers.append(module_name+"."+name)
                self._register_layer(module_name+"."+name)
                prune.remove(module.conv, 'weight')
            #######################
            # Standard torch layers
            #######################
            elif isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                self._register_layer(module_name+"."+name)
                prune.remove(module, 'weight')
            elif isinstance(module, nn.ConvTranspose2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                self._register_layer(module_name+"."+name)
                prune.remove(module, 'weight')
            elif isinstance(module, (nn.Sequential, nn.Module)): # for nested modules, sequential models, blocks, etc 
                self.prune_layers(module, module_name=module_name+"."+name, amount=amount)
    
    
    def print_pruned_layers(self):
        """Sanity check - list of pruned layers"""
        print("Pruned Layers:")
        for layer in self.pruned_layers:
            print(f"\t{layer}")


    def _register_layer(self, name):
        if name not in self.pruned_layers:
            self.pruned_layers.append(name)


    def _get_layer(self, layer_name, module):
        for name in layer_name.split('.')[1:]:
            module = getattr(module, name)
        return module


    def _check_sparisty(self, module):
        if isinstance(module, QuantLinear):
            nonzero = torch.count_nonzero(module.weight)
            total_params = torch.prod(torch.tensor(module.weight_integer.shape))
        elif isinstance(module, (QuantBnConv2d, QuantConv2d)):
            nonzero = torch.count_nonzero(module.conv.weight)
            total_params = torch.prod(torch.tensor(module.conv.weight.shape))
        elif isinstance(module, (nn.Linear, nn.ConvTranspose2d)):
            nonzero = torch.count_nonzero(module.weight)
            total_params = torch.prod(torch.tensor(module.weight.shape))
        if nonzero == 0:
            return 1
        else:
            return (1 - (nonzero/total_params)) 
        
        
    def profile(self, model):
        """Return the sparisty levels per layer"""
        print("Checking sparisty: ")
        for layer_name in self.pruned_layers:
            module = self._get_layer(layer_name, model)
            sparsity = self._check_sparisty(module)
            print(f"  {layer_name}:\t{sparsity}")
        return (self.pruned_layers)
    
    
    
class HAWQIterativePruning(Checkpoint):
    '''
    
    tip:: Saving and restoring multiple early stopping callbacks at the same time is supported under variation in the
        following arguments:

        *monitor, mode*

        Read more: :ref:`Persisting Callback State <extensions/callbacks_state:save callback state>`
    """
    '''
    
    mode_dict = {"min": torch.lt, "max": torch.gt}
    FILE_EXTENSION = ".ckpt"
    
    def __init__(
        self, 
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        ratios: list = [],
        verbose: bool = False,
        mode: str = "min",
        method: str = "L1Unstructured",
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        check_on_train_epoch_end: bool = False,
        save_weights_only: bool = False
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.ratios = ratios
        self.verbose = verbose
        self.mode = mode 
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.save_weights_only = save_weights_only
        
        self.dirpath: Optional[_PATH]
        self.__init_ckpt_dir(dirpath, filename)
        
        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")
        self.best_model = None
        self.pruning_step = 0
        self.wait_count = 0
        self.pruned_epoch = []
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        
        self.prune = QPrune(method)
        
    def __init_ckpt_dir(self, dirpath: Optional[_PATH], filename: Optional[str]) -> None:
        self._fs = get_filesystem(dirpath if dirpath else "")

        if dirpath and self._fs.protocol == "file":
            dirpath = os.path.realpath(dirpath)

        self.dirpath = dirpath
        self.filename = filename
        
        
    def _get_file_path(self, trainer: "pl.Trainer") -> str:
        '''
        Get the file path of the checkpoint
        '''
        file_path = self._format_checkpoint_name()
        
        version = 1
        while self.file_exists(file_path, trainer):
            file_path = self._format_checkpoint_name(ver=version)
            version += 1
            
        return file_path
        
        
    def _validate_condition_metric(self, logs: Dict[str, torch.Tensor]) -> bool:
        '''
        Check if the metric is actually logged by the model
        '''
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Iterative pruning conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `HAWQIterativePruning` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            raise RuntimeError(error_msg)

        return True
        
        
    def _prune_message(self, reason: str = None) -> str:
        '''
        Formats a log message that informs the users that the model is going to be prune.
        '''
        msg = f"The model is converged. New pruning ratio: {self.ratios[self.pruning_step]}"
        if reason:
            msg = f"{msg} \n {reason}"
        return msg
    
    
    def _improvement_message(self, current: torch.Tensor) -> str:
        '''
        Formats a log message that informs the user about an improvement in the monitored score.
        '''
        
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg
            
            
    def _run_pruning_check(self, trainer: pl.Trainer, pl_module: LightningModule):
        '''
        Checks if the model is pruning condition is met and if so we prune the model
        '''
        logs = trainer.callback_metrics
        
        if trainer.fast_dev_run or not self._validate_condition_metric(logs):
            return
        
        current = logs[self.monitor].squeeze()
        should_prune, reason = self._evaluate_pruning_criteria(current)
        if reason and self.verbose:
            log.info(reason)
            
        if should_prune:
            self.pruned_epoch.append(trainer.current_epoch)
            
            # save the model
            file_path = self._get_file_path(trainer)
            trainer.save_checkpoint(file_path, self.save_weights_only)
            
            # stop condition - last pruning is converged too
            if self.pruning_step == len(self.ratios):
                should_stop = trainer.strategy.reduce_boolean_decision(True, all=False)
                trainer.should_stop = trainer.should_stop or should_stop
                return
                
            # pruning
            msg = self._prune_message(reason)
            self.prune.prune_layers(pl_module, self.ratios[self.pruning_step], 'model')
            
            if self.verbose:
                print(self.prune.profile(pl_module))
                log.info(msg)
            
            self.pruning_step += 1
            # reset the best model
            torch_inf = torch.tensor(np.Inf)
            self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        
        
    def _evaluate_pruning_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        '''
        Check if the model satisfy the criteria to be prune
        '''
        should_prune = False
        reason = None
        
        if self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_prune = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_prune = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to prune."
                )
                
        return should_prune, reason
    
    
    def _format_checkpoint_name(self, filename: Optional[str] = None, ver: Optional[int] = None) -> str:
        '''
        Method to assign the name to the checkpoint based on the pruning status 
        and the version in case of duplicates
        '''
        filename = filename or self.filename
        
        # add pruning percentage
        if self.pruning_step:
            filename = f"{filename}-{self.ratios[self.pruning_step-1]}"
        # add version if duplicate
        if ver is not None:
            filename = f"{filename}-{ver}"
        # add file extension
        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name
    
        
    def file_exists(self, filepath: _PATH, trainer: "pl.Trainer") -> bool:
        '''
        Checks if a file exists on rank 0 and broadcasts the result to all other ranks, preventing the internal
        state to diverge between ranks.
        '''
        exists = self._fs.exists(filepath)
        return trainer.strategy.broadcast(exists)
    
    
    # ---------------------------------------------------------------------------- #
    #                               Callback methods                               #
    # ---------------------------------------------------------------------------- #
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor, 
            mode=self.mode,
        )
    
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "pruned_epoch": self.pruned_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
        }


    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.pruned_epoch = state_dict["pruned_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        if not self._check_on_train_epoch_end:
            return
        self._run_pruning_check(trainer, pl_module)
    
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        if self._check_on_train_epoch_end:
            return
        self._run_pruning_check(trainer, pl_module)
    
    
    
    