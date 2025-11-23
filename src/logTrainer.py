"""
Custom Trainer with loss logging callback support

Adapted from LoRA-GA: https://github.com/Outsider565/LoRA-GA
"""

from typing import Callable, Dict, Optional, Union, Any
import torch
import torch.nn as nn
from transformers import Trainer


class LogTrainer(Trainer):
    """
    Extended Trainer with callback support for logging training loss.
    
    This trainer adds a log_loss_callback parameter that is called
    after each training step with the current loss and step number.
    """
    
    def __init__(
        self,
        log_loss_callback: Optional[Callable[[float, int], None]] = None,
        **kwargs
    ):
        """
        Args:
            log_loss_callback: Optional callback function(loss, step)
                               called after each training step
            **kwargs: All other arguments passed to base Trainer
        """
        super().__init__(**kwargs)
        self.log_loss_callback = log_loss_callback
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step with optional loss logging.
        
        Args:
            model: The model being trained
            inputs: Dictionary of model inputs
            num_items_in_batch: Number of items in the batch (for compatibility with newer transformers)
            
        Returns:
            Training loss (detached)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Handle multi-GPU
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Call loss logging callback
        if self.log_loss_callback is not None:
            self.log_loss_callback(loss.item(), self.state.global_step)
        
        return loss.detach() / self.args.gradient_accumulation_steps
