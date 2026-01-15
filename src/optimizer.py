"""
Optimizer and learning rate scheduler setup
"""
import torch
import torch.optim as optim
from config import OptimizerConfig, SchedulerConfig
from train_utils import configure_weight_decay


def create_optimizer(
    model,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional selective weight decay
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
    
    Returns:
        Optimizer instance
    """
    
    if config.exclude_bn_bias_decay:
        param_groups = configure_weight_decay(model, config.weight_decay)
    else:
        param_groups = model.parameters()
    
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay if not config.exclude_bn_bias_decay else None,
            nesterov=True
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay if not config.exclude_bn_bias_decay else None,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    epochs: int,
    steps_per_epoch: int,
):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
    
    Returns:
        Scheduler instance
    """
    
    if config.scheduler == 'onecycle':
        total_steps = epochs * steps_per_epoch
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            total_steps=total_steps,
            pct_start=config.pct_start,
            anneal_strategy=config.anneal_strategy,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            last_epoch=-1,
            div_factor=25.0,
            final_div_factor=10000.0,
            verbose=False
        )
        return scheduler, 'step'  # Step per iteration
    
    elif config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0
        )
        return scheduler, 'epoch'  # Step per epoch
    
    elif config.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma
        )
        return scheduler, 'epoch'  # Step per epoch
    
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


class SchedulerWrapper:
    """Wrapper to handle different scheduler update frequencies"""
    
    def __init__(self, scheduler, step_type: str):
        """
        Args:
            scheduler: PyTorch scheduler
            step_type: 'step' for per-iteration, 'epoch' for per-epoch
        """
        self.scheduler = scheduler
        self.step_type = step_type
    
    def step(self, is_epoch_end: bool = False):
        """
        Step the scheduler
        
        Args:
            is_epoch_end: Whether this is called at epoch end
        """
        if self.step_type == 'step':
            # Called every iteration
            if not is_epoch_end:
                self.scheduler.step()
        else:
            # Called every epoch
            if is_epoch_end:
                self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]