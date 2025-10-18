"""
Training utilities: AMP, progressive resizing, weight decay configuration
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from config import ProgressiveResizeConfig


def configure_weight_decay(model: nn.Module, weight_decay: float = 1e-4) -> List[dict]:
    """
    Configure weight decay excluding BatchNorm and bias parameters
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay coefficient
    
    Returns:
        List of parameter groups for optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for BatchNorm and bias
        if 'bn' in name.lower() or 'bias' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


class ProgressiveResizer:
    """Manages progressive image resizing schedule"""
    
    def __init__(self, config: ProgressiveResizeConfig):
        self.schedule = sorted(config.schedule, key=lambda x: x[0])
        self.enabled = config.enabled
    
    def get_config(self, epoch: int) -> Tuple[int, int]:
        """Get (image_size, batch_size) for given epoch"""
        if not self.enabled:
            return 224, 256
        
        current = self.schedule[0]
        for epoch_start, img_size, batch_size in self.schedule:
            if epoch >= epoch_start:
                current = (epoch_start, img_size, batch_size)
            else:
                break
        
        return current[1], current[2]
    
    def should_update(self, epoch: int) -> bool:
        """Check if dataloader should be updated at this epoch"""
        if not self.enabled:
            return False
        
        for epoch_start, _, _ in self.schedule:
            if epoch == epoch_start:
                return True
        return False
    
    def print_schedule(self):
        """Print the entire schedule"""
        print("\nProgressive Resizing Schedule:")
        print("=" * 50)
        for epoch_start, img_size, batch_size in self.schedule:
            print(f"  Epoch {epoch_start:3d}+: {img_size}x{img_size} images, batch {batch_size}")
        print("=" * 50)


class AmpScaler:
    """Wrapper for AMP gradient scaling"""
    
    def __init__(self, enabled: bool = True, dtype: str = 'fp16'):
        self.enabled = enabled and dtype == 'fp16'
        self.dtype = dtype
        
        if self.enabled:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = None
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: torch.Tensor):
        """Backward with scaling"""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def unscale(self, optimizer):
        """Unscale gradients before clipping"""
        if self.enabled:
            self.scaler.unscale_(optimizer)
    
    def step(self, optimizer):
        """Optimizer step with scaling"""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


def setup_model_for_training(model: nn.Module, channels_last: bool = True) -> nn.Module:
    """
    Setup model with optimizations
    
    Args:
        model: PyTorch model
        channels_last: Use channels-last memory format
    
    Returns:
        Optimized model
    """
    # Enable tf32 for faster training on modern GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass
    
    # Convert to channels last format
    if channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    
    # Compile model (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="max-autotune")
    except:
        pass
    
    return model


def get_amp_context(dtype: str = 'fp16'):
    """Get autocast context manager"""
    from torch.cuda.amp import autocast
    
    enabled = dtype != 'off'
    amp_dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'off': torch.float32
    }.get(dtype, torch.float16)
    
    return autocast(enabled=enabled, dtype=amp_dtype)


def init_weights(model: nn.Module):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)