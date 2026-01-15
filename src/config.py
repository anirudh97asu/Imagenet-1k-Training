"""
Centralized configuration for all training components
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_path: str = './imagenet'
    s3_bucket: Optional[str] = None  # e.g., 's3://my-bucket/imagenet'
    batch_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 2
    image_size: int = 224
    num_classes: int = 1000


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = 'resnet50'
    num_classes: int = 1000
    pretrained: bool = False
    use_channels_last: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer: str = 'sgd'  # 'sgd' or 'adamw'
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    exclude_bn_bias_decay: bool = True


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    scheduler: str = 'onecycle'  # 'onecycle', 'multistep', or 'cosine'
    max_lr: float = 0.1
    total_steps: Optional[int] = None  # Set during training
    pct_start: float = 0.3
    anneal_strategy: str = 'cos'
    # For multistep
    milestones: List[int] = field(default_factory=lambda: [30, 60, 80])
    gamma: float = 0.1
    # For cosine
    t_max: Optional[int] = None


@dataclass
class AmpConfig:
    """Automatic Mixed Precision configuration"""
    enabled: bool = True
    amp_dtype: str = 'fp16'  # 'fp16', 'bf16', or 'off'
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class ProgressiveResizeConfig:
    """Progressive resizing configuration"""
    enabled: bool = True
    schedule: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 128, 512),    # (epoch, img_size, batch_size)
        (5, 160, 384),
        (10, 192, 320),
        (15, 224, 256),
    ])


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    enabled: bool = False
    backend: str = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    init_method: str = 'env://'


@dataclass
class TrainingConfig:
    """Main training configuration"""
    epochs: int = 90
    warmup_epochs: int = 5
    save_path: str = 'resnet50_best.pth'
    log_interval: int = 100
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: AmpConfig = field(default_factory=AmpConfig)
    progressive_resize: ProgressiveResizeConfig = field(default_factory=ProgressiveResizeConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def get_default_config() -> TrainingConfig:
    """Returns default training configuration"""
    return TrainingConfig()