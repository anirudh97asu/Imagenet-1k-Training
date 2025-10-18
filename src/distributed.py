"""
Distributed training utilities and helpers
"""
import os
import torch
import torch.distributed as dist
from typing import Optional


def is_distributed() -> bool:
    """Check if distributed training is enabled"""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def get_rank() -> int:
    """Get process rank"""
    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:
    """Get total number of processes"""
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """Get local process rank on current node"""
    return int(os.environ.get('LOCAL_RANK', 0))


def init_distributed(backend: str = 'nccl', init_method: str = 'env://'):
    """
    Initialize distributed training
    
    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU
        init_method: Initialization method
    """
    if not is_distributed():
        print("Not using distributed training")
        return False
    
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        print(f"Distributed training initialized:")
        print(f"  Rank: {rank}/{world_size}")
        print(f"  Backend: {backend}")
        print(f"  Local rank: {local_rank}")
    
    return True


def cleanup_distributed():
    """Cleanup distributed training"""
    if is_distributed():
        dist.destroy_process_group()


def reduce_dict(d: dict) -> dict:
    """
    Average metrics across all processes
    
    Args:
        d: Dictionary with metrics
    
    Returns:
        Averaged dictionary
    """
    if not is_distributed():
        return d
    
    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            torch.distributed.all_reduce(v)
            v = v / get_world_size()
        new_d[k] = v
    return new_d


def is_main_process() -> bool:
    """Check if current process is main process"""
    return get_rank() == 0


def main_process_only(func):
    """Decorator to run function only on main process"""
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def barrier():
    """Synchronize all processes"""
    if is_distributed():
        dist.barrier()


class DistributedMetricsCollector:
    """Collect and average metrics across distributed processes"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v
            self.metrics[k].append(v)
    
    def get_averages(self) -> dict:
        """Get averaged metrics across all processes"""
        averages = {}
        for k, v in self.metrics.items():
            avg = sum(v) / len(v) if len(v) > 0 else 0
            
            if is_distributed():
                avg_tensor = torch.tensor(avg, device=torch.cuda.current_device())
                dist.all_reduce(avg_tensor)
                avg = (avg_tensor / get_world_size()).item()
            
            averages[k] = avg
        return averages
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}