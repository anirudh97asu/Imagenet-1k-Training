import os
import argparse
import time
import shutil
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler

# If your file is named `data_with_aug.py`, change this import.
from data_with_aug import (
    get_data_loaders,
    AugmentationConfig,
    apply_augmentations,
    mixup_criterion,
)
from model import resnet50


# ============================================================
# Distributed helpers
# ============================================================

class DistributedBatchSampler(Sampler):
    """
    Wrap an existing batch_sampler and split its batches across ranks.

    Assumes that the underlying batch_sampler produces the SAME sequence
    of batches on every rank (we enforce this via manual seeding).
    """

    def __init__(self, batch_sampler, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank

        self.total_batches = len(self.batch_sampler)
        # Each rank gets approximately total_batches / num_replicas batches
        self.num_batches_per_rank = (self.total_batches + self.num_replicas - 1) // self.num_replicas

    def __iter__(self):
        # We rely on *identical* underlying batch_sampler order on each rank.
        for batch_idx, batch in enumerate(self.batch_sampler):
            # Each rank picks every num_replicas-th batch
            if batch_idx % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        return self.num_batches_per_rank


def setup_logger(args, rank):
    """
    Logger per rank:
      - rank 0 logs to file + stdout
      - other ranks log to file only
    """
    run_dir = os.path.join(args.output, args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if this is called more than once
    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"train_rank_{rank}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy for specified values of k."""
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k = min(k, maxk)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def save_checkpoint(state, is_best, args, filename="checkpoint.pth"):
    run_dir = os.path.join(args.output, args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, "model_best.pth")
        shutil.copyfile(ckpt_path, best_path)


# ============================================================
# Train / Val loops (DDP-aware)
# ============================================================

def train_one_epoch(
    rank,
    world_size,
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler,
    aug_cfg,
    epoch,
    logger,
    print_freq=50,
    seed=42,
):
    # IMPORTANT: ensure CPU torch RNG is the same across ranks
    # so AspectRatioBatchSampler shuffling is identical.
    torch.manual_seed(seed + epoch)

    model.train()
    start_time = time.time()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    num_batches = len(train_loader)

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Batch-level augmentations (MixUp / CutMix / Cutout / Random Erasing)
        images, targets_a, targets_b, lam = apply_augmentations(images, targets, aug_cfg)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            if lam is not None and targets_a is not None and targets_b is not None:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                effective_targets = targets_a  # for accuracy approx
            else:
                loss = criterion(outputs, targets)
                effective_targets = targets

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        top1, top5 = accuracy(outputs, effective_targets, topk=(1, 5))
        running_top1 += top1 * batch_size
        running_top5 += top5 * batch_size

        if rank == 0 and ((i + 1) % print_freq == 0 or i == 0 or i == num_batches - 1):
            avg_loss = running_loss / total_samples
            avg_top1 = running_top1 / total_samples
            avg_top5 = running_top5 / total_samples
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] "
                f"[{i+1:4d}/{num_batches:4d}]  "
                f"Loss: {avg_loss:.4f}  "
                f"Acc@1: {avg_top1:.2f}%  "
                f"Acc@5: {avg_top5:.2f}%  "
                f"LR: {current_lr:.5f}"
            )

    # --------------------------------------------------------
    # Reduce metrics across ranks
    # --------------------------------------------------------
    loss_tensor = torch.tensor(running_loss, device=device)
    top1_tensor = torch.tensor(running_top1, device=device)
    top5_tensor = torch.tensor(running_top5, device=device)
    total_tensor = torch.tensor(total_samples, device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(top1_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    global_loss = loss_tensor.item() / total_tensor.item()
    global_top1 = top1_tensor.item() / total_tensor.item()
    global_top5 = top5_tensor.item() / total_tensor.item()
    epoch_time = time.time() - start_time

    if rank == 0:
        logger.info(
            f"Epoch [{epoch}] TRAIN  "
            f"Loss: {global_loss:.4f}  "
            f"Acc@1: {global_top1:.2f}%  "
            f"Acc@5: {global_top5:.2f}%  "
            f"Time: {epoch_time:.1f}s"
        )

    return global_loss, global_top1, global_top5


def validate(
    rank,
    world_size,
    model,
    val_loader,
    criterion,
    device,
    epoch,
    logger,
    print_freq=50,
):
    # For validation, we can keep deterministic order but it's not critical
    model.eval()
    start_time = time.time()

    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    num_batches = len(val_loader)

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            running_top1 += top1 * batch_size
            running_top5 += top5 * batch_size

            if rank == 0 and ((i + 1) % print_freq == 0 or i == 0 or i == num_batches - 1):
                avg_loss = running_loss / total_samples
                avg_top1 = running_top1 / total_samples
                avg_top5 = running_top5 / total_samples
                logger.info(
                    f"Epoch [{epoch}] "
                    f"[{i+1:4d}/{num_batches:4d}]  "
                    f"ValLoss: {avg_loss:.4f}  "
                    f"ValAcc@1: {avg_top1:.2f}%  "
                    f"ValAcc@5: {avg_top5:.2f}%"
                )

    # Reduce metrics across ranks
    loss_tensor = torch.tensor(running_loss, device=device)
    top1_tensor = torch.tensor(running_top1, device=device)
    top5_tensor = torch.tensor(running_top5, device=device)
    total_tensor = torch.tensor(total_samples, device=device)

    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(top1_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    global_loss = loss_tensor.item() / total_tensor.item()
    global_top1 = top1_tensor.item() / total_tensor.item()
    global_top5 = top5_tensor.item() / total_tensor.item()
    epoch_time = time.time() - start_time

    if rank == 0:
        logger.info(
            f"Epoch [{epoch}] VAL    "
            f"Loss: {global_loss:.4f}  "
            f"Acc@1: {global_top1:.2f}%  "
            f"Acc@5: {global_top5:.2f}%  "
            f"Time: {epoch_time:.1f}s"
        )

    return global_loss, global_top1, global_top5


# ============================================================
# Main worker (per process)
# ============================================================

def main_worker(rank, world_size, args):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", args.dist_port)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # Create run dirs from rank 0, then sync
    if rank == 0:
        run_dir = os.path.join(args.output, args.run_name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    dist.barrier()

    logger = setup_logger(args, rank)
    logger.info(f"Rank {rank} initialized. World size = {world_size}, backend = {backend}")

    # --------------------------------------------------------
    # Data loaders (rectangular crops + minimal transforms)
    # --------------------------------------------------------
    # We first build the "regular" loaders, then wrap their batch_samplers
    # with DistributedBatchSampler so that batches are partitioned by rank.
    train_loader_base, val_loader_base = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        target_size=args.target_size,
    )

    train_dist_bs = DistributedBatchSampler(
        train_loader_base.batch_sampler,
        num_replicas=world_size,
        rank=rank,
    )
    val_dist_bs = DistributedBatchSampler(
        val_loader_base.batch_sampler,
        num_replicas=world_size,
        rank=rank,
    )

    train_loader = DataLoader(
        train_loader_base.dataset,
        batch_sampler=train_dist_bs,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_loader_base.dataset,
        batch_sampler=val_dist_bs,
        num_workers=args.workers,
        pin_memory=True,
    )

    num_classes = len(train_loader.dataset.classes)
    if rank == 0:
        logger.info(f"Detected {num_classes} classes")

    # --------------------------------------------------------
    # Model, criterion, optimizer, scheduler, scaler
    # --------------------------------------------------------
    base_model = resnet50(num_classes=num_classes).to(device)
    model = DDP(base_model, device_ids=[rank] if device.type == "cuda" else None)

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        logger.info(f"Model: ResNet-50 (rectangular) | params: {n_params:,}")

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = GradScaler() if (torch.cuda.is_available() and not args.no_amp) else None

    # Batch-level augmentation configuration
    aug_cfg = AugmentationConfig()
    if rank == 0:
        logger.info("Batch-level augmentation config:")
        logger.info(f"  use_mixup:          {aug_cfg.use_mixup}  (alpha={aug_cfg.mixup_alpha})")
        logger.info(f"  use_cutmix:         {aug_cfg.use_cutmix} (alpha={aug_cfg.cutmix_alpha})")
        logger.info(f"  use_cutout:         {aug_cfg.use_cutout}")
        logger.info(f"  use_random_erasing: {aug_cfg.use_random_erasing}")
        logger.info(f"  batch_aug_prob:     {aug_cfg.batch_aug_prob}")

    start_epoch = 0
    best_acc1 = 0.0

    # --------------------------------------------------------
    # Optional: resume
    # --------------------------------------------------------
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info(f"Loading checkpoint from '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.module.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scheduler", None) is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler", None) is not None and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_acc1 = checkpoint.get("best_acc1", 0.0)
            if rank == 0:
                logger.info(f"Resumed from epoch {start_epoch}, best_acc1={best_acc1:.2f}")
        else:
            if rank == 0:
                logger.warning(f"No checkpoint found at '{args.resume}'")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    if rank == 0:
        logger.info("==== Starting training ====")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            rank=rank,
            world_size=world_size,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            aug_cfg=aug_cfg,
            epoch=epoch,
            logger=logger,
            print_freq=args.print_freq,
            seed=args.seed,
        )

        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            rank=rank,
            world_size=world_size,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
            print_freq=args.print_freq,
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoints from rank 0 only
        if rank == 0:
            is_best = val_acc1 > best_acc1
            best_acc1 = max(best_acc1, val_acc1)

            ckpt_state = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_acc1": best_acc1,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "val_loss": val_loss,
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
            }

            filename = f"checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(ckpt_state, is_best, args, filename=filename)

            logger.info(
                f"Epoch [{epoch}] finished. "
                f"Train Acc@1: {train_acc1:.2f}%  Val Acc@1: {val_acc1:.2f}%  "
                f"Best Acc@1: {best_acc1:.2f}%"
            )

    if rank == 0:
        logger.info("==== Training complete ====")

    dist.destroy_process_group()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ResNet-50 DDP training with rectangular crops + batch augs")
    parser.add_argument("--data_dir", default="/mnt/data/imagenet", type=str,
                        help="Path to ImageNet root directory (with 'train' and 'val')")
    parser.add_argument("--epochs", default=90, type=int,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Per-GPU batch size")
    parser.add_argument("--workers", default=8, type=int,
                        help="DataLoader workers per process")
    parser.add_argument("--target_size", default=224, type=int,
                        help="Target shorter side for RectangularCropTfm")
    parser.add_argument("--lr", default=0.1, type=float,
                        help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay")
    parser.add_argument("--output", default="./runs_rect_ddp", type=str,
                        help="Root directory for logs/checkpoints")
    parser.add_argument("--run_name", default="resnet50_rect", type=str,
                        help="Subfolder name under output")
    parser.add_argument("--resume", default="", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--print_freq", default=50, type=int,
                        help="Print/log frequency in batches")
    parser.add_argument("--seed", default=42, type=int,
                        help="Base seed for sampler shuffling")
    parser.add_argument("--dist_port", default="12355", type=str,
                        help="Master port for DDP")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This DDP script expects at least one CUDA device.")

    world_size = torch.cuda.device_count()
    print(f"World size (GPUs): {world_size}")

    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, args),
        join=True,
    )


if __name__ == "__main__":
    main()
