import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.model import resnet50


# -----------------------------
# Progressive Resizing Schedule
# -----------------------------
class ProgressiveResizeSchedule:
    """Manages progressive image size and batch size changes during training"""
    def __init__(self, schedule=None):
        """
        Args:
            schedule: List of (epoch, img_size, batch_size) tuples
                     If None, uses default progressive schedule
        """
        if schedule is None:
            # Default schedule: start small, gradually increase
            self.schedule = [
                (0, 128, 512),    # Small images, large batches
                (5, 160, 384),    # Medium-small images
                (10, 192, 320),   # Medium-large images
                (15, 224, 256),   # Full size ImageNet resolution
            ]
        else:
            self.schedule = schedule
        
        # Sort by epoch to ensure correct ordering
        self.schedule.sort(key=lambda x: x[0])
    
    def get_config(self, epoch):
        """Get (img_size, batch_size) for current epoch"""
        current_config = self.schedule[0]
        for epoch_threshold, img_size, batch_size in self.schedule:
            if epoch >= epoch_threshold:
                current_config = (epoch_threshold, img_size, batch_size)
            else:
                break
        return current_config[1], current_config[2]  # img_size, batch_size
    
    def should_update_dataloader(self, epoch):
        """Check if we need to update the dataloader at this epoch"""
        for epoch_threshold, _, _ in self.schedule:
            if epoch == epoch_threshold:
                return True
        return False


# -----------------------------
# Weight Decay Configuration
# -----------------------------
def configure_weight_decay(model, weight_decay=1e-4):
    """
    Remove weight decay from batch normalization and bias terms.
    This optimization from the Tencent paper helps reduce training time.
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for batch norm and bias terms
        if 'bn' in name.lower() or 'bias' in name or isinstance(param, (nn.BatchNorm2d,)):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


# -----------------------------
# Training (AMP-ready with Progressive Resizing)
# -----------------------------
def train_epoch_amp(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip_max_norm: float | None = None,
    channels_last: bool = True
):
    model.train()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    steps = 0

    # GradScaler is only needed for FP16 (BF16/FP32 do not use loss scaling)
    need_scaler = (amp_dtype == torch.float16)
    if scaler is None:
        scaler = GradScaler(enabled=need_scaler)

    pbar = tqdm(loader, desc='Training')
    for inputs, targets in pbar:
        # Device + memory format
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss in mixed precision (disabled if amp_dtype == float32)
        with autocast(enabled=(amp_dtype != torch.float32),
                      dtype=(amp_dtype if amp_dtype != torch.float32 else torch.float16)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward + step with/without scaler
        if need_scaler:
            scaler.scale(loss).backward()

            if grad_clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            optimizer.step()

        # ---- Metrics (compute in FP32 for safety) ----
        running_loss += loss.item()
        steps += 1

        with torch.no_grad():
            logits = outputs.detach().float()
            predicted = logits.argmax(1)
            correct += (predicted == targets).sum().item()

            _, pred_top5 = logits.topk(5, 1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

        pbar.set_postfix({
            'loss': running_loss / steps,
            'top1': 100.0 * correct / total,
            'top5': 100.0 * correct_top5 / total
        })

    avg_loss = running_loss / steps
    top1_acc = 100.0 * correct / total
    top5_acc = 100.0 * correct_top5 / total
    return avg_loss, top1_acc, top5_acc


# -----------------------------
# Validation (AMP-ready)
# -----------------------------
def validate(model, loader, criterion, device, amp_dtype: torch.dtype = torch.float16, channels_last: bool = True):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation'):
            inputs = inputs.to(device, non_blocking=True)
            if channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=(amp_dtype != torch.float32),
                          dtype=(amp_dtype if amp_dtype != torch.float32 else torch.float16)):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()

            logits = outputs.float()
            predicted = logits.argmax(1)
            correct += (predicted == targets).sum().item()

            _, pred_top5 = logits.topk(5, 1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

    top1_acc = 100.0 * correct / total
    top5_acc = 100.0 * correct_top5 / total
    avg_loss = running_loss / len(loader)
    return avg_loss, top1_acc, top5_acc


# -----------------------------
# Main (with Progressive Resizing)
# -----------------------------
def main(args):
    # Speed/throughput toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Progressive resizing schedule
    progressive_schedule = ProgressiveResizeSchedule(args.progressive_schedule)
    
    # Get initial configuration
    initial_img_size, initial_batch_size = progressive_schedule.get_config(0)
    
    # Data loaders (will be recreated when image size changes)
    from data import get_dataloaders
    
    print(f"\n{'='*60}")
    print(f"Progressive Training Configuration:")
    print(f"{'='*60}")
    for epoch_start, img_size, batch_size in progressive_schedule.schedule:
        print(f"  Epoch {epoch_start:3d}+: {img_size}x{img_size} images, batch size {batch_size}")
    print(f"{'='*60}\n")
    
    # Initialize with first configuration
    train_loader, val_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=initial_batch_size,
        num_workers=args.workers,
        image_size=initial_img_size if args.progressive_resize else 224
    )

    # AMP mode selection
    amp_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'off': torch.float32}
    amp_dtype = amp_map[args.amp]
    use_scaler = (args.amp == 'fp16')
    scaler = GradScaler(enabled=use_scaler)

    # Model
    model = resnet50(num_classes=args.num_classes).to(device).to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer with selective weight decay (remove from BatchNorm)
    if args.no_bn_weight_decay:
        print("Using selective weight decay (excluding BatchNorm layers)")
        param_groups = configure_weight_decay(model, weight_decay=args.weight_decay)
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

    # Learning rate schedule
    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 80], gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    best_acc = 0.0
    best_top5_acc = 0.0
    current_img_size = initial_img_size
    current_batch_size = initial_batch_size

    # Training loop
    for epoch in range(args.epochs):
        # Check if we need to update dataloader for progressive resizing
        if args.progressive_resize and progressive_schedule.should_update_dataloader(epoch):
            new_img_size, new_batch_size = progressive_schedule.get_config(epoch)
            
            if new_img_size != current_img_size or new_batch_size != current_batch_size:
                print(f"\n{'='*60}")
                print(f"PROGRESSIVE RESIZE UPDATE at Epoch {epoch + 1}")
                print(f"  Image size: {current_img_size}x{current_img_size} → {new_img_size}x{new_img_size}")
                print(f"  Batch size: {current_batch_size} → {new_batch_size}")
                print(f"{'='*60}\n")
                
                # Recreate dataloaders with new configuration
                train_loader, val_loader = get_dataloaders(
                    data_path=args.data_path,
                    batch_size=new_batch_size,
                    num_workers=args.workers,
                    image_size=new_img_size
                )
                
                current_img_size = new_img_size
                current_batch_size = new_batch_size
                
                # Clear cache to free memory from old dataloader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        if args.progressive_resize:
            print(f"Image size: {current_img_size}x{current_img_size} | Batch size: {current_batch_size}")
        print(f"AMP: {args.amp} | Grad clip: {args.grad_clip}")

        train_loss, train_top1, train_top5 = train_epoch_amp(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler,
            amp_dtype=amp_dtype,
            grad_clip_max_norm=args.grad_clip,
            channels_last=True
        )

        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device,
            amp_dtype=amp_dtype,
            channels_last=True
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Top-1 Acc: {train_top1:.2f}% | Top-5 Acc: {train_top5:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Top-1 Acc: {val_top1:.2f}% | Top-5 Acc: {val_top5:.2f}%")

        # Save best model (based on top-1 accuracy)
        if val_top1 > best_acc:
            best_acc = val_top1
            best_top5_acc = val_top5
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_scaler else None,
                'amp_mode': args.amp,
                'val_top1_acc': val_top1,
                'val_top5_acc': val_top5,
                'progressive_schedule': progressive_schedule.schedule if args.progressive_resize else None,
            }, args.save_path)
            print(f"✓ Model saved! Best Top-1: {best_acc:.2f}% | Best Top-5: {best_top5_acc:.2f}%")

    print(f"\n{'=' * 50}")
    print(f"Training complete.")
    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")
    print(f"Best Top-5 Accuracy: {best_top5_acc:.2f}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet (AMP + Progressive Resizing)')
    parser.add_argument('--data-path', default='./imagenet', help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='initial batch size (overridden by progressive schedule if enabled)')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--num-classes', type=int, default=1000, help='number of classes (1000 for ImageNet)')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--save-path', default='resnet50_best.pth', help='path to save best model checkpoint')
    parser.add_argument('--scheduler', default='multistep', choices=['cosine', 'multistep'],
                        help='learning rate scheduler type')
    parser.add_argument('--amp', default='fp16', choices=['fp16', 'bf16', 'off'],
                        help='mixed precision mode: fp16, bf16, or off (fp32)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='max norm for gradient clipping (set <=0 to disable)')
    
    # Progressive resizing arguments
    parser.add_argument('--progressive-resize', action='store_true',
                        help='enable progressive image resizing during training')
    parser.add_argument('--progressive-schedule', type=str, default=None,
                        help='custom progressive schedule as: "epoch1,size1,batch1;epoch2,size2,batch2;..." '
                             'e.g., "0,128,512;5,160,384;10,192,320;15,224,256"')
    parser.add_argument('--no-bn-weight-decay', action='store_true',
                        help='remove weight decay from BatchNorm layers (Tencent optimization)')

    args = parser.parse_args()
    
    # Parse gradient clipping
    if args.grad_clip is not None and args.grad_clip <= 0:
        args.grad_clip = None
    
    # Parse custom progressive schedule if provided
    if args.progressive_schedule:
        schedule = []
        for stage in args.progressive_schedule.split(';'):
            epoch, size, batch = map(int, stage.split(','))
            schedule.append((epoch, size, batch))
        args.progressive_schedule = schedule
    else:
        args.progressive_schedule = None  # Will use default schedule
    
    main(args)