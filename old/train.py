import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.model import resnet50


# -----------------------------
# Training (AMP-ready)
# -----------------------------
def train_epoch_amp(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,        # set to torch.bfloat16 for BF16, or torch.float32 to disable AMP
    grad_clip_max_norm: float | None = None,       # e.g., 1.0 to enable clipping
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
# Main
# -----------------------------
def main(args):
    # Speed/throughput toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        # PyTorch 2.x matmul precision API (safe if unavailable)
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders (provided by your data.py)
    from data import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.workers
        # Consider enabling pin_memory/persistent_workers inside data.py
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

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
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
            }, args.save_path)
            print(f"✓ Model saved! Best Top-1: {best_acc:.2f}% | Best Top-5: {best_top5_acc:.2f}%")

    print(f"\n{'=' * 50}")
    print(f"Training complete.")
    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")
    print(f"Best Top-5 Accuracy: {best_top5_acc:.2f}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet-50 on ImageNet (AMP-enabled)')
    parser.add_argument('--data-path', default='./imagenet', help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--num-classes', type=int, default=1000, help='number of classes (1000 for ImageNet)')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--save-path', default='resnet50_best.pth', help='path to save best model checkpoint')
    parser.add_argument('--scheduler', default='multistep', choices=['cosine', 'multistep'],
                        help='learning rate scheduler type')

    # New CLI flags
    parser.add_argument('--amp', default='fp16', choices=['fp16', 'bf16', 'off'],
                        help='mixed precision mode: fp16, bf16, or off (fp32)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='max norm for gradient clipping (set <=0 to disable)')

    args = parser.parse_args()
    if args.grad_clip is not None and args.grad_clip <= 0:
        args.grad_clip = None
    main(args)
