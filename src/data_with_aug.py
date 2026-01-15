"""ImageNet loader with rectangular cropping - Minimal per-image transforms"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from PIL import Image
import random

# ============================================================================
# ESSENTIAL TRANSFORM ONLY (Rectangular Crop)
# ============================================================================

class RectangularCropTfm(object):
    """Transform that crops images based on aspect ratio"""
    def __init__(self, idx2ar, target_size=224):
        self.idx2ar = idx2ar
        self.target_size = target_size
    
    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:  # Portrait
            w = int(self.target_size / target_ar)
            size = (w // 8 * 8, self.target_size)
        else:  # Landscape
            h = int(self.target_size * target_ar)
            size = (self.target_size, h // 8 * 8)
        return transforms.functional.resize(img, size)

# ============================================================================
# BATCH-LEVEL AUGMENTATIONS (Applied in Training Loop)
# ============================================================================

class MixUp(object):
    """MixUp augmentation - BATCH LEVEL"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class CutMix(object):
    """CutMix augmentation - BATCH LEVEL"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        _, _, H, W = batch_images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_images = batch_images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        labels_a, labels_b = batch_labels, batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class Cutout(object):
    """Cutout augmentation - BATCH LEVEL"""
    def __init__(self, n_holes=1, length=56, p=0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p
    
    def __call__(self, batch_images):
        """
        Args:
            batch_images: Tensor [B, C, H, W]
        Returns:
            Tensor with cutout applied
        """
        if random.random() > self.p:
            return batch_images
        
        B, C, H, W = batch_images.shape
        mask = torch.ones((B, H, W), dtype=torch.float32, device=batch_images.device)
        
        for b in range(B):
            for _ in range(self.n_holes):
                y = random.randint(0, H)
                x = random.randint(0, W)
                
                y1 = max(0, y - self.length // 2)
                y2 = min(H, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(W, x + self.length // 2)
                
                mask[b, y1:y2, x1:x2] = 0.
        
        mask = mask.unsqueeze(1).expand_as(batch_images)
        batch_images = batch_images * mask
        
        return batch_images

class RandomErasing(object):
    """Random Erasing augmentation - BATCH LEVEL"""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, batch_images):
        """
        Args:
            batch_images: Tensor [B, C, H, W]
        """
        if random.random() > self.p:
            return batch_images
        
        B, C, H, W = batch_images.shape
        area = H * W
        
        for b in range(B):
            for _ in range(10):
                target_area = random.uniform(self.scale[0], self.scale[1]) * area
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
                
                h = int(round((target_area * aspect_ratio) ** 0.5))
                w = int(round((target_area / aspect_ratio) ** 0.5))
                
                if w < W and h < H:
                    x1 = random.randint(0, W - w)
                    y1 = random.randint(0, H - h)
                    
                    if self.value == 'random':
                        batch_images[b, :, y1:y1+h, x1:x1+w] = torch.randn(C, h, w, device=batch_images.device)
                    else:
                        batch_images[b, :, y1:y1+h, x1:x1+w] = self.value
                    break
        
        return batch_images

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sort_ar(datadir, split='train'):
    """Sort images by aspect ratio and cache"""
    idx2ar_file = os.path.join(datadir, f'sorted_idxar_{split}.pkl')
    
    if os.path.isfile(idx2ar_file):
        print(f"Loading cached aspect ratios from {idx2ar_file}")
        return pickle.load(open(idx2ar_file, 'rb'))
    
    print(f'Sorting {split} images by Aspect Ratio...')
    split_dir = os.path.join(datadir, split)
    dataset = datasets.ImageFolder(split_dir)
    sizes = [img[0].size for img in tqdm(dataset, total=len(dataset))]
    idx_ar = [(i, round(s[0]/s[1], 5)) for i, s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    
    return sorted_idxar

def chunks(l, n):
    """Split list into chunks"""
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def map_idx2ar(idx_ar_sorted, batch_size):
    """Map image index to batch mean aspect ratio"""
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
    
    return idx2ar

# ============================================================================
# DATASET
# ============================================================================

class RectDataset(datasets.ImageFolder):
    """ImageFolder with rectangular cropping only"""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, RectangularCropTfm):
                    sample = tfm(sample, index)
                else:
                    sample = tfm(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

class AspectRatioBatchSampler(torch.utils.data.Sampler):
    """Sampler that groups images by aspect ratio"""
    def __init__(self, sorted_idxar, batch_size, drop_last=True, shuffle=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batches = []
        
        for i in range(0, len(sorted_idxar), batch_size):
            chunk = sorted_idxar[i:i+batch_size]
            if len(chunk) == batch_size or not drop_last:
                self.batches.append([idx for idx, ar in chunk])
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.batches)).tolist()
        else:
            indices = list(range(len(self.batches)))
        
        for idx in indices:
            batch = self.batches[idx]
            if self.shuffle:
                batch = [batch[i] for i in torch.randperm(len(batch)).tolist()]
            yield batch
    
    def __len__(self):
        return len(self.batches)

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================

class AugmentationConfig:
    """Configuration for batch-level augmentations only"""
    def __init__(self):
        # ALL augmentations are now batch-level (in training loop)
        self.use_mixup = True
        self.mixup_alpha = 0.2
        
        self.use_cutmix = True
        self.cutmix_alpha = 1.0
        
        self.use_cutout = False
        self.cutout_n_holes = 1
        self.cutout_length = 8
        self.cutout_prob = 0.5
        
        self.use_random_erasing = True
        self.random_erasing_prob = 0.25
        self.random_erasing_scale = (0.02, 0.33)
        
        # Probability of applying batch augmentation
        self.batch_aug_prob = 0.5

def get_transforms(idx2ar, target_size=224):
    """
    Build MINIMAL transform pipeline - only essential operations
    
    Args:
        idx2ar: Index to aspect ratio mapping
        target_size: Target size
    
    Returns:
        List of transforms (minimal - just crop, convert, normalize)
    """
    return [
        RectangularCropTfm(idx2ar, target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

# ============================================================================
# MAIN DATA LOADER FUNCTION
# ============================================================================

def get_data_loaders(data_dir='/mnt/data/imagenet', batch_size=256, 
                     num_workers=8, target_size=224):
    """
    Create train and val loaders with MINIMAL per-image transforms
    All augmentations are batch-level (applied in training loop)
    
    Returns:
        train_loader, val_loader
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Sort by aspect ratio
    print("Processing train set...")
    train_sorted_idxar = sort_ar(data_dir, split='train')
    print("Processing val set...")
    val_sorted_idxar = sort_ar(data_dir, split='val')
    
    # Map indices
    train_idx2ar = map_idx2ar(train_sorted_idxar, batch_size)
    val_idx2ar = map_idx2ar(val_sorted_idxar, batch_size)
    
    # Get MINIMAL transforms (only crop + normalize)
    train_transform = get_transforms(train_idx2ar, target_size)
    val_transform = get_transforms(val_idx2ar, target_size)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RectDataset(train_dir, train_transform)
    val_dataset = RectDataset(val_dir, val_transform)
    
    # Create samplers
    train_sampler = AspectRatioBatchSampler(
        train_sorted_idxar, batch_size, drop_last=True, shuffle=True
    )
    val_sampler = AspectRatioBatchSampler(
        val_sorted_idxar, batch_size, drop_last=False, shuffle=False
    )
    
    # Create loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"{'='*60}")
    print(f"DataLoader Transforms (MINIMAL):")
    print(f"  ✓ Rectangular Crop")
    print(f"  ✓ ToTensor")
    print(f"  ✓ Normalize")
    print(f"\nAll augmentations are BATCH-LEVEL (apply in training loop):")
    print(f"  - MixUp, CutMix, Cutout, Random Erasing")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader

# ============================================================================
# BATCH AUGMENTATION HELPERS (for Training Loop)
# ============================================================================

def apply_augmentations(images, labels, config):
    """
    Apply ALL augmentations at batch level in training loop
    
    Args:
        images: Batch [B, C, H, W]
        labels: Labels [B]
        config: AugmentationConfig
    
    Returns:
        images, labels_a, labels_b, lam (or images, labels, None, None if no mixing augmentation)
    """
    # Step 1: Apply Cutout or Random Erasing (if enabled)
    if config.use_cutout:
        cutout = Cutout(
            n_holes=config.cutout_n_holes,
            length=config.cutout_length,
            p=config.cutout_prob
        )
        images = cutout(images)
    
    if config.use_random_erasing:
        random_erasing = RandomErasing(
            p=config.random_erasing_prob,
            scale=config.random_erasing_scale
        )
        images = random_erasing(images)
    
    # Step 2: Apply MixUp or CutMix (if enabled)
    if not config.use_mixup and not config.use_cutmix:
        return images, labels, None, None
    
    if random.random() > config.batch_aug_prob:
        return images, labels, None, None
    
    # Choose MixUp or CutMix
    if config.use_mixup and config.use_cutmix:
        use_mixup = random.random() < 0.5
    elif config.use_mixup:
        use_mixup = True
    else:
        use_mixup = False
    
    if use_mixup:
        mixup = MixUp(alpha=config.mixup_alpha)
        return mixup(images, labels)
    else:
        cutmix = CutMix(alpha=config.cutmix_alpha)
        return cutmix(images, labels)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for MixUp/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# EXAMPLE TRAINING LOOP
# ============================================================================

def example_training_loop():
    """
    Example showing complete training loop with batch augmentations
    """
    # 1. Setup DataLoader
    train_loader, val_loader = get_data_loaders(
        data_dir='/mnt/data/imagenet',
        batch_size=256
    )
    
    # 2. Setup augmentation config
    config = AugmentationConfig()
    config.use_mixup = True
    config.use_cutmix = True
    config.use_random_erasing = True
    config.use_cutout = False  # Usually choose one: cutout OR random erasing
    
    # 3. Training loop
    model = None  # Your model here
    optimizer = None  # Your optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move to GPU
            images = images.cuda()
            labels = labels.cuda()
            
            # Apply ALL augmentations here (batch-level)
            images, labels_a, labels_b, lam = apply_augmentations(
                images, labels, config
            )
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            if lam is not None:
                # MixUp/CutMix loss
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                # Standard loss
                loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SIMPLIFIED AUGMENTATION ARCHITECTURE")
    print("=" * 60)
    print("\n📦 DataLoader (MINIMAL transforms):")
    print("   - Rectangular Crop")
    print("   - ToTensor")
    print("   - Normalize")
    print("   → NO augmentations here!\n")
    
    print("🔀 Training Loop (ALL augmentations):")
    print("   - Cutout / Random Erasing")
    print("   - MixUp / CutMix")
    print("   → All augmentations applied to batches\n")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        data_dir='/mnt/data/imagenet',
        batch_size=64,
        num_workers=4
    )
    
    # Test
    print("\nTesting...")
    images, labels = next(iter(train_loader))
    print(f"Loaded batch (no augmentations): {images.shape}")
    
    # Setup config
    config = AugmentationConfig()
    config.use_mixup = True
    config.use_cutmix = True
    config.use_random_erasing = True
    
    # Apply augmentations
    images_aug, labels_a, labels_b, lam = apply_augmentations(
        images, labels, config
    )
    
    print(f"After augmentations: {images_aug.shape}")
    if lam is not None:
        print(f"Lambda: {lam:.3f}")
        print(f"Labels A: {labels_a[:5]}")
        print(f"Labels B: {labels_b[:5]}")
    
    print("\n✓ Ready to train!")
    print("\nIn your training loop, call:")
    print("  images, labels_a, labels_b, lam = apply_augmentations(images, labels, config)")