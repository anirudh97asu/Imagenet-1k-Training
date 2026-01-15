import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size=224, is_training=True):
    """
    Get data augmentation transforms for given image size
    
    Args:
        image_size: Target image resolution (e.g., 128, 160, 192, 224)
        is_training: Whether this is for training or validation
    
    Returns:
        transforms.Compose object
    """
    if is_training:
        # Training transforms with data augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # Validation transforms (no augmentation)
        # Use slightly larger crop for center crop (standard practice)
        resize_size = int(image_size * 256 / 224)  # Scale proportionally
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_dataloaders(
    data_path,
    batch_size=256,
    num_workers=8,
    image_size=224,
    pin_memory=True,
    persistent_workers=True
):
    """
    Create train and validation dataloaders for ImageNet
    
    Args:
        data_path: Path to ImageNet dataset (should contain 'train' and 'val' subdirectories)
        batch_size: Batch size for both train and validation
        num_workers: Number of data loading workers
        image_size: Target image resolution (supports progressive resizing)
        pin_memory: Whether to use pinned memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs (faster but more memory)
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Paths
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    
    # Verify paths exist
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(image_size=image_size, is_training=True)
    )
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_transforms(image_size=image_size, is_training=False)
    )
    
    print(f"Dataset loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"Image size: {image_size}x{image_size}, Batch size: {batch_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader


# Additional utility for on-the-fly dataloader recreation
class ProgressiveDataLoaderManager:
    """
    Helper class to manage dataloader recreation for progressive resizing
    Can be used as an alternative to recreating loaders in the training loop
    """
    def __init__(self, data_path, num_workers=8, pin_memory=True):
        self.data_path = data_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.current_train_loader = None
        self.current_val_loader = None
        self.current_size = None
        self.current_batch_size = None
    
    def get_loaders(self, image_size, batch_size):
        """
        Get dataloaders for specified configuration.
        Only recreates if configuration changed.
        """
        if (self.current_size != image_size or 
            self.current_batch_size != batch_size):
            
            print(f"\nCreating new dataloaders: {image_size}x{image_size}, batch_size={batch_size}")
            
            self.current_train_loader, self.current_val_loader = get_dataloaders(
                data_path=self.data_path,
                batch_size=batch_size,
                num_workers=self.num_workers,
                image_size=image_size,
                pin_memory=self.pin_memory,
                persistent_workers=True
            )
            
            self.current_size = image_size
            self.current_batch_size = batch_size
            
            # Clear CUDA cache after recreating loaders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.current_train_loader, self.current_val_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing dataloader with different image sizes...")
    
    # Test with standard ImageNet size
    train_loader, val_loader = get_dataloaders(
        data_path='./imagenet',
        batch_size=256,
        num_workers=4,
        image_size=224
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test with progressive resizing sizes
    for size in [128, 160, 192, 224]:
        print(f"\n--- Testing with {size}x{size} images ---")
        train_loader, val_loader = get_dataloaders(
            data_path='./imagenet',
            batch_size=256,
            num_workers=4,
            image_size=size
        )
        
        # Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")