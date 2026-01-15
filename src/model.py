"""ResNet-50 Model Implementation with Testing"""
import torch
import torch.nn as nn
from torchsummary import summary

# ============================================================================
# RESNET BUILDING BLOCKS
# ============================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ============================================================================
# RESNET ARCHITECTURE
# ============================================================================

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet50(num_classes=1000):
    """ResNet-50: 3 + 4 + 6 + 3 = 16 bottleneck blocks"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# ============================================================================
# MAIN TESTING
# ============================================================================

if __name__ == "__main__":
    # Import the data loader
    try:
        from data_with_aug import get_data_loaders, apply_augmentations, AugmentationConfig, mixup_criterion
        has_data_loader = True
    except ImportError:
        print("Warning: data_with_aug.py not found. Skipping data loader test.")
        has_data_loader = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build model
    model = resnet50(num_classes=1000).to(device)
    model.eval()
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    summary(model, (3, 224, 224), device=device)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if has_data_loader:
        print("\n" + "="*70)
        print("TESTING FORWARD PASS WITH RECTANGULAR CROPS")
        print("="*70)
        
        # Get data loaders with smaller batch size for testing
        print("\nInitializing data loaders...")
        train_loader, val_loader = get_data_loaders(
            data_dir='/mnt/data/imagenet',
            batch_size=64,  # Smaller batch for testing
            num_workers=4,
            target_size=224
        )
        
        print("\n" + "-"*70)
        print("Testing TRAIN loader forward passes:")
        print("-"*70)
        
        # Setup config
        config = AugmentationConfig()
        config.use_mixup = True
        config.use_cutmix = True
        config.use_random_erasing = True
        
        # Test several batches from train loader
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Apply augmentations
                images_aug, labels_a, labels_b, lam = apply_augmentations(
                    images, labels, config
                )
                
                # Forward pass
                outputs = model(images_aug)
                
                print(f"Batch {i+1}:")
                print(f"  Input shape: {images_aug.shape}")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Labels shape: {labels_a.shape}")
                print(f"  Aspect ratio: {images_aug.shape[3] / images_aug.shape[2]:.3f}")
                
                if i >= 4:  # Test 5 batches
                    break
        
        print("\n" + "-"*70)
        print("Testing VAL loader forward passes:")
        print("-"*70)
        
        # Test several batches from val loader
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                print(f"Batch {i+1}:")
                print(f"  Input shape: {images.shape}")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Aspect ratio: {images.shape[3] / images.shape[2]:.3f}")
                
                if i >= 4:  # Test 5 batches
                    break
        
        print("\n" + "="*70)
        print("✓ All forward passes completed successfully!")
        print("✓ Model handles varying rectangular input sizes correctly")
        print("="*70)
    
    else:
        print("\n" + "="*70)
        print("TESTING FORWARD PASS WITH STANDARD SQUARE INPUTS")
        print("="*70)
        
        # Test with standard square inputs
        test_shapes = [
            (2, 3, 224, 224),  # Standard
            (2, 3, 224, 288),  # Landscape
            (2, 3, 288, 224),  # Portrait
        ]
        
        with torch.no_grad():
            for shape in test_shapes:
                x = torch.randn(shape).to(device)
                out = model(x)
                print(f"Input: {shape} -> Output: {out.shape}")
        
        print("\n✓ Forward pass test completed!")