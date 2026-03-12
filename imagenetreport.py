import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm import tqdm

# Import model architecture
from model import resnet50


# ============================================================================
# TRANSFORMS
# ============================================================================
class RectangularCropTfm:
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


def get_transforms(idx2ar, target_size=224):
    """Build minimal transform pipeline"""
    return [
        RectangularCropTfm(idx2ar, target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]


def sort_ar(datadir, split='val'):
    """Sort images by aspect ratio and cache"""
    idx2ar_file = os.path.join(datadir, f'sorted_idxar_{split}.pkl')
    if os.path.isfile(idx2ar_file):
        print(f"Loading cached aspect ratios from {idx2ar_file}")
        return pickle.load(open(idx2ar_file, 'rb'))
    
    print(f'Sorting {split} images by aspect ratio...')
    split_dir = os.path.join(datadir, split)
    dataset = datasets.ImageFolder(split_dir)
    sizes = [img[0].size for img in tqdm(dataset, total=len(dataset))]
    idx_ar = [(i, round(s[0] / s[1], 5)) for i, s in enumerate(sizes)]
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
    """ImageFolder with rectangular cropping"""
    
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
    
    def __init__(self, sorted_idxar, batch_size, drop_last=True, shuffle=False):
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
# GRADCAM
# ============================================================================
class GradCAM:
    """Generate GradCAM heatmaps for model predictions"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        """Generate GradCAM heatmap for target class.
        Expects input_tensor with shape (1, C, H, W).
        """
        self.model.eval()

        # Ensure tensor is on the same device and has gradient tracking
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_()
        
        # Re-enable grad even if caller is under torch.no_grad()
        with torch.enable_grad():
            output = self.model(input_tensor)
            self.model.zero_grad()
            target = output[0, target_class]
            target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        
        return heatmap
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        h, w = img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay


# ============================================================================
# ANALYZER
# ============================================================================
class ResNetAnalyzer:
    """Analyze ResNet model predictions and generate visualizations"""
    
    def __init__(self, model_path, data_path, labels_path, output_dir='./output', batch_size=32, save_interval=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.data_path = data_path
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)
        self._setup_directories()
        
        # Load class labels
        self.class_labels = self._load_class_labels(labels_path)
        
        self.predictions_data = []
        self.all_labels = []
        self.all_preds = []
        self.misclassification_count = 0
    
    def _load_model(self, model_path):
        """Load pretrained ResNet model as a normal nn.Module.

        - If checkpoint is a state_dict-like dict, load directly.
        - If checkpoint is a TorchScript / nn.Module, pull its state_dict.
        - Strip leading 'model.' and map 'downsample' -> 'shortcut' to match our implementation.
        """
        print("Loading model architecture...")
        model = resnet50(num_classes=1000)

        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 1) Get a state_dict from whatever we loaded
        if isinstance(checkpoint, dict) and not hasattr(checkpoint, "state_dict"):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            # TorchScript / nn.Module: use its state_dict()
            state_dict = checkpoint.state_dict()

        # 2) Normalize keys:
        #    - remove leading "model."
        #    - rename "downsample" -> "shortcut" to match our Bottleneck
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]
            if "downsample" in k:
                k = k.replace("downsample", "shortcut")
            new_state_dict[k] = v

        # 3) Load into our ResNet implementation
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        if missing:
            print("Warning: missing keys when loading state_dict:", missing)
        if unexpected:
            print("Warning: unexpected keys when loading state_dict:", unexpected)

        model = model.to(self.device)
        model.eval()
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def _load_class_labels(self, labels_path):
        """Load ImageNet class labels from text file"""
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(labels)} class labels from {labels_path}")
        return labels
    
    def _setup_directories(self):
        """Create output directories"""
        self.incorrect_dir = self.output_dir / 'incorrect_predictions'
        self.gradcam_dir = self.output_dir / 'gradcam'
        
        self.incorrect_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_dataloader(self):
        """Create validation dataloader with aspect ratio batching"""
        val_dir = os.path.join(self.data_path, 'val')
        
        # Sort by aspect ratio
        sorted_idxar = sort_ar(self.data_path, split='val')
        idx2ar = map_idx2ar(sorted_idxar, self.batch_size)
        
        # Create dataset with custom transforms
        transform = get_transforms(idx2ar, target_size=224)
        val_dataset = RectDataset(root=val_dir, transform=transform)
        
        # Create aspect ratio batch sampler
        sampler = AspectRatioBatchSampler(
            sorted_idxar, 
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=sampler,
            num_workers=2,  # Reduced for CPU
            pin_memory=False  # Disabled for CPU
        )
        
        return val_loader, val_dataset
    
    def _get_top5_predictions(self, output):
        """Get top-5 predicted classes and probabilities"""
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
        return top5_idx, top5_prob
    
    def _generate_gradcam(self, img_path, img_tensor, pred_class, img_name):
        """Generate and save GradCAM visualization"""
        # Now that we always use our ResNet, layer4[-1] is available
        target_layer = self.model.layer4[-1]
        
        gradcam = GradCAM(self.model, target_layer)
        heatmap = gradcam.generate(img_tensor.unsqueeze(0), pred_class)
        
        # Load original image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        if img_np.shape[-1] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        overlay = gradcam.overlay_heatmap(img_np, heatmap)
        
        # Save GradCAM
        save_path = self.gradcam_dir / f"{img_name}_gradcam.jpg"
        cv2.imwrite(str(save_path), overlay)
    
    def analyze_predictions(self):
        """Main analysis pipeline"""
        val_loader, val_dataset = self._get_dataloader()
        
        print(f"Analyzing {len(val_dataset)} validation images...")
        print(f"Using device: {self.device}")
        
        batch_count = 0
        img_count = 0
        
        # Get the batch sampler to access indices
        batch_sampler = val_loader.batch_sampler
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Processing batches")):
                # Get the actual indices for this batch from the sampler
                batch_indices = batch_sampler.batches[batch_idx]
                
                images = images.to(self.device)
                labels_np = labels.numpy()
                
                outputs = self.model(images)
                top5_idx, top5_prob = self._get_top5_predictions(outputs)
                _, top1_pred = outputs.max(1)
                
                # Process each image in batch
                for i, idx in enumerate(batch_indices):
                    img_path = val_dataset.imgs[idx][0]
                    img_name = Path(img_path).stem
                    
                    true_label = labels_np[i]
                    top1 = top1_pred[i].item()
                    top5 = top5_idx[i].cpu().numpy()
                    
                    # Get text labels
                    true_label_text = self.class_labels[true_label]
                    top1_text = self.class_labels[top1]
                    top5_text = [self.class_labels[idx] for idx in top5]
                    
                    # Store for classification report
                    self.all_labels.append(true_label)
                    self.all_preds.append(top1)
                    
                    # Check if prediction is incorrect (top-5)
                    is_incorrect = true_label not in top5
                    
                    # Store prediction data
                    self.predictions_data.append({
                        'image_name': img_name,
                        'true_label_idx': true_label,
                        'true_label': true_label_text,
                        'top1_pred_idx': top1,
                        'top1_pred': top1_text,
                        'top5_pred_idx': ','.join(map(str, top5)),
                        'top5_pred': ' | '.join(top5_text),
                        'is_incorrect_top5': is_incorrect
                    })
                    
                    # Save incorrect predictions (every Nth occurrence)
                    if is_incorrect:
                        self.misclassification_count += 1
                        
                        if self.misclassification_count % self.save_interval == 0:
                            self._save_incorrect_image(img_path, img_name)
                            # GradCAM for saved misclassifications
                            self._generate_gradcam(img_path, images[i].cpu(), top1, img_name)
                    
                    img_count += 1
                
                batch_count += 1
        
        print(f"\nAnalysis complete!")
        print(f"Total images processed: {img_count}")
        print(f"Total incorrect (top-5): {sum(p['is_incorrect_top5'] for p in self.predictions_data)}")
        print(f"Saved misclassifications: {self.misclassification_count // self.save_interval} (every {self.save_interval}th)")
    
    def _save_incorrect_image(self, img_path, img_name):
        """Copy incorrect prediction to output directory"""
        img = Image.open(img_path)
        save_path = self.incorrect_dir / f"{img_name}.jpg"
        img.save(save_path)
    
    def save_predictions_csv(self):
        """Save predictions to CSV file"""
        df = pd.DataFrame(self.predictions_data)
        csv_path = self.output_dir / 'predictions.csv'
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
    
    def save_classification_report(self):
        """Generate and save sklearn classification report"""
        report = classification_report(
            self.all_labels,
            self.all_preds,
            output_dict=True,
            zero_division=0
        )
        
        df_report = pd.DataFrame(report).transpose()
        csv_path = self.output_dir / 'classification_report.csv'
        df_report.to_csv(csv_path)
        print(f"Classification report saved to {csv_path}")
    
    def run(self):
        """Execute full analysis pipeline"""
        self.analyze_predictions()
        self.save_predictions_csv()
        self.save_classification_report()
        print(f"\nAll results saved to {self.output_dir}")


if __name__ == '__main__':
    # Configure paths for EC2 instance
    MODEL_PATH = '/home/ec2-user/resnet50_imagenet_1k_model.pt'
    DATA_PATH = '/mnt/data/imagenet'  # Path to mounted volume
    LABELS_PATH = '/home/ec2-user/imagenet_1k_classes.txt'  # Path to class labels
    OUTPUT_DIR = './analysis_output'
    
    # Run analysis
    analyzer = ResNetAnalyzer(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        labels_path=LABELS_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=32,
        save_interval=10  # Save every 10th misclassification
    )
    analyzer.run()
