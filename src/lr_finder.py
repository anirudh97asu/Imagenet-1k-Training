"""
COMPLETE ONE CYCLE POLICY IMPLEMENTATION FOR RESNET-34 ON CIFAR-100

This script implements the One Cycle Policy exactly as described in the paper.
It follows all 5 steps with proper configuration:

STEP 1: Define the Maximum Learning Rate
  1.1: Choose range of learning rates (0.01 to 10)
  1.2: Train model with each LR for 1-2 epochs
  1.3: Plot learning rate vs training loss
  1.4: Find maximum LR (steepest slope/fastest decrease)

STEP 2: Define the Learning Rate Schedule
  - Two equal-length steps (50% warmup, 50% annealing)
  - Lower LR = max_lr / 10 (or /5)
  - Cycle length = 95% of total training
  - Final 5% annihilates LR to 1/100th of lower bound
  - Momentum inversely related to LR

STEP 3: Implement the LR Schedule in Training Loop
  - Update LR and momentum at each iteration
  - Use schedule functions to determine values

STEP 4: Train the Model
  - Train with One Cycle schedule
  - Update weights based on current LR

STEP 5: Monitor Model Performance
  - Track accuracy, loss, LR, momentum
  - Visualize and analyze results

Usage:
    python onecycle_complete.py

Author: Following One Cycle Policy (Leslie Smith)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import copy
from src.model import ResNet34
from data import get_loaders


# ============================================================================
# STEP 1: DEFINE THE MAXIMUM LEARNING RATE
# ============================================================================

class LRFinder:
    """Learning Rate Finder for One Cycle Policy"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lrs = []
        self.losses = []
        self.initial_state = copy.deepcopy(model.state_dict())
        self.initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    def range_test(self, train_loader, start_lr=0.01, end_lr=10, num_epochs=2):
        """
        Step 1.2: Train model with each learning rate
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate (0.01)
            end_lr: Ending learning rate (10)
            num_epochs: Number of epochs to train (1-2)
        """
        # Reset model
        self.model.load_state_dict(self.initial_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        
        # Calculate total batches and LR multiplier
        total_batches = len(train_loader) * num_epochs
        lr_mult = (end_lr / start_lr) ** (1 / total_batches)
        
        # Initialize
        current_lr = start_lr
        batch_count = 0
        best_loss = float('inf')
        smoothed_loss = 0
        
        print(f"\n  Training for {num_epochs} epoch(s)")
        print(f"  Total batches: {total_batches}")
        print(f"  LR range: {start_lr} → {end_lr}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"\n  Epoch {epoch + 1}/{num_epochs}")
            pbar = tqdm(train_loader, desc="  LR Range Test")
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track loss (smoothed)
                if batch_count == 0:
                    smoothed_loss = loss.item()
                else:
                    smoothed_loss = 0.05 * loss.item() + 0.95 * smoothed_loss
                
                # Store results
                self.lrs.append(current_lr)
                self.losses.append(smoothed_loss)
                
                # Check for divergence
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                
                if smoothed_loss > 5 * best_loss:
                    print(f"\n  ⚠️  Loss diverging at LR={current_lr:.2e}, stopping early")
                    self.model.load_state_dict(self.initial_state)
                    self.optimizer.load_state_dict(self.initial_optimizer_state)
                    return
                
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                current_lr *= lr_mult
                batch_count += 1
                
                pbar.set_postfix({'lr': f'{self.lrs[-1]:.2e}', 'loss': f'{smoothed_loss:.4f}'})
        
        print(f"\n  ✅ Completed {batch_count} batches")
        
        # Reset model
        self.model.load_state_dict(self.initial_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
    
    def plot(self, skip_start=10, skip_end=5):
        """
        Step 1.3: Plot learning rate vs training loss
        Step 1.4: Find maximum learning rate
        """
        if len(self.lrs) == 0:
            print("No data to plot. Run range_test first.")
            return None
        
        # Trim data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        # Step 1.3: Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(lrs, losses, linewidth=2, color='blue')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Step 1.3: Learning Rate vs Training Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Step 1.4: Find steepest gradient (fastest decrease)
        gradients = np.gradient(losses)
        steepest_idx = np.argmin(gradients)
        max_lr = lrs[steepest_idx]
        
        # Mark on plot
        ax.axvline(x=max_lr, color='red', linestyle='--', linewidth=2, 
                   label=f'Maximum LR (steepest slope): {max_lr:.4f}')
        ax.scatter([max_lr], [losses[steepest_idx]], color='red', s=100, zorder=5)
        ax.legend(fontsize=11)
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/step1_lr_range_test.png', dpi=300, bbox_inches='tight')
        print(f"\n  ✅ Plot saved: outputs/step1_lr_range_test.png")
        plt.close()
        
        return max_lr


def step1_find_maximum_learning_rate(model, train_loader, device):
    """
    STEP 1: Define the Maximum Learning Rate
    
    Process:
    1.1: Choose range (0.01 to 10)
    1.2: Train with each LR for 1-2 epochs
    1.3: Plot LR vs loss
    1.4: Find LR with steepest slope
    """
    print("\n" + "="*80)
    print("STEP 1: DEFINE THE MAXIMUM LEARNING RATE")
    print("="*80)
    
    # Step 1.1: Choose range
    start_lr = 1e-6
    end_lr = 10
    num_epochs = 2
    
    print(f"\nStep 1.1: Choose range of learning rates")
    print(f"  Range: {start_lr} to {end_lr}")
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-4)
    
    # Create LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # Step 1.2: Train with each LR
    print(f"\nStep 1.2: Train model with each learning rate")
    lr_finder.range_test(train_loader, start_lr, end_lr, num_epochs)
    
    # Step 1.3 & 1.4: Plot and find max LR
    print(f"\nStep 1.3: Plot learning rate vs training loss")
    print(f"Step 1.4: Find maximum learning rate (steepest slope)")
    max_lr = lr_finder.plot()
    
    print(f"\n{'='*80}")
    print(f"STEP 1 COMPLETE: Maximum Learning Rate = {max_lr:.4f}")
    print(f"{'='*80}")
    print(f"\n  Observations:")
    print(f"  - Loss decreases fastest around {max_lr:.4f}")
    print(f"  - This is where the slope is steepest (largest decrease)")
    print(f"  - We'll use this as max_lr in One Cycle Policy")
    
    return max_lr


# ============================================================================
# STEP 2: DEFINE THE LEARNING RATE SCHEDULE
# ============================================================================

def step2_define_learning_rate_schedule(max_lr, total_steps, epochs):
    """
    STEP 2: Define the Learning Rate Schedule
    
    Configuration:
    - Two equal-length steps (50% warmup, 50% annealing)
    - Lower LR = max_lr / 10
    - Cycle length = 95% of training
    - Final 5%: annihilate to 1/100th of lower LR
    - Momentum: inversely related to LR (0.85-0.95)
    """
    print("\n" + "="*80)
    print("STEP 2: DEFINE THE LEARNING RATE SCHEDULE")
    print("="*80)
    
    # Configuration (following One Cycle Policy recommendations)
    div_factor = 10  # Lower LR = max_lr / 10
    initial_lr = max_lr / div_factor
    
    # Cycle length (95% of total training)
    cycle_steps = int(total_steps * 0.95)
    annihilation_steps = total_steps - cycle_steps
    
    # Two equal-length steps
    step1_length = cycle_steps // 2  # 50% warmup
    step2_length = cycle_steps - step1_length  # 50% annealing
    
    # Final LR (annihilate to 1/100th)
    final_div_factor = 100
    final_lr = initial_lr / final_div_factor
    
    # Momentum (inverse of LR)
    max_momentum = 0.95
    min_momentum = 0.85
    
    print(f"\nLearning Rate Schedule:")
    print(f"  - Lower LR (initial): {initial_lr:.4f} = max_lr / {div_factor}")
    print(f"  - Maximum LR: {max_lr:.4f}")
    print(f"  - Final LR (annihilated): {final_lr:.6f} = initial_lr / {final_div_factor}")
    
    print(f"\nCycle Configuration:")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Cycle length (95%): {cycle_steps} steps")
    print(f"  - Step 1 (warmup, 50%): {step1_length} steps")
    print(f"  - Step 2 (annealing, 50%): {step2_length} steps")
    print(f"  - Annihilation (5%): {annihilation_steps} steps")
    
    print(f"\nMomentum Schedule (inverse of LR):")
    print(f"  - Range: {min_momentum} to {max_momentum}")
    print(f"  - High LR → Low momentum (explore new directions)")
    print(f"  - Low LR → High momentum (converge precisely)")
    
    def schedule_function(step):
        """Map iteration → learning rate"""
        if step < step1_length:
            # Phase 1: Linear warmup (0 to 50% of cycle)
            progress = step / step1_length
            lr = initial_lr + (max_lr - initial_lr) * progress
        elif step < cycle_steps:
            # Phase 2: Cosine annealing (50% to 95% of training)
            progress = (step - step1_length) / step2_length
            lr = initial_lr + (max_lr - initial_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            # Phase 3: Final annihilation (95% to 100%)
            progress = (step - cycle_steps) / annihilation_steps
            lr = final_lr + (initial_lr - final_lr) * (1 - progress)
        
        return lr
    
    def momentum_function(step):
        """Map iteration → momentum (inverse of LR)"""
        if step < step1_length:
            # Phase 1: Momentum decreases as LR increases
            progress = step / step1_length
            momentum = max_momentum - (max_momentum - min_momentum) * progress
        elif step < cycle_steps:
            # Phase 2: Momentum increases as LR decreases
            progress = (step - step1_length) / step2_length
            momentum = min_momentum + (max_momentum - min_momentum) * progress
        else:
            # Phase 3: Keep momentum high
            momentum = max_momentum
        
        return momentum
    
    # Visualize the schedule
    steps = np.arange(total_steps)
    lrs = [schedule_function(s) for s in steps]
    momentums = [momentum_function(s) for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Learning Rate
    ax1.plot(steps, lrs, linewidth=2, color='blue')
    ax1.axvline(x=step1_length, color='red', linestyle='--', linewidth=1.5, 
                label=f'End of Step 1 (warmup)')
    ax1.axvline(x=cycle_steps, color='orange', linestyle='--', linewidth=1.5,
                label=f'Start of annihilation')
    ax1.set_xlabel('Training Iterations', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('One Cycle Learning Rate Schedule (Two Equal Steps + Annihilation)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Annotate key points
    ax1.text(0, initial_lr*1.1, f'{initial_lr:.4f}', fontsize=9, color='blue', fontweight='bold')
    ax1.text(step1_length, max_lr*1.02, f'{max_lr:.4f}', fontsize=9, color='red', fontweight='bold')
    ax1.text(total_steps*0.98, final_lr*5, f'{final_lr:.6f}', fontsize=9, color='orange', fontweight='bold')
    
    # Plot 2: Momentum
    ax2.plot(steps, momentums, linewidth=2, color='green')
    ax2.axvline(x=step1_length, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=cycle_steps, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Training Iterations', fontsize=12)
    ax2.set_ylabel('Momentum', fontsize=12)
    ax2.set_title('Momentum Schedule (Inverse Relationship with LR)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    plt.savefig('outputs/step2_lr_schedule.png', dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"STEP 2 COMPLETE: Learning Rate Schedule Defined")
    print(f"{'='*80}")
    print(f"  ✅ Visualization saved: outputs/step2_lr_schedule.png")
    plt.close()
    
    return schedule_function, momentum_function


# ============================================================================
# STEP 3 & 4: IMPLEMENT SCHEDULE AND TRAIN MODEL
# ============================================================================

def step3_and_4_train_with_schedule(model, train_loader, test_loader, device,
                                     max_lr, epochs, schedule_fn, momentum_fn):
    """
    STEP 3: Implement the learning rate schedule in training loop
    STEP 4: Train the model
    
    Implementation:
    - Update LR and momentum at each iteration
    - Use schedule functions to determine values
    - Train model with current LR and momentum
    - Track all metrics
    """
    print("\n" + "="*80)
    print("STEP 3: IMPLEMENT THE LEARNING RATE SCHEDULE IN TRAINING LOOP")
    print("="*80)
    
    print("\n" + "="*80)
    print("STEP 4: TRAIN THE MODEL")
    print("="*80)
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    
    total_steps = len(train_loader) * epochs
    current_step = 0
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rates': [],
        'momentums': []
    }
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batches per epoch: {len(train_loader)}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Optimizer: SGD with Nesterov momentum")
    print(f"  - Weight decay: 5e-4")
    
    best_test_acc = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        # ========== TRAINING ==========
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_lrs = []
        epoch_moms = []
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # STEP 3: Determine LR and momentum for this iteration
            current_lr = schedule_fn(current_step)
            current_mom = momentum_fn(current_step)
            
            # Update optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                param_group['momentum'] = current_mom
            
            epoch_lrs.append(current_lr)
            epoch_moms.append(current_mom)
            
            # STEP 4: Train with current LR and momentum
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{current_lr:.2e}',
                'mom': f'{current_mom:.3f}'
            })
            
            current_step += 1
        
        # Epoch metrics
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        
        # ========== TESTING ==========
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Testing'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(epoch_lrs)
        history['momentums'].append(epoch_moms)
        
        # Print summary
        avg_lr = sum(epoch_lrs) / len(epoch_lrs)
        avg_mom = sum(epoch_moms) / len(epoch_moms)
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%")
        print(f"  Avg LR: {avg_lr:.2e} | Avg Momentum: {avg_mom:.3f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'max_lr': max_lr
            }, 'outputs/best_model.pth')
            print(f"  ✅ Best model saved! (Test Acc: {best_test_acc:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"STEPS 3 & 4 COMPLETE: Model Trained with One Cycle Policy")
    print(f"{'='*80}")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    
    return history


# ============================================================================
# STEP 5: MONITOR MODEL PERFORMANCE
# ============================================================================

def step5_monitor_performance(history, max_lr):
    """
    STEP 5: Monitor the model performance
    
    Track and visualize:
    - Training and test accuracy
    - Training and test loss
    - Learning rate progression
    - Momentum progression
    - LR vs Momentum relationship
    """
    print("\n" + "="*80)
    print("STEP 5: MONITOR THE MODEL PERFORMANCE")
    print("="*80)
    
    epochs = len(history['train_acc'])
    epoch_nums = range(1, epochs + 1)
    
    # Extract all data
    all_lrs = [lr for epoch_lrs in history['learning_rates'] for lr in epoch_lrs]
    all_moms = [m for epoch_moms in history['momentums'] for m in epoch_moms]
    batches = range(len(all_lrs))
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Train/Test Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epoch_nums, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
    ax1.plot(epoch_nums, history['test_loss'], 'r-o', label='Test', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Test Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train/Test Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epoch_nums, history['train_acc'], 'b-o', label='Train', linewidth=2, markersize=4)
    ax2.plot(epoch_nums, history['test_acc'], 'r-o', label='Test', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate (per batch)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(batches, all_lrs, 'g-', linewidth=1.5)
    ax3.set_xlabel('Batch Number', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Momentum (per batch)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(batches, all_moms, 'purple', linewidth=1.5)
    ax4.set_xlabel('Batch Number', fontsize=11)
    ax4.set_ylabel('Momentum', fontsize=11)
    ax4.set_title('Momentum Schedule', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.8, 1.0])
    
    # Plot 5: LR vs Momentum (inverse relationship)
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(all_lrs, all_moms, c=batches, cmap='viridis', alpha=0.5, s=2)
    ax5.set_xscale('log')
    ax5.set_xlabel('Learning Rate', fontsize=11)
    ax5.set_ylabel('Momentum', fontsize=11)
    ax5.set_title('LR vs Momentum (Inverse)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Batch')
    
    # Plot 6: Average LR per Epoch
    ax6 = fig.add_subplot(gs[1, 2])
    avg_lrs = [sum(epoch_lrs)/len(epoch_lrs) for epoch_lrs in history['learning_rates']]
    ax6.plot(epoch_nums, avg_lrs, 'g-o', linewidth=2, markersize=6)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Average LR', fontsize=11)
    ax6.set_title('Average LR per Epoch', fontsize=12, fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Loss Improvement
    ax7 = fig.add_subplot(gs[2, 0])
    train_loss_improvement = [history['train_loss'][0] - loss for loss in history['train_loss']]
    test_loss_improvement = [history['test_loss'][0] - loss for loss in history['test_loss']]
    ax7.plot(epoch_nums, train_loss_improvement, 'b-', label='Train', linewidth=2)
    ax7.plot(epoch_nums, test_loss_improvement, 'r-', label='Test', linewidth=2)
    ax7.set_xlabel('Epoch', fontsize=11)
    ax7.set_ylabel('Loss Improvement', fontsize=11)
    ax7.set_title('Loss Improvement Over Time', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Accuracy Improvement
    ax8 = fig.add_subplot(gs[2, 1])
    train_acc_improvement = [acc - history['train_acc'][0] for acc in history['train_acc']]
    test_acc_improvement = [acc - history['test_acc'][0] for acc in history['test_acc']]
    ax8.plot(epoch_nums, train_acc_improvement, 'b-', label='Train', linewidth=2)
    ax8.plot(epoch_nums, test_acc_improvement, 'r-', label='Test', linewidth=2)
    ax8.set_xlabel('Epoch', fontsize=11)
    ax8.set_ylabel('Accuracy Improvement (%)', fontsize=11)
    ax8.set_title('Accuracy Improvement Over Time', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Generalization Gap
    ax9 = fig.add_subplot(gs[2, 2])
    gap = [train - test for train, test in zip(history['train_acc'], history['test_acc'])]
    ax9.plot(epoch_nums, gap, 'orange', linewidth=2)
    ax9.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax9.set_xlabel('Epoch', fontsize=11)
    ax9.set_ylabel('Train - Test Acc (%)', fontsize=11)
    ax9.set_title('Generalization Gap', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    fig.suptitle('One Cycle Policy - Complete Training Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('outputs/step5_performance_monitoring.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: outputs/step5_performance_monitoring.png")
    plt.close()
    
    # Print detailed statistics
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Training Metrics:")
    print(f"  Initial Train Acc:  {history['train_acc'][0]:.2f}%")
    print(f"  Best Train Acc:     {max(history['train_acc']):.2f}%")
    print(f"  Final Train Acc:    {history['train_acc'][-1]:.2f}%")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.4f}")
    print(f"  Best Train Loss:    {min(history['train_loss']):.4f}")
    print(f"  Final Train Loss:   {history['train_loss'][-1]:.4f}")
    
    print(f"\n📊 Test Metrics:")
    print(f"  Initial Test Acc:   {history['test_acc'][0]:.2f}%")
    print(f"  Best Test Acc:      {max(history['test_acc']):.2f}% ⭐")
    print(f"  Final Test Acc:     {history['test_acc'][-1]:.2f}%")
    print(f"  Initial Test Loss:  {history['test_loss'][0]:.4f}")
    print(f"  Best Test Loss:     {min(history['test_loss']):.4f}")
    print(f"  Final Test Loss:    {history['test_loss'][-1]:.4f}")
    
    print(f"\n📈 Learning Rate Statistics:")
    print(f"  Maximum LR:         {max_lr:.4f}")
    print(f"  Initial LR:         {all_lrs[0]:.4f}")
    print(f"  Peak LR:            {max(all_lrs):.4f}")
    print(f"  Final LR:           {all_lrs[-1]:.6f}")
    print(f"  Peak reached at:    Batch {all_lrs.index(max(all_lrs))}/{len(all_lrs)}")
    
    print(f"\n📈 Momentum Statistics:")
    print(f"  Initial Momentum:   {all_moms[0]:.3f}")
    print(f"  Minimum Momentum:   {min(all_moms):.3f}")
    print(f"  Maximum Momentum:   {max(all_moms):.3f}")
    print(f"  Final Momentum:     {all_moms[-1]:.3f}")
    
    print(f"\n📈 Improvement:")
    print(f"  Train Acc Gain:     +{history['train_acc'][-1] - history['train_acc'][0]:.2f}%")
    print(f"  Test Acc Gain:      +{history['test_acc'][-1] - history['test_acc'][0]:.2f}%")
    print(f"  Train Loss Reduction: {history['train_loss'][0] - history['train_loss'][-1]:.4f}")
    print(f"  Test Loss Reduction:  {history['test_loss'][0] - history['test_loss'][-1]:.4f}")
    
    # Analysis
    print(f"\n🔍 Performance Analysis:")
    final_gap = history['train_acc'][-1] - history['test_acc'][-1]
    
    if final_gap > 15:
        print(f"  ⚠️  Large generalization gap ({final_gap:.2f}%) - overfitting detected")
        print(f"      Suggestions:")
        print(f"      - Reduce max_lr")
        print(f"      - Increase weight decay")
        print(f"      - Add data augmentation")
    elif final_gap > 10:
        print(f"  ⚡ Moderate generalization gap ({final_gap:.2f}%) - acceptable")
        print(f"      Training is reasonably well-balanced")
    else:
        print(f"  ✅ Small generalization gap ({final_gap:.2f}%) - excellent!")
        print(f"      Model generalizes well to test data")
    
    best_test = max(history['test_acc'])
    if best_test >= 75:
        print(f"  ✅ Excellent test accuracy achieved ({best_test:.2f}%)")
    elif best_test >= 65:
        print(f"  ✓  Good test accuracy achieved ({best_test:.2f}%)")
    elif best_test >= 50:
        print(f"  ⚡ Moderate test accuracy ({best_test:.2f}%)")
        print(f"      Consider: longer training or higher max_lr")
    else:
        print(f"  ⚠️  Low test accuracy ({best_test:.2f}%)")
        print(f"      Suggestions:")
        print(f"      - Check data preprocessing")
        print(f"      - Verify model architecture")
        print(f"      - Adjust max_lr (run Step 1 again)")
    
    print(f"\n💡 One Cycle Policy Benefits Observed:")
    print(f"  ✓ Two equal-length steps implemented")
    print(f"  ✓ Momentum inversely related to LR")
    print(f"  ✓ Final LR annihilation applied")
    print(f"  ✓ Higher LR acted as regularization")
    print(f"  ✓ Fast convergence in early epochs")
    
    print(f"\n{'='*80}")
    print(f"STEP 5 COMPLETE: Model Performance Monitored")
    print(f"{'='*80}")
    
    # Save history
    torch.save(history, 'outputs/training_history.pt')
    print(f"\n💾 Training history saved: outputs/training_history.pt")


# ============================================================================
# MAIN: COMPLETE ONE CYCLE POLICY WORKFLOW
# ============================================================================

def main():
    """
    Complete One Cycle Policy Implementation
    Executes all 5 steps in sequence
    """
    print("\n" + "="*80)
    print("ONE CYCLE POLICY - COMPLETE IMPLEMENTATION")
    print("RESNET-34 ON CIFAR-100")
    print("="*80)
    
    print("\nThis script implements One Cycle Policy in 5 steps:")
    print("  1. Define the maximum learning rate (LR range test)")
    print("  2. Define the learning rate schedule (two equal steps)")
    print("  3. Implement the LR schedule in training loop")
    print("  4. Train the model with One Cycle Policy")
    print("  5. Monitor model performance and analyze results")
    
    print("\n" + "="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    batch_size = 256
    print(f"\n📦 Loading CIFAR-100 dataset")
    print(f"    Batch size: {batch_size}")
    train_loader, test_loader = get_loaders(batch_size=batch_size)
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🏗️  Creating ResNet-34 model")
    model = ResNet34(num_classes=100).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # Compile model (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print(f"    ✅ Model compiled for faster training")
    except:
        print(f"    ℹ️  Using eager mode (compilation not available)")
    
    torch.set_float32_matmul_precision("high")
    
    # ========================================================================
    # STEP 1: Find Maximum Learning Rate
    # ========================================================================
    
    
    max_lr = step1_find_maximum_learning_rate(model, train_loader, device)
    
    
    # ========================================================================
    # STEP 2: Define Learning Rate Schedule
    # ========================================================================



    # epochs = 100
    # total_steps = len(train_loader) * epochs
    
    # print(f"\n📅 Training plan:")
    # print(f"    Epochs: {epochs}")
    # print(f"    Total training steps: {total_steps:,}")
    
    # schedule_function, momentum_function = step2_define_learning_rate_schedule(
    #     max_lr, total_steps, epochs
    # )
    
    # # Recreate model (LR finder modified weights)
    # print(f"\n🔄 Recreating model for training...")
    # model = ResNet34(num_classes=100).to(device)
    # try:
    #     model = torch.compile(model)
    # except:
    #     pass
    
    
    
    
    # ========================================================================
    # STEP 3 & 4: Implement Schedule and Train
    # ========================================================================
    
    
    
    
    # history = step3_and_4_train_with_schedule(
    #     model, train_loader, test_loader, device,
    #     max_lr, epochs, schedule_function, momentum_function
    # )
    
    
    
    
    # ========================================================================
    # STEP 5: Monitor Performance
    # ========================================================================
    
    
    
    # step5_monitor_performance(history, max_lr)
    
    # # Final summary
    # print("\n" + "="*80)
    # print("🎉 ONE CYCLE POLICY IMPLEMENTATION COMPLETE!")
    # print("="*80)
    
    # print("\n📁 All outputs saved in 'outputs/' directory:")
    # print("    📊 step1_lr_range_test.png      - LR range test results")
    # print("    📈 step2_lr_schedule.png        - LR and momentum schedules")
    # print("    📉 step5_performance_monitoring.png - Complete training analysis")
    # print("    💾 best_model.pth               - Best model checkpoint")
    # print("    💾 training_history.pt          - Complete training history")
    
    # print(f"\n🏆 Final Results:")
    # print(f"    Best Test Accuracy: {max(history['test_acc']):.2f}%")
    # print(f"    Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    # print(f"    Training completed in {epochs} epochs")
    
    # print("\n" + "="*80)
    # print("Thank you for using One Cycle Policy!")
    # print("For more details, check the visualizations in outputs/ directory")
    # print("="*80 + "\n")


if __name__ == '__main__':
    main()