import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy


def set_parameter_requires_grad(model, feature_extracting, backbone_type='resnet18'):
    """
    Freeze or unfreeze model parameters

    Args:
        model: PyTorch model
        feature_extracting: True = freeze backbone, False = unfreeze all
        backbone_type: 'resnet18' | 'efficientnet'
    """
    if feature_extracting:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier only
        if backbone_type == 'resnet18' or 'Resnet18' in str(type(model)):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif backbone_type == 'efficientnet' or 'EfficientNet' in str(type(model)):
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            # For custom models with attention
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            if hasattr(model.backbone, 'fc'):
                for param in model.backbone.fc.parameters():
                    param.requires_grad = True
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            if hasattr(model.backbone, 'classifier'):
                for param in model.backbone.classifier.parameters():
                    param.requires_grad = True


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Returns:
        avg_loss: Average training loss
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss


def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(val_loader.dataset)
    return avg_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path, patience=7):
    """
    Complete training loop with early stopping

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: 'cuda' or 'cpu'
        num_epochs: Number of epochs to train
        save_path: Path to save the best model
        patience: Early stopping patience

    Returns:
        model: Trained model
        history: Training history
    """
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"Start training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    return model, history


def get_optimizer_and_scheduler(model, optimizer_name='adam', lr=1e-4, weight_decay=1e-4, scheduler_name='plateau', patience=3, factor=0.5):
    """
    Get optimizer and and learning rate scheduler

    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'adamw', 'sgd'
        lr: Learning rate
        weight_decay: L2 regularization
        scheduler_name: 'plateau', 'cosine', 'step', None
        patience: Patience for ReduceLROnPlateau
        factor: LR reduction factor
    
    Returns:
        optimizer, scheduler
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Scheduler
    if scheduler_name is None or scheduler_name.lower() == 'none':
        scheduler = None
    elif scheduler_name.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            verbose=True
        )
    elif scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
    elif scheduler_name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return optimizer, scheduler
    

def differentially_train(model, train_loader, val_loader, criterion, device, num_epochs, save_path, backbone_lr=1e-5, classifier_lr=1e-3, patience=7):
    """
    Train model with differential learning rates for backbone and classifier

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: 'cuda' or 'cpu'
        num_epochs: Number of epochs to train
        save_path: Path to save the best model
        backbone_lr: Learning rate for backbone (smaller)
        classifier_lr: Learning rate for classifier (larger)
        patience: Early stopping patience
    """

    # Separate parameters
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': classifier_lr}
    ], weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5,
        verbose=True
    )

    print(f"\n Differential Learning Rate:")
    print(f"  Backbone LR: {backbone_lr:.2e}")
    print(f"  Classifier LR: {classifier_lr:.2e}\n")
    print(f"  Backbone params: {len(backbone_params)}")
    print(f"  Classifier params: {len(classifier_params)}")

    return train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, save_path, patience)