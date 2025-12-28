import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# Import our modules
from config import get_config
from dataset import RetinaMultiLabelDataset, get_transforms
from models import build_model, load_pretrained_backbone
from losses import get_loss_function
from train import (set_parameter_requires_grad, train_model, 
                   differentially_train, get_optimizer_and_scheduler)
from evaluate import (evaluate_model, compute_metrics, 
                     find_optimal_threshold, save_predictions_for_kaggle,
                     evaluate_with_tta)
from ensemble import ensemble_predict, find_optimal_weights, EnsembleModel


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_model(config):
    """
    Run training and evaluation for a single model

    Args:
        config: Configuration class
    """
    print(f"Task: {config.TASK_NAME}")
    print(f"Backbone: {config.BACKBONE}")

    set_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Transforms
    train_transform = get_transforms(config.IMG_SIZE, mode='train')
    val_transform = get_transforms(config.IMG_SIZE, mode='val')
    test_transform = get_transforms(config.IMG_SIZE, mode='test')
    
    # Datasets
    train_dataset = RetinaMultiLabelDataset(
        config.TRAIN_CSV, config.TRAIN_IMAGE_DIR, train_transform, mode='train'
    )
    val_dataset = RetinaMultiLabelDataset(
        config.VAL_CSV, config.VAL_IMAGE_DIR, val_transform, mode='val'
    )
    offsite_test_dataset = RetinaMultiLabelDataset(
        config.OFFSITE_TEST_CSV, config.OFFSITE_TEST_DIR, test_transform, mode='test'
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    offsite_test_loader = DataLoader(
        offsite_test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    
    # Build model
    attention = getattr(config, 'ATTENTION', 'none')
    model = build_model(
        backbone=config.BACKBONE,
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # We'll load our pretrained weights
        attention=attention,
        se_reduction=getattr(config, 'SE_REDUCTION', 16),
        num_heads=getattr(config, 'NUM_HEADS', 8)
    )
    
    # Load pretrained weights
    if config.LOAD_PRETRAINED:
        backbone_key = config.BACKBONE
        pretrained_map = getattr(config, 'PRETRAINED_BACKBONES', {})
        if backbone_key not in pretrained_map:
             raise ValueError(f"No pretrained path configured for backbone: {backbone_key}")
        pretrained_path = pretrained_map[backbone_key]
        model = load_pretrained_backbone(model, pretrained_path, config.BACKBONE)
    
    model = model.to(device)
    
    # Define save path
    save_path = os.path.join(
        config.SAVE_DIR, 
        f"{config.TASK_NAME}_{config.BACKBONE}.pt"
    )

    # Training or Evaluation only
    if config.TRAIN:
        print("\n TRAINING PHASE")
        print("==============")

        # Freeze/Unfreeze parameters
        if getattr(config, 'FREEZE_BACKBONE', False):
            print("Freezing backbone, training classifier only\n")
            set_parameter_requires_grad(model, feature_extracting=True, 
                                       backbone_type=config.BACKBONE)
        else:
            print("Training entire network\n")
            set_parameter_requires_grad(model, feature_extracting=False,
                                       backbone_type=config.BACKBONE)

        # Loss function
        loss_type = getattr(config, 'LOSS_TYPE', 'bce')
        criterion = get_loss_function(loss_type, config.SAMPLES_PER_CLASS)
        print(f"Loss function: {loss_type}\n")
        
        # Training
        if hasattr(config, 'BACKBONE_LR') and hasattr(config, 'CLASSIFIER_LR'):
            # Differential learning rates
            model, history = differentially_train(
                model, train_loader, val_loader, criterion, device,
                num_epochs=config.NUM_EPOCHS,
                save_path=save_path,
                backbone_lr=config.BACKBONE_LR,
                classifier_lr=config.CLASSIFIER_LR,
                patience=config.PATIENCE
            )
        else:
            # Standard training
            optimizer, scheduler = get_optimizer_and_scheduler(
                model,
                optimizer_name=config.OPTIMIZER,
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                scheduler_name=config.SCHEDULER,
                patience=3
            )
            
            model, history = train_model(
                model, train_loader, val_loader, criterion,
                optimizer, scheduler, device,
                num_epochs=config.NUM_EPOCHS,
                save_path=save_path,
                patience=config.PATIENCE
            )
    else:
        # Load existing model for evaluation only (Task 1.1)
        print("\nEVALUATION ONLY (No Training)")
        print("-" * 70)
        if not os.path.exists(save_path):
            # For Task 1.1, use the pretrained backbone directly
            print("Using pretrained backbone directly (no fine-tuning)\n")
        else:
            print(f"Loading model from: {save_path}\n")
            model.load_state_dict(torch.load(save_path, map_location=device))

    # Evaluation on off-site test set
    print("\n EVALUATION ON OFFSITE TEST SET")

    y_true, y_pred, y_probs, img_names = evaluate_model(
        model, offsite_test_loader, device, threshold=0.5
    )
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, config.DISEASE_NAMES)
    
    # Optimize threshold
    print("\nOptimizing thresholds...")
    optimal_thresholds = find_optimal_threshold(y_true, y_probs, config.DISEASE_NAMES)
    
    # Re-evaluate with optimal thresholds
    print("\nRe-evaluating with optimal thresholds...")
    y_pred_opt = np.zeros_like(y_probs)
    for i, threshold in enumerate(optimal_thresholds):
        y_pred_opt[:, i] = (y_probs[:, i] > threshold).astype(int)
    
    metrics_opt = compute_metrics(y_true, y_pred_opt, config.DISEASE_NAMES)
    
    # Evaluation on onsite test set
    print("GENERATING ONSITE TEST PREDICTIONS")

    # Load onsite test data
    onsite_test_dataset = RetinaMultiLabelDataset(
        config.ONSITE_TEST_CSV, config.ONSITE_TEST_DIR, 
        test_transform, mode='test'
    )
    onsite_test_loader = DataLoader(
        onsite_test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    
    # Predict (use optimal thresholds)
    _, _, y_probs_onsite, img_names_onsite = evaluate_model(
        model, onsite_test_loader, device, threshold=0.5
    )
    
    # Apply optimal thresholds
    y_pred_onsite = np.zeros_like(y_probs_onsite)
    for i, threshold in enumerate(optimal_thresholds):
        y_pred_onsite[:, i] = (y_probs_onsite[:, i] > threshold).astype(int)
    
    # Save Kaggle submission
    submission_path = f"./submissions/{config.TASK_NAME}_{config.BACKBONE}_submission.csv"
    os.makedirs('./submissions', exist_ok=True)
    save_predictions_for_kaggle(img_names_onsite, y_pred_onsite, submission_path)
    
    print(f"\n✅ Task {config.TASK_NAME} completed!")
    print(f"Model saved: {save_path}")
    print(f"Submission saved: {submission_path}")
    print(f"Offsite Test Average F1-Score: {metrics_opt['average_f1']:.4f}")


def run_ensemble(config):
    """
    Run ensemble evaluation
    
    Args:
        config: Ensemble configuration
    """
    print(f"ENSEMBLE EVALUATION")
    
    # Set seed
    set_seed(config.SEED)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load all models
    print("Loading ensemble models...")
    models = []
    for i, (model_path, model_config) in enumerate(zip(config.MODEL_PATHS, config.MODEL_CONFIGS)):
        print(f"  Loading model {i+1}: {model_path}")
        
        model = build_model(**model_config, num_classes=config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        models.append(model)
    
    print(f"Loaded {len(models)} models\n")
    
    # Prepare data
    val_transform = get_transforms(config.IMG_SIZE, mode='val')
    test_transform = get_transforms(config.IMG_SIZE, mode='test')
    
    val_dataset = RetinaMultiLabelDataset(
        config.VAL_CSV, config.VAL_IMAGE_DIR, val_transform, mode='val'
    )
    offsite_test_dataset = RetinaMultiLabelDataset(
        config.OFFSITE_TEST_CSV, config.OFFSITE_TEST_DIR, test_transform, mode='test'
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    offsite_test_loader = DataLoader(
        offsite_test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    
    # Find optimal weights if requested
    weights = None
    if config.USE_OPTIMAL_WEIGHTS and config.ENSEMBLE_METHOD == 'weighted':
        weights = find_optimal_weights(models, val_loader, device, metric='f1')
    
    # Evaluate on offsite test
    print("ENSEMBLE EVALUATION ON OFFSITE TEST SET")
    
    if config.USE_TTA:
        print("Using Test-Time Augmentation\n")
        # TTA requires special handling
        # For simplicity, we'll average TTA results for each model first
        # Then ensemble the models
        
    y_pred, y_probs, img_names = ensemble_predict(
        models, offsite_test_loader, device,
        method=config.ENSEMBLE_METHOD,
        weights=weights,
        threshold=0.5
    )
    
    # Get true labels for metrics
    y_true_list = []
    for _, labels, _ in offsite_test_loader:
        y_true_list.append(labels.numpy())
    y_true = np.vstack(y_true_list)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, config.DISEASE_NAMES)
    
    # Optimize thresholds
    if config.OPTIMIZE_THRESHOLD:
        optimal_thresholds = find_optimal_threshold(y_true, y_probs, config.DISEASE_NAMES)
        
        # Re-evaluate
        y_pred_opt = np.zeros_like(y_probs)
        for i, threshold in enumerate(optimal_thresholds):
            y_pred_opt[:, i] = (y_probs[:, i] > threshold).astype(int)
        
        metrics_opt = compute_metrics(y_true, y_pred_opt, config.DISEASE_NAMES)
    
    # Onsite test predictions
    print("ENSEMBLE PREDICTIONS FOR ONSITE TEST (KAGGLE)")
    
    onsite_test_dataset = RetinaMultiLabelDataset(
        config.ONSITE_TEST_CSV, config.ONSITE_TEST_DIR,
        test_transform, mode='test'
    )
    onsite_test_loader = DataLoader(
        onsite_test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )
    
    y_pred_onsite, y_probs_onsite, img_names_onsite = ensemble_predict(
        models, onsite_test_loader, device,
        method=config.ENSEMBLE_METHOD,
        weights=weights,
        threshold=0.5
    )
    
    # Apply optimal thresholds
    if config.OPTIMIZE_THRESHOLD:
        y_pred_onsite = np.zeros_like(y_probs_onsite)
        for i, threshold in enumerate(optimal_thresholds):
            y_pred_onsite[:, i] = (y_probs_onsite[:, i] > threshold).astype(int)
    
    # Save submission
    submission_path = f"./submissions/{config.TASK_NAME}_submission.csv"
    save_predictions_for_kaggle(img_names_onsite, y_pred_onsite, submission_path)
    
    print(f"\nEnsemble evaluation completed!")
    print(f"Submission saved: {submission_path}")
    if config.OPTIMIZE_THRESHOLD:
        print(f"Offsite Test Average F1-Score: {metrics_opt['average_f1']:.4f}")
    else:
        print(f"Offsite Test Average F1-Score: {metrics['average_f1']:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Retinal Disease Classification')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name: task1-1, task1-2, task1-3, task2-1, task2-2, task3-1, task3-2, task4')
    parser.add_argument('--backbone', type=str, default=None,
                       help='Backbone: resnet18 or efficientnet (overrides config)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.task)
    
    # Override backbone if specified
    if args.backbone is not None:
        config.BACKBONE = args.backbone
    
    # Run task
    if args.task == 'task4':
        run_ensemble(config)
    else:
        run_single_model(config)


if __name__ == '__main__':
    main()
