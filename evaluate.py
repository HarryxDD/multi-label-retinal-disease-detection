import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, cohen_kappa_score, confusion_matrix)
from tqdm import tqdm
import os

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: cuda or cpu
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        img_names: Image names
    """
    model.eval()
    
    y_true_list = []
    y_probs_list = []
    img_names_list = []
    
    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Store results
            y_probs_list.append(probs)
            
            if labels.numel() > 0:  # Has labels
                y_true_list.append(labels.numpy())
            
            img_names_list.extend(names)
    
    # Concatenate all batches
    y_probs = np.vstack(y_probs_list)
    y_pred = (y_probs > threshold).astype(int)
    
    if len(y_true_list) > 0:
        y_true = np.vstack(y_true_list)
    else:
        y_true = None
    
    return y_true, y_pred, y_probs, img_names_list


def compute_metrics(y_true, y_pred, disease_names=['D', 'G', 'A']):
    """
    Compute detailed metrics for each disease
    
    Args:
        y_true: True labels (N, 3)
        y_pred: Predicted labels (N, 3)
        disease_names: List of disease names
    
    Returns:
        metrics_dict: Dictionary of metrics
    """
    metrics_dict = {}
    
    print("DETAILED EVALUATION METRICS")
    
    # Overall metrics
    overall_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    # Per-disease metrics
    f1_scores = []
    
    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Compute metrics
        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, zero_division=0)
        recall = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store metrics
        metrics_dict[disease] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'kappa': kappa,
            'specificity': specificity,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        f1_scores.append(f1)
        
        # Print results
        print(f"\n{disease} (Diabetic Retinopathy)" if disease == 'D' 
              else f"\n{disease} (Glaucoma)" if disease == 'G'
              else f"\n{disease} (AMD)")
        print("-" * 40)
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {precision:.4f}")
        print(f"  Recall     : {recall:.4f}")
        print(f"  F1-Score   : {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Kappa      : {kappa:.4f}")
        print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Average F1-score (THIS IS THE MAIN METRIC!)
    avg_f1 = np.mean(f1_scores)
    metrics_dict['average_f1'] = avg_f1
    
    print(f"AVERAGE F1-SCORE: {avg_f1:.4f}")
    
    return metrics_dict


def find_optimal_threshold(y_true, y_probs, disease_names=['D', 'G', 'A']):
    """
    Find optimal threshold for each disease to maximize F1-score
    
    Args:
        y_true: True labels (N, 3)
        y_probs: Predicted probabilities (N, 3)
        disease_names: List of disease names
    
    Returns:
        optimal_thresholds: List of optimal thresholds
    """
    print("THRESHOLD OPTIMIZATION")
    
    optimal_thresholds = []
    
    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_probs[:, i]
        
        best_f1 = 0
        best_threshold = 0.5
        
        # Try different thresholds
        for threshold in np.arange(0.3, 0.7, 0.05):
            pred = (y_p > threshold).astype(int)
            f1 = f1_score(y_t, pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"{disease}: Optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
    
    return optimal_thresholds


def save_predictions_for_kaggle(img_names, y_pred, save_path='submission.csv'):
    """
    Save predictions in Kaggle submission format
    
    Args:
        img_names: List of image names
        y_pred: Predicted labels (N, 3)
        save_path: Path to save CSV
    """
    # Create DataFrame
    df = pd.DataFrame({
        'id': img_names,
        'D': y_pred[:, 0],
        'G': y_pred[:, 1],
        'A': y_pred[:, 2]
    })
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nKaggle submission saved to: {save_path}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Format: id, D, G, A")
    print(f"\nPrediction distribution:")
    print(f"  D (Diabetic Retinopathy): {df['D'].sum()}")
    print(f"  G (Glaucoma): {df['G'].sum()}")
    print(f"  A (AMD): {df['A'].sum()}")


def evaluate_with_tta(model, test_dataset, device, batch_size=32, 
                      threshold=0.5, num_workers=4):
    """
    Evaluate with Test-Time Augmentation
    
    Args:
        model: PyTorch model
        test_dataset: Test dataset (without augmentation)
        device: cuda or cpu
        batch_size: Batch size
        threshold: Classification threshold
        num_workers: Number of dataloader workers
    
    Returns:
        y_true, y_pred, y_probs, img_names
    """
    from dataset import get_tta_transforms
    import copy
    
    model.eval()
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    y_probs_all = []
    y_true_list = []
    img_names_list = []
    
    print(f"\nPerforming Test-Time Augmentation ({len(tta_transforms)} variations)")
    
    # Apply each TTA transform
    for tta_idx, tta_transform in enumerate(tta_transforms):
        print(f"  TTA {tta_idx + 1}/{len(tta_transforms)}...")
        
        # Create new dataset with this transform
        tta_dataset = copy.copy(test_dataset)
        tta_dataset.transform = tta_transform
        
        # Create dataloader
        tta_loader = torch.utils.data.DataLoader(
            tta_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Predict
        y_probs_tta = []
        with torch.no_grad():
            for imgs, labels, names in tqdm(tta_loader, desc=f"TTA {tta_idx+1}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_probs_tta.append(probs)
                
                # Store labels and names only once
                if tta_idx == 0:
                    if labels.numel() > 0:
                        y_true_list.append(labels.numpy())
                    img_names_list.extend(names)
        
        y_probs_tta = np.vstack(y_probs_tta)
        y_probs_all.append(y_probs_tta)
    
    # Average all TTA predictions
    y_probs = np.mean(y_probs_all, axis=0)
    y_pred = (y_probs > threshold).astype(int)
    
    if len(y_true_list) > 0:
        y_true = np.vstack(y_true_list)
    else:
        y_true = None
    
    print("TTA completed.")
    
    return y_true, y_pred, y_probs, img_names_list