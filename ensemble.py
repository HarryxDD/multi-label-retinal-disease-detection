import torch
import numpy as np
from tqdm import tqdm


def ensemble_predict(models, test_loader, device, method='average', 
                    weights=None, threshold=0.5):
    """
    Ensemble prediction from multiple models
    
    Args:
        models: List of PyTorch models
        test_loader: Test data loader
        device: cuda or cpu
        method: 'average', 'weighted', 'voting', 'max'
        weights: Weights for weighted average (must sum to 1)
        threshold: Classification threshold
    
    Returns:
        y_pred: Ensemble predictions
        y_probs: Ensemble probabilities
        img_names: Image names
    """
    print(f"\nEnsemble Prediction ({method}) with {len(models)} models")
    
    # Set all models to eval mode
    for model in models:
        model.eval()
    
    # Collect predictions from all models
    all_probs = [[] for _ in range(len(models))]
    img_names_list = []
    
    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader, desc="Ensemble"):
            imgs = imgs.to(device)
            
            # Get predictions from each model
            for i, model in enumerate(models):
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs[i].append(probs)
            
            # Store image names for every batch so lengths match predictions
            img_names_list.extend(names)
    
    # Concatenate all batches for each model
    all_probs = [np.vstack(probs) for probs in all_probs]
    
    # Ensemble methods
    if method == 'average':
        # Simple average
        y_probs = np.mean(all_probs, axis=0)
        
    elif method == 'weighted':
        # Weighted average
        if weights is None:
            raise ValueError("Weights must be provided for weighted ensemble")
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
        
        y_probs = np.zeros_like(all_probs[0])
        for i, w in enumerate(weights):
            y_probs += w * all_probs[i]
        
    elif method == 'voting':
        # Hard voting (majority vote on binary predictions)
        all_preds = [(probs > threshold).astype(int) for probs in all_probs]
        votes = np.sum(all_preds, axis=0)
        y_pred = (votes > len(models) / 2).astype(int)
        y_probs = votes / len(models)  # Convert votes to probabilities
        
        print(f"Voting ensemble complete")
        return y_pred, y_probs, img_names_list
        
    elif method == 'max':
        # Take maximum probability across models
        y_probs = np.max(all_probs, axis=0)
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Apply threshold
    y_pred = (y_probs > threshold).astype(int)
    
    print(f"{method.capitalize()} ensemble complete")
    
    return y_pred, y_probs, img_names_list


def find_optimal_weights(models, val_loader, device, metric='f1'):
    """
    Find optimal weights for weighted ensemble using validation set
    
    Args:
        models: List of models
        val_loader: Validation data loader
        device: cuda or cpu
        metric: 'f1' or 'accuracy'
    
    Returns:
        optimal_weights: Optimal weights for each model
    """
    from sklearn.metrics import f1_score, accuracy_score
    from scipy.optimize import minimize
    
    print("\nFinding optimal ensemble weights...")
    
    # Get predictions from all models
    print("  Collecting predictions from all models...")
    all_probs = [[] for _ in range(len(models))]
    y_true_list = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device)
            
            for i, model in enumerate(models):
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs[i].append(probs)
            
            y_true_list.append(labels.numpy())
    

    all_probs = [np.vstack(probs) for probs in all_probs]
    y_true = np.vstack(y_true_list)
    
    # Objective function to minimize (negative metric)
    def objective(weights):
        weights = weights / weights.sum()
        
        # Weighted ensemble
        y_probs = np.zeros_like(all_probs[0])
        for i, w in enumerate(weights):
            y_probs += w * all_probs[i]
        
        y_pred = (y_probs > 0.5).astype(int)
        
        # Compute metric
        if metric == 'f1':
            # Average F1 across all diseases
            f1_scores = []
            for i in range(y_true.shape[1]):
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                f1_scores.append(f1)
            score = np.mean(f1_scores)
        else:
            score = accuracy_score(y_true.flatten(), y_pred.flatten())
        
        return -score  # Negative for minimization
    
    # Initial weights (equal)
    initial_weights = np.ones(len(models)) / len(models)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in range(len(models))]
    
    # Optimize
    print("  Optimizing weights...")
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / result.x.sum()  # Normalize
    best_score = -result.fun
    
    print(f"\nOptimal weights found!")
    for i, w in enumerate(optimal_weights):
        print(f"    Model {i+1}: {w:.4f}")
    print(f"  Best {metric}: {best_score:.4f}\n")
    
    return optimal_weights

class EnsembleModel:
    """
    Wrapper class for ensemble of models
    """
    def __init__(self, models, method='average', weights=None):
        """
        Args:
            models: List of PyTorch models
            method: Ensemble method
            weights: Weights for weighted average
        """
        self.models = models
        self.method = method
        self.weights = weights
        
    def predict(self, test_loader, device, threshold=0.5):
        return ensemble_predict(self.models, test_loader, device, 
                               self.method, self.weights, threshold)
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def to(self, device):
        self.models = [model.to(device) for model in self.models]
        return self
