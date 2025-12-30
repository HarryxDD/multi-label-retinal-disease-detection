"""
Stacking Ensemble: Train a meta-learner on base model predictions
This typically gives +2-3% over simple averaging
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm


class StackingMetaLearner(nn.Module):
    """Simple neural network meta-learner for stacking"""
    def __init__(self, num_models, num_classes):
        super(StackingMetaLearner, self).__init__()
        self.fc1 = nn.Linear(num_models * num_classes, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, num_models * num_classes)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_base_predictions(models, dataloader, device):
    """Get predictions from all base models"""
    all_probs = [[] for _ in range(len(models))]
    all_labels = []
    all_names = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for imgs, labels, names in tqdm(dataloader, desc="Collecting base predictions"):
            imgs = imgs.to(device)
            
            for i, model in enumerate(models):
                outputs = model(imgs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs[i].append(probs)
            
            all_labels.append(labels.numpy())
            all_names.extend(names)
    
    # Concatenate all batches
    all_probs = [np.vstack(probs) for probs in all_probs]
    all_labels = np.vstack(all_labels)
    
    # Stack predictions: (N, num_models, num_classes) -> (N, num_models * num_classes)
    stacked_probs = np.hstack(all_probs)
    
    return stacked_probs, all_labels, all_names


def train_stacking_sklearn(models, train_loader, val_loader, device):
    """Train stacking ensemble using sklearn LogisticRegression (fast & effective)"""
    print("\Training Stacking Ensemble")
    
    # Get predictions from base models
    print("Getting train predictions...")
    X_train, y_train, _ = get_base_predictions(models, train_loader, device)
    
    print("Getting validation predictions...")
    X_val, y_val, _ = get_base_predictions(models, val_loader, device)
    
    # Train a separate logistic regression for each class
    meta_models = []
    print("\nTraining meta-learners...")
    for i, disease in enumerate(['D', 'G', 'A']):
        print(f"  Training meta-learner for {disease}...")
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        meta_model.fit(X_train, y_train[:, i])
        
        # Validate
        val_pred = meta_model.predict(X_val)
        f1 = f1_score(y_val[:, i], val_pred)
        print(f"    Validation F1: {f1:.4f}")
        
        meta_models.append(meta_model)
    
    # Overall validation F1
    val_preds_all = np.column_stack([m.predict(X_val) for m in meta_models])
    overall_f1 = f1_score(y_val, val_preds_all, average='macro')
    print(f"\nOverall Validation F1: {overall_f1:.4f}")
    
    return meta_models


def train_stacking_nn(models, train_loader, val_loader, device, num_epochs=20):
    """Train stacking ensemble using neural network meta-learner"""
    print("\n=== Training Stacking Ensemble (Neural Network) ===")
    
    # Get predictions from base models
    print("Getting train predictions...")
    X_train, y_train, _ = get_base_predictions(models, train_loader, device)
    
    print("Getting validation predictions...")
    X_val, y_val, _ = get_base_predictions(models, val_loader, device)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    # Create meta-learner
    num_models = len(models)
    num_classes = y_train.shape[1]
    meta_model = StackingMetaLearner(num_models, num_classes).to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
    
    best_f1 = 0
    best_state = None
    
    print("\nTraining meta-learner...")
    for epoch in range(num_epochs):
        # Train
        meta_model.train()
        optimizer.zero_grad()
        outputs = meta_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validate
        meta_model.eval()
        with torch.no_grad():
            val_outputs = meta_model(X_val)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(y_val.cpu().numpy(), val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = meta_model.state_dict()
    
    # Load best model
    meta_model.load_state_dict(best_state)
    print(f"\nBest Validation F1: {best_f1:.4f}")
    
    return meta_model


def stacking_predict(base_models, meta_models, test_loader, device, use_nn=False):
    """Make predictions using stacking ensemble"""
    print("\nStacking Ensemble Prediction")
    
    # Get base model predictions
    X_test, _, img_names = get_base_predictions(base_models, test_loader, device)
    
    if use_nn:
        # Neural network meta-learner
        meta_models.eval()
        X_test = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            outputs = meta_models(X_test)
            y_probs = torch.sigmoid(outputs).cpu().numpy()
    else:
        # Sklearn meta-learners
        y_probs = np.column_stack([
            m.predict_proba(X_test)[:, 1] for m in meta_models
        ])
    
    y_pred = (y_probs >= 0.5).astype(int)
    
    return y_pred, y_probs, img_names


def optimize_stacking_thresholds(base_models, meta_models, val_loader, device, use_nn=False):
    """Find optimal thresholds for each disease in stacking ensemble"""
    print("\nOptimizing Stacking Thresholds")
    
    # Get predictions
    X_val, y_val, _ = get_base_predictions(base_models, val_loader, device)
    
    if use_nn:
        meta_models.eval()
        X_val = torch.FloatTensor(X_val).to(device)
        with torch.no_grad():
            outputs = meta_models(X_val)
            y_probs = torch.sigmoid(outputs).cpu().numpy()
    else:
        y_probs = np.column_stack([
            m.predict_proba(X_val)[:, 1] for m in meta_models
        ])
    
    # Find optimal threshold for each disease
    optimal_thresholds = []
    disease_names = ['D', 'G', 'A']
    
    for i, disease in enumerate(disease_names):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.7, 0.05):
            preds = (y_probs[:, i] >= threshold).astype(int)
            f1 = f1_score(y_val[:, i], preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"  {disease}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    
    return optimal_thresholds
