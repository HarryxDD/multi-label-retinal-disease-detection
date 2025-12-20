"""
Configuration file for all experiments
Modify these parameters for different tasks
"""

class BaseConfig:
    """Base configuration"""
    # Paths
    TRAIN_CSV = './train.csv'
    VAL_CSV = './val.csv'
    OFFSITE_TEST_CSV = './offsite_test.csv'
    ONSITE_TEST_CSV = './onsite_test_submission.csv'  # Template
    
    TRAIN_IMAGE_DIR = './images/train'
    VAL_IMAGE_DIR = './images/val'
    OFFSITE_TEST_DIR = './images/offsite_test'
    ONSITE_TEST_DIR = './images/onsite_test'
    
    PRETRAINED_RESNET18 = './pretrained_backbone/ckpt_resnet18_ep50.pt'
    PRETRAINED_EFFICIENTNET = './pretrained_backbone/ckpt_efficientnet_ep50.pt'
    
    SAVE_DIR = './checkpoints'
    
    # Dataset
    IMG_SIZE = 256
    NUM_CLASSES = 3
    DISEASE_NAMES = ['D', 'G', 'A']  # Diabetic Retinopathy, Glaucoma, AMD
    SAMPLES_PER_CLASS = [517, 163, 142]  # From training set
    
    # Training
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = 'cuda'  # or 'cpu'
    
    # Random seed for reproducibility
    SEED = 42


class Task1_1_Config(BaseConfig):
    """Task 1.1: No Fine-Tuning"""
    TASK_NAME = 'task1-1'
    BACKBONE = 'efficientnet'  # or 'resnet18'
    
    # No training parameters needed
    TRAIN = False
    LOAD_PRETRAINED = True


class Task1_2_Config(BaseConfig):
    """Task 1.2: Frozen Backbone + Fine-tune Classifier"""
    TASK_NAME = 'task1-2'
    BACKBONE = 'efficientnet'  # Choose based on Task 1.1 results
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = True
    LOAD_PRETRAINED = True
    
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3  # Higher LR for classifier only
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    
    PATIENCE = 7  # Early stopping patience


class Task1_3_Config(BaseConfig):
    """Task 1.3: Full Fine-Tuning"""
    TASK_NAME = 'task1-3'
    BACKBONE = 'efficientnet'
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = False
    LOAD_PRETRAINED = True
    
    NUM_EPOCHS = 50
    
    # Differential learning rates
    BACKBONE_LR = 1e-5  # Smaller for pretrained backbone
    CLASSIFIER_LR = 1e-3  # Larger for classifier
    
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    PATIENCE = 7


class Task2_1_Config(BaseConfig):
    """Task 2.1: Focal Loss"""
    TASK_NAME = 'task2-1'
    BACKBONE = 'efficientnet'  # Use better performing backbone
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = False
    LOAD_PRETRAINED = True
    
    # Loss function
    LOSS_TYPE = 'focal'
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None  # Will be computed from class distribution
    
    NUM_EPOCHS = 50
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    PATIENCE = 7


class Task2_2_Config(BaseConfig):
    """Task 2.2: Class-Balanced Loss"""
    TASK_NAME = 'task2-2'
    BACKBONE = 'efficientnet'
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = False
    LOAD_PRETRAINED = True
    
    # Loss function
    LOSS_TYPE = 'class_balanced'
    CB_BETA = 0.9999  # For very imbalanced data
    CB_GAMMA = 0.5
    
    NUM_EPOCHS = 50
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    PATIENCE = 7


class Task3_1_Config(BaseConfig):
    """Task 3.1: Squeeze-and-Excitation Attention"""
    TASK_NAME = 'task3-1'
    BACKBONE = 'efficientnet'
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = False
    LOAD_PRETRAINED = True
    
    # Attention
    ATTENTION = 'se'
    SE_REDUCTION = 16
    
    # Loss (use best from Task 2)
    LOSS_TYPE = 'focal'  # or 'class_balanced' or 'combined'
    
    NUM_EPOCHS = 50
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    PATIENCE = 7


class Task3_2_Config(BaseConfig):
    """Task 3.2: Multi-Head Attention"""
    TASK_NAME = 'task3-2'
    BACKBONE = 'efficientnet'
    
    # Training
    TRAIN = True
    FREEZE_BACKBONE = False
    LOAD_PRETRAINED = True
    
    # Attention
    ATTENTION = 'mha'
    NUM_HEADS = 8
    
    # Loss (use best from Task 2)
    LOSS_TYPE = 'focal'
    
    NUM_EPOCHS = 50
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = 'adamw'
    SCHEDULER = 'plateau'
    PATIENCE = 7


class Task4_Ensemble_Config(BaseConfig):
    """Task 4: Ensemble (Best approach for maximum score!)"""
    TASK_NAME = 'task4-ensemble'
    
    # Models to ensemble
    MODEL_PATHS = [
        './checkpoints/task1-3_efficientnet.pt',  # Full fine-tuning
        './checkpoints/task2-1_efficientnet_focal.pt',  # Focal loss
        './checkpoints/task2-2_efficientnet_cb.pt',  # Class-balanced
        './checkpoints/task3-1_efficientnet_se.pt',  # SE attention
        './checkpoints/task3-2_efficientnet_mha.pt',  # MHA attention
    ]
    
    MODEL_CONFIGS = [
        {'backbone': 'efficientnet', 'attention': 'none'},
        {'backbone': 'efficientnet', 'attention': 'none'},
        {'backbone': 'efficientnet', 'attention': 'none'},
        {'backbone': 'efficientnet', 'attention': 'se'},
        {'backbone': 'efficientnet', 'attention': 'mha'},
    ]
    
    # Ensemble method
    ENSEMBLE_METHOD = 'weighted'  # 'average', 'weighted', 'voting'
    USE_OPTIMAL_WEIGHTS = True  # Find optimal weights on validation set
    
    # Test-Time Augmentation
    USE_TTA = True
    
    # Threshold optimization
    OPTIMIZE_THRESHOLD = True


def get_config(task='task1-1'):
    """
    Get configuration for specific task
    
    Args:
        task: Task identifier
    
    Returns:
        Config class
    """
    configs = {
        'task1-1': Task1_1_Config,
        'task1-2': Task1_2_Config,
        'task1-3': Task1_3_Config,
        'task2-1': Task2_1_Config,
        'task2-2': Task2_2_Config,
        'task3-1': Task3_1_Config,
        'task3-2': Task3_2_Config,
        'task4': Task4_Ensemble_Config,
    }
    
    if task not in configs:
        raise ValueError(f"Unknown task: {task}")
    
    return configs[task]