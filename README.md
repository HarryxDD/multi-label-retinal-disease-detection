# Navigate to directory
cd final_project_resources

# Task 1.1: No fine-tuning (EfficientNet)
python main.py --task task1-1 --backbone efficientnet

# Task 1.1: No fine-tuning (ResNet18)
python main.py --task task1-1 --backbone resnet18

# Task 1.2: Frozen backbone (EfficientNet)
python main.py --task task1-2 --backbone efficientnet

# Task 1.3: Full fine-tuning (EfficientNet)
python main.py --task task1-3 --backbone efficientnet

# Task 2.1: Focal Loss
python main.py --task task2-1

# Task 2.2: Class-Balanced Loss
python main.py --task task2-2

# Task 3.1: SE Attention
python main.py --task task3-1

# Task 3.2: Multi-Head Attention
python main.py --task task3-2

# Task 4: Ensemble (run after completing other tasks)
python main.py --task task4