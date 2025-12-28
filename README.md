# Multi Label Retinal Disease Detection

## Task 1: Transfer Learning

### Task 1.1: No fine-tuning
Run command to build model for task 1
```shell
python vua.py --task task1-1 --backbone efficientnet
```
Backbone could be one of: efficient, resnet18


### Task 1.2: Frozen backbone
```shell
python vua.py --task task1-2 --backbone efficientnet
```
Backbone could be one of: efficient, resnet18

### Task 1.3: Full fine-tuning
```shell
python vua.py --task task1-3 --backbone efficientnet
```
Backbone could be one of: efficient, resnet18

## Task 2: Loss Function
### Task 2.1: Focal Loss
```shell
python vua.py --task task2-1
```

### Task 2.2: Class-Balanced Loss
```shell
python vua.py --task task2-2
```
## Task 3: Attention Mechanisms
### Task 3.1: SE Attention
```shell
python vua.py --task task3-1
```

### Task 3.2: Multi-Head Attention
```shell
python vua.py --task task3-2
```

## Task 4: Ensemble Learning
> Only run this after completing other tasks
```shell
python vua.py --task task4
```