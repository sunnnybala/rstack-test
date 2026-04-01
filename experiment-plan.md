# Experiment Plan

## Objective
Test whether dropout *type* (Standard, SpatialDropout, DropBlock) matters more than dropout *rate* (0.3 vs 0.5) for regularization effectiveness in a simple CNN on CIFAR-10.

## Dataset
- **Name:** CIFAR-10
- **Source:** torchvision.datasets.CIFAR10 (auto-download)
- **Size:** 50,000 training + 10,000 test images, 32x32 RGB, 10 classes
- **Split:** 45,000 train / 5,000 validation / 10,000 test

## Model Architecture
Simple CNN with 3 convolutional blocks + 2 fully connected layers:
- Conv Block 1: Conv2d(3, 32, 3) → BatchNorm → ReLU → [Dropout Layer] → MaxPool2d(2)
- Conv Block 2: Conv2d(32, 64, 3) → BatchNorm → ReLU → [Dropout Layer] → MaxPool2d(2)
- Conv Block 3: Conv2d(64, 128, 3) → BatchNorm → ReLU → [Dropout Layer] → MaxPool2d(2)
- FC1: Linear(128*2*2, 256) → ReLU → Standard Dropout (same rate)
- FC2: Linear(256, 10)

[Dropout Layer] varies by condition:
- Standard: nn.Dropout2d (element-wise on feature maps)
- SpatialDropout: nn.Dropout2d (drops entire channels)
- DropBlock: Custom DropBlock2D (block_size=3)

## Training Protocol
- **Optimizer:** SGD, momentum=0.9, weight_decay=1e-4
- **Learning rate:** 0.01 with CosineAnnealingLR
- **Batch size:** 128
- **Epochs:** 50
- **Data augmentation:** RandomCrop(32, padding=4), RandomHorizontalFlip

## Conditions (7 total)
1. No dropout (baseline)
2. Standard Dropout, rate=0.3
3. Standard Dropout, rate=0.5
4. SpatialDropout, rate=0.3
5. SpatialDropout, rate=0.5
6. DropBlock, rate=0.3 (block_size=3)
7. DropBlock, rate=0.5 (block_size=3)

Seeds: 42, 123, 456 (3 per condition = 21 runs total)

## Evaluation Metrics
- **Primary:** Test accuracy (mean ± std over 3 seeds)
- **Secondary:** Convergence speed (epochs to 80% val acc), generalization gap (train_acc - val_acc), final val loss
- **Analysis:** Two-way ANOVA (type × rate) on test accuracy

## Estimated Runtime
~3-5 minutes per run on RTX 4070 × 21 runs ≈ 60-100 minutes total
