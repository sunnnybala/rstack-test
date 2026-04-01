# Refined Research Idea

## Problem Statement
Standard element-wise dropout is widely used but known to be less effective in convolutional layers due to spatial correlation of activations. Structural alternatives (SpatialDropout, DropBlock) address this by dropping entire feature maps or contiguous regions. However, practitioners lack clear guidance on which dropout type to use and at what rate, because existing papers propose each variant in isolation without controlled head-to-head comparisons on the same architecture. This experiment provides that comparison.

## Proposed Approach
Train a simple CNN (3 conv blocks + 2 FC layers) on CIFAR-10 under a factorial design:

**Dropout types:**
1. **Standard Dropout** — element-wise, the classic approach (Srivastava et al., 2014)
2. **SpatialDropout (2D)** — drops entire feature maps (Tompson et al., 2015)
3. **DropBlock** — drops contiguous spatial regions of feature maps (Ghiasi et al., 2018)

**Dropout rates:** 0.3, 0.5

**Total conditions:** 3 types x 2 rates + 1 no-dropout baseline = 7 conditions
**Seeds:** 3 per condition = 21 total training runs
**Training:** 50 epochs, SGD with momentum 0.9, learning rate 0.01, batch size 128

Dropout is applied after each convolutional block. The FC layers use standard dropout at the same rate in all conditions (to isolate the effect in conv layers).

## What is Novel
- Direct head-to-head comparison of three dropout types on the same simple CNN with identical hyperparameters
- Factorial design that disentangles the effect of dropout type vs dropout rate
- Tests the hypothesis that dropout type matters more than rate for CNNs — a claim implied by the literature but not directly verified in a controlled experiment
- Full training dynamics analysis (not just final accuracy)

## Key Assumptions
- CIFAR-10 is complex enough to reveal differences between dropout strategies
- The simple CNN has sufficient capacity for overfitting to occur (so dropout has something to regularize)
- DropBlock's block_size parameter matters — we use block_size=3 following Ghiasi et al. (2018)

## Evaluation Plan
- **Primary metric:** Test accuracy (mean +/- std over 3 seeds)
- **Secondary metrics:** Convergence speed (epochs to 80% val accuracy), generalization gap (train_acc - val_acc), final train/val loss
- **Analysis:** Two-way ANOVA (type x rate) to test whether type effect > rate effect
- **Baselines:** No-dropout baseline for reference
- **Dataset:** CIFAR-10 (45k train, 5k val, 10k test)

## Risks
- Simple CNN may be too small for meaningful differences between dropout types
- DropBlock requires tuning block_size — our fixed choice of 3 may not be optimal
- 50 epochs may not be enough for full convergence with heavy dropout
- GPU cost: 21 runs, but each is small (CIFAR-10 CNN trains in minutes)

## Next Actions
1. Implement CNN architecture in PyTorch with swappable dropout layer
2. Implement DropBlock (not in standard PyTorch — requires custom implementation or library)
3. Implement training loop with per-epoch metric logging
4. Run 7 conditions x 3 seeds = 21 training runs
5. Generate comparison plots and ANOVA analysis
6. Write up findings
