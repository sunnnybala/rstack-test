# Research Idea: Dropout Type vs Rate in Simple CNNs

## Hypothesis
In convolutional neural networks, the *type* of dropout (Standard, SpatialDropout, DropBlock) matters more than the *rate* (0.3 vs 0.5) for regularization effectiveness. We test this with a factorial design on CIFAR-10.

## Motivation
Standard element-wise dropout is ineffective in convolutional layers because spatially correlated features compensate for dropped pixels (Tompson et al., 2015; Ghiasi et al., 2018). SpatialDropout and DropBlock address this by dropping entire feature maps or contiguous regions. While each variant has been proposed in isolation, direct head-to-head comparisons on the same architecture with identical settings are rare. This experiment tests whether practitioners should prioritize choosing the right dropout *type* over tuning the dropout *rate*.

## Approach
- Factorial design: 3 dropout types (Standard, SpatialDropout, DropBlock) x 2 rates (0.3, 0.5) + no-dropout baseline = 7 conditions
- Simple CNN on CIFAR-10, 3 seeds per condition
- Track training dynamics: loss curves, accuracy, convergence speed, generalization gap

## Expected Outcomes
- SpatialDropout and DropBlock will outperform standard dropout at the same rate
- The performance gap between dropout types will be larger than the gap between rates
- Optimal rate may differ by dropout type (standard dropout may prefer 0.5, structured variants may prefer 0.3)
