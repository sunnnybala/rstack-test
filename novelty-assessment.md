# Novelty Assessment: Dropout Type vs Rate in Simple CNNs (Revised)

Generated on 2026-04-01

## Paper-by-Paper Comparison

### Direct Competitors
- **None found** — No paper directly compares Standard Dropout, SpatialDropout, and DropBlock head-to-head on the same architecture with a factorial type x rate design.

### Partial Overlap
- **Srivastava et al. (2014)** — `foundational`. Tested multiple dropout rates but only standard dropout.
- **Wu & Gu (2015)** — `partial_overlap`. Compared dropout strategies in CNNs but with different variants (max-pooling dropout, FC dropout) and different experimental setup.
- **Dropout Surveys (2019, 2022, 2023)** — `partial_overlap`. Discuss all three variants and provide guidelines, but do not run controlled head-to-head experiments.
- **Dropout vs Model Complexity (2021)** — `partial_overlap`. Studies rate selection but only for standard dropout.

### Complementary
- **DropBlock (2018)** — `complementary`. Proposes DropBlock and compares against standard dropout, but on large-scale models (ResNet-50 on ImageNet), not simple CNNs. Does not include SpatialDropout.
- **Tompson et al. (2015)** — `complementary`. Proposes SpatialDropout but does not compare against DropBlock (which didn't exist yet).
- **Dropout Reduces Underfitting (2023)** — `complementary`. Different dimension (temporal scheduling).

## What is Genuinely Novel
- **Controlled factorial comparison**: No existing paper runs Standard Dropout, SpatialDropout, and DropBlock at multiple rates on the same CNN with identical hyperparameters.
- **Type vs rate disentanglement**: The factorial design (3 types x 2 rates) allows ANOVA analysis to formally test whether type or rate has a larger effect — a question implied but never directly answered.
- **Simple CNN focus**: Most dropout variant papers test on large models (ResNet, WideResNet). Testing on a simple CNN provides practical guidance for smaller projects.

## What Overlaps with Existing Work
- Each dropout type has been proposed and evaluated individually
- DropBlock paper (2018) compares against standard dropout (but not SpatialDropout, and on ImageNet-scale models)
- Surveys describe all three but rely on results from separate papers with different experimental setups

## What Gaps This Addresses
- Direct, controlled comparison of dropout types on the same architecture
- Interaction between type and rate (does optimal rate differ by type?)
- Practical guidance for practitioners using simple CNNs

## Novelty Score: 5/10

This is an incremental but useful empirical contribution. The factorial design comparing dropout types against rates in a controlled setting fills a genuine gap. No single paper has done this comparison with modern variants. While each component is well-known, the combination and the specific question ("does type matter more than rate?") is novel enough for a workshop paper or empirical study.
