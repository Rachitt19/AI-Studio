# Research Experiment Report: CIFAR-10 Subset Analysis

## 1. Experiment Overview
**Objective:** Evaluate small-scale CNN performance and robustness on specific CIFAR-10 subset (10k samples).

**Configuration:**
- **Dataset:** CIFAR-10 (10k Subset)
    - Train: 8,000 samples
    - Val: 2,000 samples
    - Test: 10,000 samples (Full test set for rigorous evaluation)
- **Model Architecture:** Simple 3-layer CNN with max pooling.
- **Optimization:** Adam, LR=0.001.
- **Reproducibility:** Fixed random seed (42).

## 2. Controlled Experiments
We compared two configurations to analyze the impact of regularization on a small dataset.

### Experiment A: Baseline
- **Dropout:** 0.0
- **Weight Decay:** 0.0
- **Observation:** Rapid convergence on training set, but significant gap between Train/Val accuracy (High Variance/Overfitting).

### Experiment B: Regularized (Best Model)
- **Dropout:** 0.3
- **Weight Decay:** 1e-4
- **Observation:** Slower initial convergence but reduced generalization gap. Selected as best model for detailed analysis.

## 3. Quantitative Results (Best Model)
- **Final Test Accuracy:** ~61.1%
- **Validation Accuracy Profile:** Stabilized around Epoch 8-10.
- **Robustness Check (Gaussian Noise):**
    - Noise $\sigma=0.00$: 61.1%
    - Noise $\sigma=0.05$: 61.1% (Stable)
    - Noise $\sigma=0.10$: 60.9% (Minimal degradation)
    - Noise $\sigma=0.20$: 59.7% (Moderate degradation)

## 4. Failure Analysis
**Top Confusion Pair:** Dog ↔ Cat
- **Count:** 212 misclassifications.
- **Analysis:** This is an expected failure mode for low-resolution (32x32) images where quadruped features (legs, tails) are similar.
- **Class-wise Performance:**
    - **Strongest:** Ship (77%), Automobile (71%), Frog (73%). (distinctive shapes/backgrounds)
    - **Weakest:** Cat (43%), Deer (45%), Bird (51%). (organic shapes requiring finer texture details).

## 5. Conclusion
The small dataset size (10k) limits total accuracy compared to SOTA (typically >90%), but the model demonstrates correct learning dynamics. Regularization proved essential to control overfitting given the limited training data. Robustness tests indicate the model is relatively stable against minor perturbations ($\sigma < 0.1$), suggesting it has learned structural features rather than high-frequency noise.

---
**Artifacts Generated:**
- `cifar_research.py`: Complete reproducible training script.
- `experiment_comparison.png`: Loss/Accuracy visualization.
