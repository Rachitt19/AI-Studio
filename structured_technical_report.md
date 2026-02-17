# structured_technical_report.md

## 1. Experimental Setup
We evaluated the performance, scalability, and robustness of small-scale convolutional neural networks (CNNs) on a 10,000-sample subset of CIFAR-10. The goal was to characterize learning dynamics under data-constrained regimes.
- **Dataset:** CIFAR-10 Subset (8k Train / 2k Val).
- **Core Architecture:** 3-layer CNN (Conv-Pool-Conv-Pool-FC-FC) with ReLU activations.
- **Optimization:** Adam (LR=0.001), CrossEntropyLoss, 15 Epochs.
- **Environment:** Reproducible seed (42), Metal Performance Shaders (MPS) acceleration.

## 2. Baseline Model Performance
The baseline model (Model B: 32-64 filters) achieved:
- **Final Validation Accuracy:** 59.90%
- **Final Train Accuracy:** 97.41%
- **Generalization Gap:** ~37.5%

**Observation:** The model exhibited classic overfitting characteristics, achieving near-perfect training accuracy while validation accuracy plateaued early (around Epoch 4-5). This confirms the model has sufficient representational capacity but is severely data-limited.

## 3. Capacity Scaling Analysis
We compared three model variants to isolate the effect of parameter count on over-parameterization and generalization.

| Model | Parameters | Train Acc | Val Acc | Gen Gap | Overfit Onset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Model A (Small) | 1,059,306 | 99.64% | 59.70% | 39.94% | Epoch 5 |
| **Model B (Baseline)** | **2,122,186** | **97.41%** | **59.90%** | **37.51%** | **Epoch 4** |
| Model C (Large) | 4,275,594 | 98.81% | 58.80% | 40.01% | Epoch 5 |

**Findings:**
1.  **Diminishing Returns:** Quadrupling parameters (Model A to C) yielded negligible improvement (-0.9% validation accuracy). This strongly suggests the task is **data-bound**, not capacity-bound.
2.  **Overfitting Dynamics:** All models, regardless of size, began overfitting almost immediately (Epoch 4-5). Larger models did not learn "better" features; they merely memorized the training set faster.

## 4. Effect of Data Augmentation
We introduced `RandomHorizontalFlip` and `RandomCrop` to the Baseline Model training pipeline.

- **Baseline Val Acc:** 59.90%
- **Augmented Val Acc:** 66.15% (+6.25%)
- **Convergence:** Slower. At Epoch 5, Augmented model was at ~55% (vs. ~58% Baseline), but continued to improve steadily without the sharp overfitting plateau seen in the non-augmented runs.
- **Interpretation:** Augmentation successfully acted as a regularizer, forcing the model to learn invariant features rather than memorizing pixel-perfect patterns. The significant accuracy boost confirms that lack of data diversity was the primary bottleneck.

## 5. Robustness Evaluation
We tested the Baseline Model's resilience to Gaussian noise ($\sigma=0.05$) on the validation set.

- **Clean Accuracy:** 59.90%
- **Noisy Accuracy:** 59.70%
- **Drop:** 0.20%

**Interpretation:** The model is unexpectedly robust to low-magnitude independent and identically distributed (i.i.d.) noise. A <1% drop indicates the learned features (likely edges and simple shapes) are stable against high-frequency perturbations, unlike comparable adversarial attacks.

## 6. Failure Mode Analysis
**Confusion Matrix Insights:**
- **Top Confusion Pair:** Dog / Cat
- **Class-wise Accuracy:**
    - **High:** Frog (79.9%), Automobile (70.1%), Ship (74.7%).
    - **Low:** Cat (33.9%), Bird (46.0%), Dog (46.5%).

**Reasoning:**
- **Texture vs. Shape:** The model excels at rigid objects with distinct silhouettes (Cars, Ships). It struggles heavily with organic, deformable objects (Cats, Dogs, Birds) where texture and pose variance are high.
- **Resolution Limit:** At 32x32 resolution, the distinguishing features between a Cat and a Dog (e.g., snout shape) are often lost to downsampling, leading to high inter-class confusion among quadrupeds.

## 7. Deployment & Applied Reasoning
**2B Stage Analysis / Edge Implications:**
1.  **Model Size vs. Utility:** The "Small" model (Model A, ~1MB) performed identically to the "Large" model (Model C, ~4MB). For edge deployment (e.g., microcontroller or embedded DSP), preferring the smaller architecture saves 75% memory with zero accuracy penalty.
2.  **Robustness in the Wild:** The stability against Gaussian noise suggests these simple models might handle sensor thermal noise well, but testing against motion blur or compression artifacts is required for real-world video feed reliability.
3.  **Low-Resource Strategy:** Given the data constraints, investing in compute for larger models is wasteful. Resources should instead be allocated to data curation (augmentation, collection) or transfer learning from compressed, pre-trained backbones (e.g., MobileNetV3-Small) to break the 60% accuracy ceiling.

---
**Artifacts Produced:**
- `cifar_research.py`: Reproducible experimental code.
- `capacity_study.csv`: Raw metrics for model scaling.
- `learning_curves.png`: Loss/Accuracy visualization.
- `confusion_matrix.png`: Failure analysis heatmap.
