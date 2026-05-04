# Project Presentation: Attn-ResAE for Anomalous Sound Detection

**Team members:** Muhammad Anas, Huzaifa Jamil, Nabeel Ahmed  
**Course:** Deep Learning  
**Date:** May 2026  

---

## 1. Problem & Motivation

### The Challenge
- **Anomalous Sound Detection (ASD)**: Identifying machine failure from audio.
- **Unsupervised**: We only have *normal* sound for training. Anomalies are rare and varied.
- **First-Shot Requirement**: Deploy models on *novel* machines without hyperparameter tuning or any anomalous samples.
- **Domain Shift**: Machine conditions (speed, load, environment) change between training (Source) and testing (Target).

---

## 2. DCASE 2023 Task 2 Baseline

### The Official Approach
- **Architecture**: Simple Multi-Layer Perceptron (MLP) Autoencoder.
- **Input**: Mel-spectrogram frames concatenated with a context window ($P=5$).
- **Scoring**: Mean Squared Error (MSE) and Selective Mahalanobis Distance.
- **Reproduced Performance**:
  - Source AUC: 0.6846
  - Target AUC: 0.5312
  - **Total Score (Ω): 0.5720**
- **Observation**: Performance drops significantly in the Target domain due to domain shift.

---

## 3. Proposed Method: Attn-ResAE

### Why Change the Architecture?
The baseline MLP lacks the ability to explicitly model temporal dependencies and can lose fine-grained details during deep compression.

### Key Innovations:
1. **Residual Connections**: Skip connections preserve acoustic features and stabilize training.
2. **Self-Attention Bottleneck**: Multi-head attention ($P=5$) weights the importance of temporal frames before compression.
3. **LeakyReLU & Dropout**: Improves latent space expression and reduces overfitting to specific source characteristics.

---

## 4. Experimental Setup

- **Dataset**: DCASE 2023 Task 2 Development Set (Fan, Gearbox, Bearing, etc.)
- **Training**: 100 Epochs, Adam Optimizer (LR 0.001), PyTorch.
- **Hardware**: GPU-accelerated training for all machine types.
- **Objective**: Improve the quality of the learned latent space to better distinguish normal distributions from anomalous ones.

---

## 5. Results & Analysis

### Performance Comparison (Harmonic Mean)

| Model | Score Type | AUC (Source) | AUC (Target) | pAUC | Total Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline AE | Mahalanobis | 0.6846 | **0.5312** | 0.5260 | 0.5720 |
| **Attn-ResAE** | Mahalanobis | **0.6972** | 0.5263 | **0.5270** | **0.5734** |

### Key Takeaways:
- **Source AUC improved** (+1.26%): The model captures the source distribution much more accurately.
- **Total Score improved**: Attn-ResAE provides a more robust baseline for first-shot tasks.
- **The Trade-off**: Increased capacity leads to slight overfitting, visible in the slight Target AUC drop.

---

## 6. Analysis and Discussion

### Why it works:
- **Attention** effectively isolates the most informative frames in the context window.
- **Skip connections** ensure the reconstruction has high fidelity, leading to better covariance estimates for Mahalanobis scoring.

### Limitations:
- Domain shift remains a major hurdle. 
- High-capacity models tend to "specialize" in the source domain.
- Future work: Integration of outlier exposure or adversarial domain adaptation.

---

## 7. Conclusions

- **Success**: Demonstrated that architectural innovation (Attention + Residuals) can improve upon the standard DCASE baseline.
- **Scalability**: The model remains efficient enough for first-shot deployment.
- **Contribution**: Provided a documented, reproducible pipeline for advanced unsupervised ASD.

---

## 8. Q&A

**Thank you!**  
Questions?
