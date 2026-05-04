# DCASE 2023 Task 2: Attention-Enhanced Residual Autoencoder (Attn-ResAE)

[![DCASE 2023](https://img.shields.io/badge/DCASE-2023-blue.svg)](https://dcase.community/challenge2023/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains the implementation of the **Attention-Enhanced Residual Autoencoder (Attn-ResAE)**, a modified architecture designed to improve "First-Shot" Unsupervised Anomalous Sound Detection (ASD) under domain-shifted conditions. This project was developed for the DCASE 2023 Challenge Task 2.

## 🚀 Key Features
- **Residual Blocks (`ResBlock`)**: Implements skip connections to preserve fine-grained acoustic features and mitigate vanishing gradients in deep layers.
- **Self-Attention Bottleneck**: Integrates a Multi-Head Self-Attention layer (`nn.MultiheadAttention`) to dynamically weigh temporal dependencies among concatenated input frames.
- **Improved Regularization**: Uses LeakyReLU activations and Dropout (0.2) to enhance generalization and prevent dead neurons.
- **Mahalanobis Distance Scoring**: Fully compatible with the DCASE baseline Selective Mahalanobis distance scoring for robust anomaly detection.

## 📊 Performance Summary
Our Attn-ResAE achieves a superior **Source Domain AUC** compared to the official DCASE 2023 baseline, resulting in a higher overall harmonic mean score.

| Model | Scoring Mode | AUC (Source) | AUC (Target) | pAUC | Total Score (Ω) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline AE | Mahalanobis | 0.6846 | **0.5312** | 0.5260 | 0.5720 |
| **Attn-ResAE (Ours)** | Mahalanobis | **0.6972** | 0.5263 | **0.5270** | **0.5734** |

## 🛠 Installation

### Prerequisites
- Ubuntu 22.04+
- Python 3.11+
- CUDA 11.8+

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AnasA18H/DCASE-2023_Antt_Resnt.git
   cd DCASE-2023_Antt_Resnt
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset Preparation
This project uses the **DCASE 2023 Task 2 Development Dataset**. 
1. Download the dataset from [Zenodo](https://zenodo.org/record/7882613).
2. Unzip and place the data in the following structure:
   ```text
   data/
   └── dcase2023t2/
       └── dev_data/
           └── raw/
               ├── fan/
               ├── gearbox/
               └── ...
   ```

## 🏃 Usage

### 1. Training
To train the Attn-ResAE across all machine types:
```bash
./train_ae.sh
```
Or run for specific sections using `train.py`:
```bash
python train.py -d
```

### 2. Testing & Evaluation
To calculate anomaly scores and generate the `result.csv`:
```bash
./test_ae.sh
```

### 3. Summarize Results
```bash
./03_summarize_results.sh DCASE2023T2 -d
```

## 🏗 Architecture
The Attn-ResAE replaces the standard MLP layers of the baseline with a more sophisticated structure:
1. **Encoder**: 4x Residual Blocks (Linear -> BatchNorm -> LeakyReLU -> Dropout).
2. **Attention Layer**: Multi-Head Self-Attention applied to the temporal context window ($P=5$).
3. **Bottleneck**: Compressed 8-dimensional latent representation.
4. **Decoder**: Mirror of the encoder to reconstruct the input mel-spectrogram bins.

## 📜 Citation & References
If you use this work, please cite the following:
- **DCASE 2023 Baseline**: Dohi, K., et al. (2023). "Description and Discussion on DCASE 2023 Challenge Task 2."
- **Attention Mechanism**: Vaswani, A., et al. (2017). "Attention Is All You Need."
- **Residual Learning**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition."

## 👥 Contributors
- **Muhammad Anas** (i221987)
- **Huzaifa Jamil** (i22-1899)
- **Nabeel Ahmed** (22i-2040)

---
*Note: This project was conducted as part of the Deep Learning Course assignment.*
