# Project Submission Index (Task 4 Deliverables)

This document serves as a guide for the submitted code repository and experimental artifacts.

## 1. Code Repository
- **GitHub Repository**: [https://github.com/AnasA18H/DCASE-2023_Antt_Resnt](https://github.com/AnasA18H/DCASE-2023_Antt_Resnt)
- **Main Implementation**: [Task_3/code/networks/dcase2023t2_ae/network.py](https://github.com/AnasA18H/DCASE-2023_Antt_Resnt/blob/main/networks/dcase2023t2_ae/network.py) (Attn-ResAE Model)
- **Training Script**: `train.py` / `train_ae.sh`
- **Inference Script**: `test_network.py` / `test_ae.sh`

## 2. Experiment Logs
Detailed training and evaluation logs are provided to ensure transparency and reproducibility.
- **Master Log**: `Task_3/code/baseline.log` (Contains full execution trace)
- **Machine-Specific Logs**: `Task_3/code/logs/` (Directory containing individual machine training logs)
- **Performance Summaries**: `Task_3/code/results/` (CSV files containing AUC/pAUC for each machine type)

## 3. Dataset Preprocessing Scripts
Preprocessing logic is integrated into the training pipeline for end-to-end reproducibility.
- **Audio Loading & Mel-Spectrogram Conversion**: [Task_3/code/common.py](https://github.com/AnasA18H/DCASE-2023_Antt_Resnt/blob/main/common.py)
- **Data Augmentation & Batching**: [Task_3/code/train.py](https://github.com/AnasA18H/DCASE-2023_Antt_Resnt/blob/main/train.py)

## 4. Final Deliverables
- **Final Research Report**: `Task_4/Final_Report.md` (10-12 pages content)
- **Presentation Slides**: `Task_4/Presentation.md`
- **Model Checkpoints**: Located in `Task_3/code/models/` (One `.pth` file per machine section)

---
*Prepared by Team 04 for the Deep Learning Course.*
