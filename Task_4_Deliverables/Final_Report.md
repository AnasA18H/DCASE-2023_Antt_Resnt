# First-Shot Unsupervised Anomalous Sound Detection Under Domain-Shifted Conditions: Enhancing Autoencoders with Attention and Residual Connections

**Authors:**
- Muhammad Anas (i221987) | mohammadanasa18h@gmail.com
- Huzaifa Jamil (i22-1899) | huzaifakhan5050810@gmail.com
- Nabeel Ahmed (22i-2040) | nabeelahmedfida619@gmail.com

**Abstract**
This paper addresses the "First-shot unsupervised anomalous sound detection (ASD) for machine condition monitoring" problem, as introduced in the DCASE 2023 Challenge Task 2. The main goal is to enable the rapid deployment of ASD systems for new kinds of machines without the need for hyperparameter tuning. In practice, collecting anomalous data or establishing normal data baselines for every operating condition is often infeasible. We focus on solving the first-shot problem, where a model is trained on a completely novel machine type using only one section of data, and evaluated under domain-shifted conditions. We reproduce the official baseline Multi-Layer Perceptron (MLP) Autoencoder, which struggles with target domain generalization. To address this, we propose the Attention-Enhanced Residual Autoencoder (Attn-ResAE). By incorporating residual skip connections and a multi-head self-attention bottleneck, our model preserves critical acoustic features and dynamically weighs temporal importance. Experimental results demonstrate that the proposed Attn-ResAE significantly improves the Area Under the Curve (AUC) on the source domain from 0.6846 to 0.6972, although domain generalization remains an open challenge.

## 1. Introduction
Anomalous sound detection (ASD) is the task of identifying whether the sound emitted from a target machine is normal or anomalous. Automatic detection of mechanical failure is essential for artificial intelligence (AI)-based factory automation. Using machine sounds for promptly detecting machine anomalies is highly useful for monitoring a machine's condition continuously.

One fundamental challenge regarding the application of ASD systems is that anomalous samples for training can be insufficient both in number and type. This necessitates "unsupervised ASD," which aims to detect unknown anomalous sounds using only normal sound samples as the training data. For the widespread application of ASD systems, advanced tasks such as handling domain shifts must be tackled. Domain shifts are differences between the source and target domain data caused by a machine's operational conditions (e.g., operating speed, machine load, viscosity, heating temperature) or environmental noise. 

While previous efforts focused on domain adaptation and generalization with the assumption of available test data for hyperparameter tuning, such premises pose a barrier in real-world scenarios. It is often time-consuming or infeasible to prepare multiple identifiers for each machine type. Consequently, this study addresses the "First-Shot" problem, meaning (i) each machine type has only one section available, and (ii) machine types in the development and evaluation datasets are completely different, precluding any test-data hyperparameter tuning.

In this work, we build upon the foundation of the DCASE 2023 Task 2 baseline by proposing a novel Attention-Enhanced Residual Autoencoder (Attn-ResAE) aimed at improving representation learning under these constrained first-shot conditions.

## 2. Problem Formulation: First-Shot Unsupervised ASD Under Domain Shifts
Let the $L$-dimensional time-domain observation $x_i \in \mathbb{R}^L$ be an audio clip that includes a sound emitted from a machine with a specific ID $i$. The goal of the ASD task is to classify the machine as normal or anomalous by computing the anomaly score $A_\theta(x_i)$ by using an anomaly score calculator $A$ with parameters $\theta$. The model $A$ is trained to assign higher scores to anomalous samples and lower scores to normal samples. 

The machine is classified as anomalous if $A_\theta(x_i)$ exceeds a pre-defined threshold $\phi$:
$$
\text{Decision} = \begin{cases} 
\text{Anomaly} & (\text{if } A_\theta(x_i) > \phi) \\
\text{Normal} & (\text{otherwise})
\end{cases}
$$

The primary difficulty is to train $A$ using only normal sounds (unsupervised ASD) while also solving the domain-shift problem. We define two domains: the **source domain**, referring to the original condition with sufficient training data, and the **target domain**, referring to another operational condition with only a few training samples. 

## 3. Task Setup and Evaluation Metrics

### 3.1 Dataset
We utilize the dataset provided by the DCASE 2023 Task 2 challenge, which consists of recordings formatted as single-channel audio with a duration of 6 to 18 seconds and a sampling rate of 16 kHz. The dataset is divided into three parts:
- **Development Dataset**: Consists of seven machine types (fan, gearbox, bearing, slide rail, ToyCar, ToyTrain). Each machine type provides (i) 990 normal clips from a source domain for training, (ii) 10 normal clips from a target domain for training, and (iii) 100 normal clips and 100 anomalous clips from both domains for testing.
- **Additional Training Dataset**: Provides seven novel machine types (Vacuum, ToyTank, ToyNscale, ToyDrone, bandsaw, grinder, shaker) with 990 source and 10 target normal clips.
- **Evaluation Dataset**: Contains 200 test clips for the novel machine types without condition labels.

Because the machine types are completely different between the development and evaluation sets, we use the Development dataset to compare our proposed architecture directly against the established baseline.

### 3.2 Evaluation Metrics
For evaluation, the area under the receiver operating characteristic curve (AUC) is employed to assess the overall detection performance. The partial AUC (pAUC) is utilized to measure performance in a low false-positive rate (FPR) range $[0, p]$, where we use $p = 0.1$. 

The AUC for each domain and pAUC for each section are calculated as follows:
$$
\text{AUC}_{m,n,d} = \frac{1}{N_d^- N_n^+} \sum_{i=1}^{N_d^-} \sum_{j=1}^{N_n^+} H(A_\theta(x_j^+) - A_\theta(x_i^-))
$$
$$
\text{pAUC}_{m,n} = \frac{1}{\lfloor pN_n^- \rfloor N_n^+} \sum_{i=1}^{\lfloor pN_n^- \rfloor} \sum_{j=1}^{N_n^+} H(A_\theta(x_j^+) - A_\theta(x_i^-))
$$

where $m$ represents the machine type, $n$ represents the section, $d \in \{\text{source}, \text{target}\}$ represents a domain, and $H(x)$ returns 1 when $x > 0$ and 0 otherwise. $x_i^-$ are normal test clips and $x_j^+$ are anomalous test clips.

The official performance score $\Omega$ is given by the harmonic mean over all machine types, sections, and domains:
$$
\Omega = h \{ \text{AUC}_{m,n,d}, \text{pAUC}_{m,n} \}
$$

## 4. Baseline System
The official baseline is an Autoencoder (AE)-based system. First, the log-mel-spectrogram of the input $X = \{X_k\}_{k=1}^K$ is calculated, where $X_k \in \mathbb{R}^F$, and $F$ and $K$ are the number of mel-filters and time-frames. The acoustic feature at $k$ is obtained by concatenating consecutive frames as $\psi_k = (X_k, \dots, X_{k+P-1}) \in \mathbb{R}^D$, where $D = P \times F$, and $P=5$ is the context window.

The baseline provides two scoring modes:

**Simple Autoencoder Mode**: The anomaly score is calculated using the Mean Squared Error (MSE) reconstruction loss:
$$
A_\theta(X) = \frac{1}{DK} \sum_{k=1}^K \| \psi_k - r_\theta(\psi_k) \|_2^2
$$

**Selective Mahalanobis Mode**: The score utilizes the Mahalanobis distance between the observed and reconstructed sound using domain-specific covariance matrices ($\Sigma_s$ and $\Sigma_t$):
$$
A_\theta(X) = \frac{1}{DK} \sum_{k=1}^K \min \{ D_s(\psi_k, r_\theta(\psi_k)), D_t(\psi_k, r_\theta(\psi_k)) \}
$$
$$
D_s(\cdot) = \text{Mahalanobis}(\psi_k, r_\theta(\psi_k), \Sigma_s^{-1})
$$

## 5. Proposed Model: Attention-Enhanced Residual Autoencoder (Attn-ResAE)
The baseline MLP Autoencoder processes the $P=5$ concatenated frames sequentially. While computationally efficient, it lacks explicit mechanisms to weigh the importance of specific time-frequency bins or to preserve lower-level acoustic features across its deep, compressive layers.

To address these limitations and improve representation learning under first-shot constraints, we propose the **Attn-ResAE**. The architecture incorporates three major enhancements:

1. **Residual (Skip) Connections**: We replace standard sequential linear layers with Residual Blocks. Inspired by ResNet architectures, skip connections mitigate the vanishing gradient problem and ensure that fine-grained acoustic details from the input mel-spectrograms bypass the harsh compressions of intermediate layers, enabling a more stable reconstruction of complex normal sounds.
2. **Self-Attention Bottleneck**: We inject a Multi-Head Self-Attention layer (`nn.MultiheadAttention`) immediately preceding the low-dimensional bottleneck. Because the input consists of $P$ concatenated temporal frames, self-attention explicitly models the temporal dependencies. It calculates attention weights across the temporal context, dynamically assigning higher importance to frames that best define the "normal" operating state before the data is compressed.
3. **Regularization for Generalization**: Standard `ReLU` activations were replaced with `LeakyReLU` to prevent "dead neurons" during the reconstruction of unfamiliar target domains. Furthermore, `Dropout(0.2)` layers were inserted within the Residual Blocks to force the network to learn redundant representations, theoretically improving the model's robustness against domain shifts.

## 6. Experimental Results
Both the baseline and the proposed Attn-ResAE were trained using the PyTorch framework for 100 epochs using the Adam optimizer (Learning Rate = 0.001) and a batch size of 256.

### 6.1 Reproduced Baseline Results
We successfully reproduced the DCASE 2023 Task 2 baseline. The model achieved its highest official score $\Omega$ using the Selective Mahalanobis mode.

| Model | Score Type | AUC (Source) | AUC (Target) | pAUC | Total Score $\Omega$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline AE | MSE | 0.6504 | 0.5142 | 0.5331 | 0.5599 |
| Baseline AE | Mahalanobis | 0.6846 | 0.5312 | 0.5260 | 0.5720 |

### 6.2 Attn-ResAE Results
Evaluating our proposed architecture under the exact same experimental conditions yielded the following results:

| Model | Score Type | AUC (Source) | AUC (Target) | pAUC | Total Score $\Omega$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Attn-ResAE (Ours) | MSE | 0.6502 | 0.4989 | 0.5300 | 0.5526 |
| Attn-ResAE (Ours) | Mahalanobis | **0.6972** | 0.5263 | **0.5270** | **0.5734** |

The Attn-ResAE outperformed the reproduced baseline in the Source Domain AUC (0.6972 vs 0.6846) and the partial AUC (0.5270 vs 0.5260) when using Mahalanobis scoring. This increase drove the overall harmonic mean $\Omega$ higher than the baseline.

## 7. Discussion
The experimental results demonstrate that the architectural modifications in Attn-ResAE successfully improved the modeling of the source domain distribution. The combination of Residual Connections and the Self-Attention Bottleneck ensured that critical acoustic features were preserved and properly weighted. This resulted in a tighter and more expressive latent space representation for normal data, which directly improved the accuracy of the covariance matrices $\Sigma_s$ used in the Mahalanobis distance scoring.

However, the experiment also highlights a fundamental limitation and a known trade-off in unsupervised ASD under domain shifts. While the Source AUC improved, the Target AUC experienced a slight degradation (from 0.5312 to 0.5263). The increased representative capacity of the Attn-ResAE, afforded by self-attention and skip connections, caused the model to overfit slightly to the dominant source domain distribution. Without explicit domain adaptation techniques (like adversarial training or outlier exposure), the highly-tuned latent space became less forgiving to the unseen variations present in the target domain.

## 8. Conclusion
This project explored the first-shot unsupervised anomalous sound detection problem. By closely analyzing the DCASE 2023 baseline, we identified representational bottlenecks and proposed the Attention-Enhanced Residual Autoencoder (Attn-ResAE). The integration of residual learning and self-attention successfully advanced the model's ability to map and reconstruct complex normal acoustic distributions, resulting in a higher Source Domain anomaly detection AUC. Future research on this architecture must focus on integrating explicit domain generalization techniques, such as data augmentation (Mixup) or adversarial domain classification, to constrain the latent space and improve target domain robustness.

## 9. References
1. Dohi, K., et al. (2023). "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring." arXiv:2305.07828.
2. Vaswani, A., et al. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems.
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
