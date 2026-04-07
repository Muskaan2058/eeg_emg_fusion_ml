# EEG–EMG Fusion for Upper-Limb Movement Decoding
### Using Deep Learning Models on the NeBULA Dataset
**Muskaan Garg | H00416442 | Heriot-Watt University**
*BSc Honours Computer Science (Artificial Intelligence) — Supervised by Dr. Heba El-Shimy*

---

## Overview
This project investigates multimodal fusion of **electroencephalography (EEG)** and **electromyography (EMG)** signals for upper-limb movement decoding using deep learning. By combining cortical motor intention (EEG) with muscular activation (EMG), the study evaluates whether fusion models outperform single-modality approaches, and explores the temporal relationship between neural and muscular activity.

---

## Research Questions
1. Do deep learning fusion models (EEG+EMG) outperform single-modality models for upper-limb reaching task classification?
2. Which architecture — spatial-temporal (EEGNet/EMGNet) or sequential (CNN-LSTM) — best captures neural and muscular movement patterns?
3. What is the subject-specific temporal delay between EEG motor cortex activity and EMG muscle onset, and does robotic assistance alter this delay?

---

## Dataset — NeBULA
**NeBULA (Neuromechanical Biomarkers for Upper Limb Assessment)**
Garro et al. (2025) — *Scientific Data*, Nature Publishing Group

> Download: https://doi.org/10.6084/m9.figshare.27301629
> Place the downloaded folder at: `./data/nebula/`

| Modality | Device | Channels | Sampling Rate |
|----------|--------|----------|---------------|
| **EEG** | BrainProducts ActiCHamp (128-ch) | 15 motor cortex ROI | 1000 Hz |
| **EMG** | Cometa Waveplus (wireless) | 11 muscles | 1000 Hz |

- **40 healthy subjects** (sub-03 excluded — documented EMG noise)
- **3 standardised reaching tasks** across 3 conditions: unassisted (`free`), low assistance (`low`), high assistance (`high`)
- **~30 trials per subject** per condition (10 per task)
- **BIDS-compliant**, hardware-synchronised, open access (CC BY-NC-ND 4.0)
- **Verified synchronisation:** EEG and EMG event onsets match to within 1ms across all trials

**EMG muscles recorded:**
Biceps, Anterior/Mid/Posterior Deltoid, Triceps, Upper/Mid/Lower Trapezius, Pectoralis, Brachioradialis, Pronator Teres

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Muskaan2058/eeg_emg_fusion_ml
cd eeg_emg_fusion_ml
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Download the dataset
Download NeBULA from: https://doi.org/10.6084/m9.figshare.27301629
Place it so the structure looks like:
```
eeg_emg_fusion_ml/
└── data/
    └── nebula/
        ├── participants.tsv
        ├── sub-01/
        ├── sub-02/
        └── ...
```

### 4. Run feasibility validation
```bash
# Auto-detects NeBULA folder
python feasibility.py

# Or with explicit path
python feasibility.py ./data/nebula

# All subjects, specific condition
python feasibility.py ./data/nebula --all-subjects --condition free
```

---

## Preprocessing Pipeline

### EEG
| Step | Details |
|------|---------|
| Channel selection | 15 motor cortex channels (C3, C4, Cz, FC3, FC4, CP3, CP4 + neighbours) |
| Detrending | Linear detrend |
| Bandpass filter | 1–45 Hz (Butterworth, 4th order) |
| Notch filter | 50 Hz powerline removal |
| Re-referencing | Common Average Reference (CAR) |
| Resampling | 1000 Hz → 200 Hz |
| Normalisation | Z-score per channel |

### EMG
| Step | Details |
|------|---------|
| Bandpass filter | 20–400 Hz |
| Notch filter | 50 Hz |
| Rectification | Full-wave |
| Envelope | RMS with 100ms sliding window |
| Resampling | 1000 Hz → 200 Hz |
| Normalisation | Z-score per channel |

## Modelling Plan

| Modality | Model | Architecture | Role |
|----------|-------|-------------|------|
| EEG only | EEGNet | Spatial-temporal CNN | Baseline EEG classifier |
| EEG only | CNN-LSTM | Hybrid CNN + Recurrent | Sequential EEG classifier |
| EMG only | EMGNet | 1D Temporal CNN | Baseline EMG classifier |
| EMG only | CNN-LSTM | Hybrid CNN + Recurrent | Sequential EMG classifier |
| EEG + EMG | Fusion Model | Feature-level multimodal | Best EEG + EMG embeddings concatenated |

**Classification:** 3-class (Task 1, Task 2, Task 3), unassisted condition, subject-dependent split
**Timing analysis:** EEG ERD onset vs EMG activation onset, cross-correlation, condition comparison

---

## Evaluation Metrics
- Accuracy
- Macro F1-Score
- ROC-AUC
- Confusion Matrix
- Wilcoxon signed-rank test (statistical comparison between models)
- EEG→EMG delay in milliseconds (timing analysis)

---

## Tools & Libraries
- **Language:** Python 3.13
- **Signal processing:** MNE-Python, SciPy, NumPy
- **Deep learning:** TensorFlow / PyTorch
- **Visualisation:** Matplotlib, Seaborn
- **Environment:** Virtual environment / Google Colab GPU

---

## References
Garro, F. et al. (2025). An EEG-EMG dataset from a standardized reaching task for biomarker research in upper limb assessment. *Scientific Data*, 12, 831. https://doi.org/10.1038/s41597-025-05042-4