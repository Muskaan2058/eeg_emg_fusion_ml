# EEG–EMG Fusion for Upper-Limb Gesture Classification  
### Using Machine Learning and Deep Learning Models  

---

## Overview  
This project investigates the fusion of **electroencephalography (EEG)** and **electromyography (EMG)** signals for upper-limb gesture recognition using machine learning and deep learning techniques.  
By combining cortical and muscular activity, the study aims to improve gesture classification accuracy compared to single-modality systems.  

---

## Research Aim & Objectives  
**Aim:**  
Develop and evaluate a multimodal framework that integrates EEG and EMG signals for upper-limb gesture classification.  

**Objectives:**  
1. Implement preprocessing pipelines for EEG and EMG signals.  
2. Train and compare spatial–temporal CNNs (EEGNet) and sequence models (CNN–LSTM).  
3. Evaluate fusion versus single-modality performance using standard metrics (Accuracy, F1-Score, ROC-AUC).  
4. Assess model interpretability and generalisation across subjects.  

---

## Dataset  
**Source:**  
[EEG–EMG Dataset for Upper Limb Gesture Classification (Mendeley)](https://data.mendeley.com/datasets/m6t78vngbt/1)  

**Specifications:**  
| Modality | Device | Channels | Sampling Rate | Description |
|-----------|---------|-----------|----------------|--------------|
| **EEG** | OpenBCI Ultracortex IV | 8 | 250 Hz | Cortical neural activity |
| **EMG** | Myo Armband | 8 | 200 Hz | Muscular activation from upper limb |

Each of the 11 subjects performed **7 gestures** with **6 trials** per gesture, yielding synchronized EEG and EMG recordings for multimodal analysis.  

---

## Preprocessing  
### EEG  
- Detrending  
- Band-pass filter (0.5–45 Hz)  
- 50 Hz notch filter  
- Normalization  

### EMG  
- Rectification  
- Band-pass filter (20–90 Hz)  
- 100 ms moving-average envelope  

---

## Feasibility & Preliminary Analysis  

### **4.1.1 EEG Channel Correlation Heatmap**  
- Confirms spatial coherence across 8 EEG channels.  
- Validates signal quality and supports CNN-based spatial feature learning.  

### **4.1.2 EEG Power Spectral Density (PSD)**  
- Mean ± SE PSD across gestures (0–50 Hz).  
- EEG energy concentrated in α (8–12 Hz) and β (13–30 Hz) bands.  

### **4.1.3 EEG vs EMG Frequency Distribution Comparison**  
- EEG < 40 Hz: neural intention signals.  
- EMG 20–90 Hz: muscular execution signals.  
- Confirms complementary frequency ranges for multimodal fusion.  

### **4.1.4 Raw EMG + Envelope Visualization**  
- Demonstrates clear muscle activation bursts and low baseline noise.  
- Indicates high-quality, physiologically valid EMG signals.  

---

## Modelling Plan  
| Modality | Model | Type | Notes |
|-----------|--------|------|-------|
| EEG-only | EEGNet | CNN | Spatial–temporal patterns |
| EMG-only | EMGNet / CNN-LSTM | CNN / Hybrid | Temporal muscle activation |
| Fusion | EEGNet + CNN-LSTM | Multimodal | Early- or mid-level feature fusion |

---

## Key Findings from Feasibility  
- EEG and EMG signals exhibit distinct, non-overlapping frequency domains.  
- Correlation and PSD plots confirm signal integrity and repeatability.  
- EMG envelopes show consistent contraction timing across subjects.  
- Dataset is clean, synchronized, and feasible for deep-learning workflows.  

---

## Tools & Libraries  
- **Languages:** Python 3.10  
- **Core Libraries:** NumPy, Pandas, Matplotlib, SciPy, Seaborn, MNE-Python, PyTorch / TensorFlow  
- **Environment:** Google Colab GPU / Jupyter Notebook  

---

## Evaluation Metrics  
- Accuracy  
- F1-Score  
- ROC–AUC  
- Confusion Matrix  
- Latency / Inference Time  

---
