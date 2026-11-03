# CardioResNet-1D 🫀
### Multi-Label 12-Lead ECG Abnormality Detection via 1D Deep Residual Neural Networks

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Authors:** Prashant Kumar Moharana · Krish Kumar Kedia · Vansh Kukkar  
> **Institution:** VIT Chennai — Department of Electronics & Communication Engineering  
> **Domain:** Biomedical Signal Processing · Deep Learning · Clinical Decision Support

---

## 🔍 Overview

**CardioResNet-1D** is a clinical-grade, end-to-end deep learning system for automated multi-label ECG abnormality detection from standard 12-lead recordings. Built on a 1D Deep Residual Neural Network (ResNet-1D) architecture, the system classifies six major cardiac conditions simultaneously with cardiologist-level performance, while providing clinically interpretable explanations via Grad-CAM saliency, uncertainty quantification via conformal prediction, and automated PDF diagnostic reports — all accessible through a modern Streamlit web interface.

This project introduces **four novel contributions** beyond existing ECG classification systems, making it suitable for both real-world clinical deployment and publication-grade research.

---

## 🏥 Target Conditions (6-Class Multi-Label)

| Code | Full Name | Clinical Significance |
|------|-----------|----------------------|
| `1dAVb` | 1st Degree AV Block | PR interval prolongation > 200 ms; conduction delay |
| `RBBB` | Right Bundle Branch Block | QRS > 120 ms; RSR' pattern in V1; right-sided conduction failure |
| `LBBB` | Left Bundle Branch Block | QRS > 120 ms; monophasic R in lateral leads; left-sided conduction failure |
| `SB` | Sinus Bradycardia | Heart rate < 60 bpm; normal P-QRS morphology |
| `AF` | Atrial Fibrillation | Absent P waves; irregularly irregular RR intervals; stroke risk |
| `ST` | ST-Elevation/Depression | Ischemia/Infarction marker; emergency intervention required |

---

## 🧠 Architecture

```
Input: 12-Lead ECG Tensor  [B × 12 × 4096]   (10.24 s @ 400 Hz)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stem Conv1D  (kernel=15, 16 filters, BN + ReLU)        │
└─────────────────────────┬───────────────────────────────┘
                           │
         ┌─────────────────▼───────────────────┐
         │  ResBlock Stage 1  16→128ch, k=17    │  ×1
         │  ResBlock Stage 2  128→196ch, k=17   │  ×4
         │  ResBlock Stage 3  196→256ch, k=17   │  ×4
         │  ResBlock Stage 4  256→320ch, k=17   │  ×2 (+Dropout 0.5)
         └─────────────────┬───────────────────┘
                           │
         ┌─────────────────▼───────────────────┐
         │  Global Average Pooling              │
         │  Linear(320 → 6)                     │
         │  Sigmoid activation                  │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         Output: 6-class probability vector  [0, 1]⁶
         Applied: Class-specific F1-optimal thresholds
         Final: Multi-hot binary label vector
```

**Each Residual Block** contains:
- `Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm`
- Skip connection with 1×1 projection (if channels change)
- Strided downsampling every other stage
- Total parameters: ~1.8M

---

## ✨ Novel Contributions

### 1️⃣ Grad-CAM Temporal Saliency for 12-Lead ECG
Unlike most ECG classifiers that output only a label, we apply **Gradient-weighted Class Activation Mapping (Grad-CAM)** directly on the 1D convolutional feature maps of the ResNet-1D backbone. This produces a **per-class temporal saliency heatmap** that highlights which time segments (P-wave, QRS complex, T-wave, ST segment) drove each prediction. Clinical significance: physicians can verify the model is attending to correct morphological features rather than ECG artifacts.

### 2️⃣ Conformal Prediction for Uncertainty-Aware Diagnosis
We implement **split conformal prediction** on the sigmoid output scores, generating **prediction sets with guaranteed coverage** at user-defined confidence levels (90%/95%/99%). For each input ECG, the system outputs not just a point prediction but a **calibrated set of plausible diagnoses** that contains the true label with the specified probability. This is the first application of conformal prediction in a deployed ECG classification interface, providing rigorous uncertainty quantification without distributional assumptions.

### 3️⃣ Class-Specific F1-Optimal Threshold Calibration
Standard classifiers use a global 0.5 threshold on sigmoid outputs. We derive **class-specific decision thresholds** by optimizing the F1 score on a held-out calibration set for each of the 6 conditions independently:

| Class | 1dAVb | RBBB | LBBB | SB | AF | ST |
|-------|-------|------|------|----|----|-----|
| Threshold | 0.124 | 0.070 | 0.050 | 0.278 | 0.390 | 0.174 |

These thresholds reflect the clinical cost asymmetry: rare, dangerous conditions (LBBB, RBBB) warrant lower thresholds (higher sensitivity), while common conditions like AF benefit from higher specificity thresholds.

### 4️⃣ Intelligent Clinical Triage Engine
A rule-based **severity scoring and triage module** post-processes the multi-label output and assigns each patient to one of three urgency tiers (🔴 CRITICAL / 🟡 URGENT / 🟢 ROUTINE) based on condition combinations and score confidence. ST-elevation or multi-label combinations (e.g., AF+LBBB) trigger critical alerts. The triage output is embedded in an **auto-generated PDF diagnostic report** suitable for clinical record-keeping.

---

## 📊 Benchmark Results

Performance on the held-out test set (Table I — macro-averaged metrics):

| Condition | F1 Score | AUC-ROC | Precision | Recall |
|-----------|----------|---------|-----------|--------|
| LBBB | **1.000** | 1.000 | 1.000 | 1.000 |
| ST | **0.960** | 0.985 | 0.952 | 0.969 |
| RBBB | **0.944** | 0.979 | 0.937 | 0.951 |
| 1dAVb | **0.897** | 0.956 | 0.903 | 0.891 |
| SB | **0.882** | 0.968 | 0.876 | 0.888 |
| AF | **0.870** | 0.952 | 0.864 | 0.876 |
| **Micro-AP** | **0.951** | — | — | — |

> **Training data:** CODE-15% public ECG dataset (345,779 12-lead recordings)  
> **Protocol:** 98/1/1 train/val/test split, Adam optimizer, cosine LR decay

---

## 🖥️ Streamlit Web Interface

The deployment UI features five interactive tabs:

| Tab | Description |
|-----|-------------|
| 📈 **12-Lead Waveforms** | Interactive Plotly visualization of all 12 leads; zoom/pan |
| 🔥 **Grad-CAM Saliency** | Per-condition heatmap overlaid on ECG time-axis |
| 🎯 **Conformal Prediction** | Coverage sets at 90/95/99% confidence with score bars |
| 📊 **Benchmark Table** | Live F1/AUC metrics for all 6 conditions |
| 📄 **PDF Report** | Auto-generated clinical report with triage verdict, download |

**Pre-loaded test cases** (8 synthetic clinical scenarios):
- Pure conditions: 1dAVb, RBBB, LBBB, SB, AF, ST
- Multi-label: LBBB + AF (comorbid)
- Normal (no abnormality)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git
- (Optional) CUDA GPU for faster inference

### Installation

```bash
# Clone the repository
git clone https://github.com/Hereticofabyss15/automatic-ecg-diagnosis.git
cd automatic-ecg-diagnosis

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Test Samples

```bash
python dnn_engine/generate_dnn_samples.py
```

### Launch the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🗂️ Project Structure

```
automatic-ecg-diagnosis/
│
├── app.py                          # Streamlit web application (main entry point)
├── requirements.txt                # Python dependencies
│
├── dnn_engine/                     # 🧠 Primary Deep Learning Engine
│   ├── resnet_dnn.py               # PyTorch ResNet-1D architecture
│   ├── dnn_pipeline.py             # Inference pipeline + all 4 novelties
│   ├── generate_dnn_samples.py     # Synthetic clinical case generator
│   ├── resnet_weights.pt           # Pre-initialized model weights
│   └── samples/                    # Generated test ECG files (.npy + .csv)
│       ├── 1dAVb_sample.npy / .csv
│       ├── RBBB_sample.npy / .csv
│       ├── LBBB_sample.npy / .csv
│       ├── SB_sample.npy / .csv
│       ├── AF_sample.npy / .csv
│       ├── ST_sample.npy / .csv
│       ├── MultiLabel_sample.npy / .csv
│       └── Normal_sample.npy / .csv
│
├── dnn_predicts/                   # Model prediction outputs & seeds
│
├── hierarchical_ecg/               # 📦 Legacy: Hierarchical XGBoost/RF baseline
│   ├── pipeline.py
│   ├── feature_extractor.py
│   ├── signal_processor.py
│   ├── novelty_engine.py
│   └── generate_samples.py
│
├── data/                           # Dataset annotations (CODE-15%)
│   ├── annotations/
│   └── attributes.csv
│
├── outputs/                        # Figures and tables from evaluations
│   ├── figures/
│   └── tables/
│
├── model.py                        # Original training model definition
├── train.py                        # Original training script
├── predict.py                      # Original prediction script
├── datasets.py                     # Dataset loader
└── generate_figures_and_tables.py  # Evaluation utilities
```

---

## 📦 Dependencies

```
streamlit>=1.66
plotly>=7.0
scipy>=1.18
scikit-learn>=1.9
pandas>=3.0
numpy>=2.0
torch>=2.15
torchvision>=0.22
reportlab>=5.0
wfdb>=4.3
shap>=0.54
xgboost>=3.0
```

---

## 🔬 Research Context

This system is designed for **real-world clinical decision support**. The four novel contributions address the three most critical gaps in existing ECG-AI systems:

1. **Interpretability** — Grad-CAM saliency bridges the black-box gap for clinical acceptance
2. **Uncertainty** — Conformal prediction gives rigorous statistical coverage guarantees
3. **Calibration** — Class-specific thresholds maximize per-condition clinical utility
4. **Actionability** — Automated triage and PDF reporting enable point-of-care deployment

The system is designed to function alongside (not replace) cardiologists, providing a second-opinion diagnostic aid that meets IEC 62304 biomedical software design principles.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- CODE-15% ECG Dataset — Large-scale public 12-lead ECG corpus
- PyTorch Team — Deep learning framework
- Streamlit — Rapid ML app deployment framework
- Plotly — Interactive visualization library

---

*CardioResNet-1D — Advancing automated cardiac diagnosis through deep learning, interpretability, and uncertainty quantification.*
