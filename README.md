# Neuro Vision – AI-Powered Alzheimer’s Progression Predictor

Neuro Vision predicts Alzheimer’s disease progression using multimodal data: MRI/PET imaging, cognitive scores, and (optionally) genetic markers. It combines **CNNs** for imaging with **Transformer-based** sequence models for tabular/textual features.

## Features
- **Data Pipeline:** loaders for MRI/PET scans and clinical/cognitive datasets
- **Modeling:** CNN for imaging; Transformers/attention for non-imaging features
- **Training:** stratified splits, early stopping, and hyperparameter tuning
- **Evaluation:** accuracy, AUC, sensitivity/specificity for early-stage detection
- **Reporting:** confusion matrix and ROC plots

## Tech Stack
- **Language:** Python 3.9+
- **Frameworks:** PyTorch/TensorFlow
- **Libraries:** numpy, pandas, scikit-learn, nibabel, matplotlib

## Setup
```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
python train.py  # or the main training script
```
Artifacts (models/plots) are saved under `outputs/` (configure in the script).

## Project Structure
```
data/                # (placeholder) imaging + tabular data
src/ or scripts/     # training, evaluation, data loaders
models/              # model definitions
outputs/             # saved models, metrics, plots
```

## Results (example)
- Accuracy: **85%**
- Early detection sensitivity/specificity: **88% / 86%**

