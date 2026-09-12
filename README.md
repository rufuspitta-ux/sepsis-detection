# Early Sepsis Detection — Temporal ICU ML Prototype

A healthcare machine-learning prototype that uses **hourly ICU vital signs** to model changing physiological patterns over time. The project focuses on temporal feature engineering, XGBoost inference, threshold analysis, SHAP explanations, and a Gradio demonstration interface.

> **Research / portfolio prototype:** this project is not a clinical decision tool and must not be used for patient care. Clinical deployment would require independent validation, prospective evaluation, safety engineering, regulatory review, and qualified clinical oversight.

## Live demo

**[Open the Hugging Face demo](https://huggingface.co/spaces/Rufuspitta1/early-sepsis-detection)**

The public demo is intended to show the inference workflow and interface, not to provide medical advice.

## Why this project matters

Unlike a single-value classifier, this prototype treats ICU measurements as a **time series**. It asks whether changing trends and missingness patterns provide useful signals for an ML model.

```text
Hourly ICU vitals
       ↓
Data cleaning + missingness handling
       ↓
Temporal feature engineering
       ↓
XGBoost classifier
       ↓
Operating-threshold analysis
       ↓
SHAP explanation
       ↓
Model-score trajectory
       ↓
Gradio interface
```

## Input

Upload a CSV containing hourly patient observations:

```text
HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender, ICULOS
```

The application produces:

- A model-score trajectory across ICU hours
- SHAP-based feature explanations for the latest model output
- An hourly feature/detail table
- Stable and deteriorating synthetic examples for interface demonstration

The displayed value is a **model score**, not a calibrated clinical probability.

## Dataset

The project uses the **PhysioNet / Computing in Cardiology Challenge 2019** dataset.

The original dataset is not redistributed in this repository; obtain it directly from PhysioNet according to its terms.

## Feature engineering

The deployed inference pipeline in `app.py` constructs temporal features including:

- Rolling means and standard deviations
- Rate-of-change features
- Missingness indicators
- Hours-since-measurement features
- Shock index
- Patient/context variables

This makes the project primarily about **temporal physiological pattern modeling**, rather than classifying one isolated vital-sign snapshot.

## Model and interpretability

**XGBoost** is used for structured tabular inference after temporal feature engineering.

**SHAP** is used to provide local feature-attribution information for the latest model output.

The current repository contains the deployment/inference pipeline and serialized baseline model. The original training scripts referenced in earlier project documentation are not currently present, so a clean clone cannot independently reproduce the exact historical training configuration.

## Threshold analysis

The demonstration uses an **80% model-score operating threshold**. This is an application setting for the prototype, not a clinical alert rule.

Historical project evaluation explored the trade-off between false alarms and lead time, but those figures are not independently reproducible from the current public repository because the original threshold-sweep artifacts and training pipeline are not included.

## Example data

Two synthetic demonstration files are included:

- `sample_patient_stable.csv`
- `sample_patient_deteriorating.csv`

They exist only to demonstrate the UI and trajectory behavior. They are **not clinical patient records**.

## Deployment

The application uses **Gradio on Hugging Face Spaces**. The inference path is designed for CPU execution.

For deployment notes and known issues, see `STATUS.md`.

## Repository structure

```text
sepsis-detection/
├── app.py                         # Gradio inference application
├── baseline_model.pkl             # Serialized baseline model
├── requirements.txt               # Runtime dependencies
├── STATUS.md                      # Deployment/status notes
├── sample_patient_stable.csv      # Synthetic demo input
└── sample_patient_deteriorating.csv
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The local application expects `baseline_model.pkl` in the repository root.

## Honest limitations

- ICU data only; external populations are not validated
- Requires sequential hourly observations
- No prospective clinical validation
- Not intended for diagnosis or treatment decisions
- Model scores are not calibrated clinical probabilities
- Dataset shift and missing-data behavior may affect performance
- Threshold and lead-time findings require independent reproduction
- The complete model-training/evaluation pipeline is not currently published

## Reproducibility roadmap

The next research-grade release should publish:

1. Exact patient-level train/validation/test splits
2. Random seeds and XGBoost hyperparameters
3. Complete preprocessing and feature definitions
4. Training and threshold-selection commands
5. Versioned evaluation artifacts
6. Patient-disjoint metrics including sensitivity, specificity, AUROC, AUPRC, false-alarm rate, and lead time
7. Calibration analysis before presenting any probability-like output

## Acknowledgments

Built using the PhysioNet/CinC 2019 Challenge dataset for research and learning.

## License

See the repository license file for project licensing information.
