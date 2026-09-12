# Early Sepsis Detection

A healthcare machine-learning prototype that uses **hourly ICU vital signs** to model changing sepsis-related signal patterns over time. The project focuses on temporal feature engineering, XGBoost inference, threshold analysis, SHAP explanations, and a Gradio demonstration interface.

> **Research / portfolio prototype:** This project is not a clinical decision tool and must not be used for patient care. Clinical deployment would require independent validation, prospective evaluation, safety engineering, regulatory review, and qualified clinical oversight.

**Live demo:** https://huggingface.co/spaces/Rufuspitta1/early-sepsis-detection

## What it does

Upload a CSV containing hourly patient observations:

`HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender, ICULOS`

The application produces:

- A model-score trajectory showing how the classifier output changes over time
- SHAP-based feature explanations for the latest prediction
- An hourly detail table
- Built-in stable and deteriorating example patients for demonstration

The displayed score is a **model output**, not a calibrated clinical probability.

## Dataset

The project uses the **PhysioNet / Computing in Cardiology Challenge 2019** dataset.

- 40,336 ICU patients
- Approximately 1.55 million hourly rows
- Approximately 7.27% patient-level sepsis prevalence
- The original dataset is not redistributed in this repository; obtain it from PhysioNet according to its terms

## Machine-learning pipeline

```text
Hourly ICU vitals
       ↓
Data cleaning
       ↓
Temporal feature engineering
       ↓
XGBoost classifier
       ↓
Project threshold analysis
       ↓
SHAP explanation
       ↓
Model-score tier + trajectory
       ↓
Gradio application
```

### Engineered features

The deployed inference pipeline in `app.py` builds temporal features including:

- Rolling mean and standard deviation over recent observations
- Rate-of-change features
- Missingness indicators
- Hours-since-measurement features
- Shock index
- Patient/context variables

The prototype is designed around **trends rather than a single vital-sign snapshot**.

> **Repository scope note:** the current public repository contains the deployment/inference pipeline and a serialized baseline model. The original model-training scripts referenced in earlier documentation are not currently present in the repository, so the exact training split and training configuration cannot be independently reproduced from this repository alone.

## Model

**XGBoost** is used for structured tabular data after temporal feature engineering. SHAP is used to provide local feature-attribution views for the latest model output.

Any performance figures from prior experiments should be treated as **project evaluation results only** until the full training/evaluation pipeline is published with patient-level splits, seeds, feature definitions, and reproducible commands.

## Threshold analysis

The demonstration uses an **80% model-score operating threshold**. This is a project setting for the deployed prototype, not a clinical alert rule.

Earlier project evaluation associated this threshold with an approximate false-alarm/lead-time tradeoff. Those figures are **not independently reproducible from the current public repository** because the original threshold-sweep artifacts and training pipeline are not included.

## Why labs are excluded

The prototype focuses on routinely available hourly vital signs. This keeps the research question centered on physiological trends rather than adding irregular laboratory measurements that may be influenced by an existing clinical concern.

## Why model-score tiers instead of raw probabilities?

The training setup uses class weighting to address class imbalance. The application therefore avoids presenting a raw classifier output such as `0.85` as an actual 85% probability of sepsis.

## Deployment

The application is deployed with **Gradio on Hugging Face Spaces**. The prediction path is designed to run on CPU; deployment notes and known fixes are documented in `STATUS.md`.

## Example data

Two small synthetic demonstration files are included:

- `sample_patient_stable.csv`
- `sample_patient_deteriorating.csv`

They are intended only to demonstrate the interface and trajectory visualization. They are **not clinical patient records** and must not be interpreted as validated clinical examples.

## Honest limitations

- ICU data only; external populations are not validated
- Requires sequential hourly observations
- No prospective clinical validation
- Not intended for diagnosis or treatment decisions
- Raw classifier scores are not calibrated clinical probabilities
- Dataset shift and missing-data patterns may affect performance
- Threshold and lead-time results require independent reproduction
- The public repository does not currently contain the full training/evaluation pipeline

## Repository structure

```text
sepsis-detection/
├── app.py
├── baseline_model.pkl
├── requirements.txt
├── STATUS.md
├── sample_patient_stable.csv
└── sample_patient_deteriorating.csv
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The local app expects `baseline_model.pkl` in the repository root.

## Reproducibility roadmap

For an independent reproduction, the next release should publish:

1. The exact patient-level train/validation/test split
2. Random seeds and XGBoost hyperparameters
3. The complete feature-definition and preprocessing pipeline
4. Training and threshold-selection commands
5. Versioned evaluation artifacts
6. Patient-disjoint metrics including sensitivity, specificity, AUROC, AUPRC, false-alarm rate, and lead time
7. Calibration analysis if probabilities are ever presented

## Acknowledgments

Built using the PhysioNet/CinC 2019 Challenge dataset for research and learning.

## License

See the repository license file for project licensing information.
