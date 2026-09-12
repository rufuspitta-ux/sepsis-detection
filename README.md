# Early Sepsis Detection

An end-to-end healthcare ML project that uses **hourly ICU vital signs** to flag rising sepsis risk. The project focuses on the complete workflow: temporal feature engineering, model training, threshold calibration, explainability, and deployment.

> **Research / portfolio prototype:** This project is not a clinical decision tool and must not be used for patient care. Clinical deployment would require independent validation, prospective evaluation, safety engineering, regulatory review, and qualified clinical oversight.

**Live demo:** https://huggingface.co/spaces/Rufuspitta1/early-sepsis-detection

## What it does

Upload a CSV containing hourly patient observations:

`HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender, ICULOS`

The application produces:

- A risk trajectory showing how the model score changes over time
- SHAP-based feature explanations for the latest prediction
- An hourly detail table
- Built-in stable and deteriorating example patients for demonstration

## Dataset

The project uses the **PhysioNet / Computing in Cardiology Challenge 2019** dataset.

- 40,336 ICU patients
- Approximately 1.55 million hourly rows
- Approximately 7.27% patient-level sepsis prevalence
- Dataset is not redistributed in this repository; obtain it from PhysioNet according to its terms

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
Threshold calibration
       ↓
SHAP explanation
       ↓
Risk tier + trajectory
       ↓
Gradio application
```

### Engineered features

The model uses **46 features**, including:

- Rolling statistics
- Rate-of-change features
- Missingness indicators
- Shock index
- Patient/context variables

The model is designed around trends rather than a single vital-sign snapshot.

### Model

**XGBoost** was selected because the input is structured tabular data after feature engineering. It provides strong performance on this type of data while remaining relatively lightweight and works well with SHAP for local explanations.

Reported cross-validation performance:

- **AUROC:** ~0.78 ± 0.006
- **AUPRC:** ~0.10

These numbers should be interpreted as project evaluation results, not evidence of clinical effectiveness.

## Threshold calibration

Rather than treating the classifier's raw output as a clinical probability, the project evaluates thresholds using a false-alarm/lead-time tradeoff.

The current demonstration uses an **80% model-score threshold**, associated in the project's evaluation with approximately:

- **5.2% false-alarm rate**
- **~49-hour median lead time**

These figures require independent validation before they could support any clinical claim.

## Why labs are excluded

The project deliberately focuses on routinely available hourly vital signs. Laboratory measurements can be irregular and may also reflect the fact that clinicians have already become concerned about a patient. Restricting the prototype to vital signs keeps the research question focused on earlier physiological trends rather than simply reproducing an existing clinical suspicion.

## Why risk tiers instead of raw probabilities?

The training setup uses class weighting to address the strong class imbalance. Because this affects the interpretation of the raw classifier score, the application presents **risk tiers** rather than implying that a score such as `0.85` means an actual 85% probability of sepsis.

## Deployment

The application is deployed with **Gradio on Hugging Face Spaces**. The prediction pipeline runs on CPU; the deployment configuration is documented in `STATUS.md`.

## Honest limitations

- ICU data only; external populations are not validated
- Requires sequential hourly observations
- No prospective clinical validation
- Not intended for diagnosis or treatment decisions
- Raw classifier scores should not be interpreted as calibrated clinical probabilities
- Dataset shift and missing-data patterns may affect performance
- Threshold/lead-time results require independent reproduction

## Repository structure

```text
early-sepsis-detection/
├── app.py
├── baseline_model.pkl
├── requirements.txt
├── STATUS.md
├── data/
│   ├── sample_patient_stable.csv
│   └── sample_patient_deteriorating.csv
└── src/
    ├── 03_feature_engineering.py
    ├── 04_model_training.py
    ├── 05_threshold_calibration.py
    └── 06_shap_explainability.py
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Reproducibility

For a stronger independent reproduction, use the same dataset version and document the exact patient-level split, preprocessing steps, random seeds, feature columns, XGBoost configuration, and threshold-selection procedure. Evaluation should include patient-disjoint validation and clinically meaningful metrics such as sensitivity, specificity, AUROC, AUPRC, false-alarm rate, and lead time.

## Acknowledgments

Built using the PhysioNet/CinC 2019 Challenge dataset for research and learning.

## License

See the repository license file for project licensing information.
