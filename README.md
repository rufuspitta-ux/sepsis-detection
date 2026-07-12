# Early Sepsis Detection

An early-warning system that flags rising sepsis risk in ICU patients using
only hourly vital signs — built as a portfolio project to demonstrate an
end-to-end ML workflow: data engineering, model selection, calibration,
explainability, and deployment.

**Live demo:** https://huggingface.co/spaces/Rufuspitta1/early-sepsis-detection

---

## What it does

Upload a CSV of a patient's hourly vitals (HR, O2Sat, Temp, SBP, MAP, DBP,
Resp, Age, Gender, ICULOS) and the app returns:

- A risk trajectory chart showing how predicted sepsis risk changes hour by
  hour
- A SHAP-based explanation of which factors are driving the most recent
  prediction
- An hourly detail table

Two example patients are built into the demo — no file needed to try it:
- **Stable patient** — normal vitals throughout, model correctly stays quiet
- **Deteriorating patient** — gradual physiological decline consistent with
  sepsis onset, model correctly flags rising risk

## Why this exists

Sepsis is time-critical: earlier detection meaningfully improves outcomes,
but early signs are subtle and easy to miss in a busy ICU. This project asks
whether hourly vitals alone — without waiting on lab results — can surface
a useful early warning signal.

## Dataset

[PhysioNet / Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
— 40,336 ICU patients, ~1.55M hourly rows, 7.27% patient-level sepsis
prevalence. Not included in this repo due to size; download instructions are
at the link above.

## Model & approach

- **Model:** XGBoost, trained on 46 engineered features (rolling statistics,
  rate-of-change, missingness indicators, shock index)
- **Performance:** AUROC ~0.78 ± 0.006 (cross-validated), AUPRC ~0.10
- **Alert threshold:** 80% (calibrated from a false-alarm-rate vs. lead-time
  tradeoff — see *Key design decisions* below)

### Key design decisions

**Why XGBoost over a neural network?**
The predictive features here are already hand-engineered tabular values
(rolling means, rate-of-change, shock index) rather than raw sequential
signal. Neural networks earn their advantage by learning representations
from raw, unstructured input themselves — that extraction step was already
done manually here, which is exactly the setting where gradient-boosted
trees match or beat deep learning, with far less data and tuning. XGBoost
also pairs natively with SHAP, giving per-prediction explanations with no
extra machinery — important for a tool meant to support clinical judgment,
not replace it with a black box.

**Why are lab values excluded from the features?**
Labs are ordered *because* a clinician already suspects sepsis — so a model
trained on lab results partly just detects that a human already suspected
something, rather than adding independent predictive value. Including them
would look like strong performance but would actually be data leakage, and
would defeat the point of an *early* warning system, which needs to work
before that clinical suspicion forms. Labs are also missing 90–99% of the
time in this dataset (not drawn hourly), which reinforces the same
decision.

**Why does the app show risk tiers instead of raw probabilities?**
The model uses `scale_pos_weight` to handle the ~7% class imbalance
(without it, the model would learn to just predict "no sepsis" almost
always). That reweighting improves detection but distorts the raw output
numbers — a score of "0.85" is not a trustworthy 85% real-world probability.
Rather than present a falsely precise number, the app shows risk tiers,
which honestly reflects what the model can actually claim.

**Why is the alert threshold set at 80%?**
Chosen from an explicit threshold sweep, not a default or round-number
guess. At 80%, the model achieves roughly a 5.2% false-alarm rate — low
enough to stay clinically usable without causing alert fatigue — while
still providing a median ~49-hour lead time before the same deterioration
would otherwise become obvious.

## Deployment

Deployed as a [Gradio](https://gradio.app) app on Hugging Face Spaces
(ZeroGPU hardware tier — see `STATUS.md` for deployment notes). The model
itself runs entirely on CPU in well under a second; ZeroGPU is used only to
satisfy Spaces' hardware requirements, decoupled from the actual prediction
pipeline.

## Honest limitations

- Trained and validated on ICU data only — not tested on ED, general ward,
  or outpatient vitals
- This is a research / portfolio demonstration, **not a clinical decision
  tool**, and should never be used for actual patient care
- Requires several hours of sequential hourly readings; a single timepoint
  is not enough, since the model relies on rolling trends rather than
  snapshots
- Raw output probabilities are miscalibrated (see design decisions above)
  — the app deliberately shows risk tiers instead

## Project structure

```
early-sepsis-detection/
├── app.py                              # Gradio app (deployed to HF Spaces)
├── baseline_model.pkl                  # Trained XGBoost model + feature columns
├── requirements.txt
├── STATUS.md                           # Deployment log and technical notes
├── data/
│   ├── sample_patient_stable.csv
│   └── sample_patient_deteriorating.csv
└── src/
    ├── 03_feature_engineering.py       # engineer_features() — mirrored in app.py
    ├── 04_model_training.py
    ├── 05_threshold_calibration.py
    └── 06_shap_explainability.py
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Acknowledgments

Built on the PhysioNet/CinC 2019 Challenge dataset. Thanks to the
PhysioNet team and challenge organizers for making this data available for
research and learning.
