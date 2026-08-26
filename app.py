"""
Sepsis Early Warning - Gradio app for Hugging Face Spaces.

Takes a CSV of one patient's hourly vitals (HR, O2Sat, Temp, SBP, MAP, DBP,
Resp, Age, Gender, ICULOS columns - matching PhysioNet CinC2019 format) and
returns a risk trajectory chart plus a plain-language explanation of the
most recent prediction.

IMPORTANT / HONEST LIMITATIONS (shown in the app itself, not hidden):
- Trained on ICU data only; not validated on ED, ward, or outpatient vitals.
- Research/portfolio demo, not a clinical decision tool.
- Requires several hours of sequential readings -- a single timepoint isn't
  enough, because key features are rolling trends, not snapshots.
"""
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=4)
    def _touch_gpu():
        """Keep ZeroGPU-compatible deployments healthy; prediction itself is CPU-only."""
        return None
else:
    def _touch_gpu():
        return None

BASE_DIR = Path(__file__).resolve().parent
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

with (BASE_DIR / "baseline_model.pkl").open("rb") as f:

    saved = pickle.load(f)
MODEL, FEATURE_COLS = saved["model"], saved["feature_cols"]
EXPLAINER = shap.TreeExplainer(MODEL)
THRESHOLD = 0.8  # chosen from the threshold sweep: ~5% false-alarm rate, 49h median lead time


def engineer_features(df):
    """Mirrors src/03_feature_engineering.py exactly, for a single patient."""
    df = df.sort_values("ICULOS").reset_index(drop=True)
    for col in VITALS:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df[VITALS] = df[VITALS].ffill()
    for col in VITALS:
        df[col] = df[col].fillna(df[col].median())

    n = len(df)
    row_idx = np.arange(n)
    for col in VITALS:
        present = (df[f"{col}_missing"].values == 0)
        idx_if_present = np.where(present, row_idx, -1)
        last_measured_idx = np.maximum.accumulate(idx_if_present)
        last_measured_idx = np.where(last_measured_idx < 0, 0, last_measured_idx)
        df[f"{col}_hrs_since_measured"] = row_idx - last_measured_idx

    for col in VITALS:
        df[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll6_std"] = df[col].rolling(6, min_periods=1).std().fillna(0)
        df[f"{col}_roc3"] = df[col].diff(3).fillna(0)

    df["shock_index"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["shock_index"] = df["shock_index"].fillna(df["shock_index"].median())
    return df


def predict(csv_file):
    if csv_file is None:
        return None, "Upload a CSV first.", None

    path = Path(str(csv_file))
    if not path.exists():
        return None, "The uploaded file could not be found.", None
    if path.stat().st_size > 5 * 1024 * 1024:
        return None, "CSV files must be smaller than 5 MB.", None

    try:
        raw = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        return None, f"Could not read the CSV: {exc}", None

    required = VITALS + ["Age", "Gender", "ICULOS"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        return None, f"CSV is missing required columns: {missing_cols}", None
    if len(raw) < 6:
        return None, "Need at least 6 hourly rows for rolling features to be meaningful.", None

    raw = raw[required].copy()
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    non_vital = ["Age", "Gender", "ICULOS"]
    if raw[non_vital].isna().any().any():
        return None, "Age, Gender, and ICULOS must contain numeric values.", None
    if not np.isfinite(raw[non_vital].to_numpy()).all():
        return None, "Age, Gender, and ICULOS must be finite.", None
    if raw["ICULOS"].duplicated().any() or (raw["ICULOS"] < 0).any():
        return None, "ICULOS must contain unique non-negative hour values.", None
    if ((raw["Age"] < 0) | (raw["Age"] > 120)).any():
        return None, "Age must be between 0 and 120.", None
    if (~raw["Gender"].isin([0, 1])).any():
        return None, "Gender must use the dataset encoding 0 or 1.", None

    feats = engineer_features(raw)

    probs = MODEL.predict_proba(feats[FEATURE_COLS])[:, 1]
    feats["risk"] = probs

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(feats["ICULOS"], feats["risk"] * 100, marker="o", color="#c0392b")
    ax.axhline(THRESHOLD * 100, color="black", linestyle="--", alpha=0.6, label=f"Alert threshold ({int(THRESHOLD*100)}%)")
    ax.set_xlabel("Hours since ICU admission")
    ax.set_ylabel("Predicted sepsis risk (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Predicted sepsis risk over time")
    fig.tight_layout()

    latest = feats.iloc[-1]
    latest_risk = latest["risk"] * 100
    shap_vals = EXPLAINER.shap_values(feats[FEATURE_COLS].iloc[[-1]])[0]
    contrib = pd.Series(shap_vals, index=FEATURE_COLS).sort_values(key=abs, ascending=False)

    lines = [f"**Latest hour risk: {latest_risk:.1f}%** "
             f"({'ABOVE' if latest_risk/100 >= THRESHOLD else 'below'} the {int(THRESHOLD*100)}% alert threshold)\n",
             "Top factors driving this prediction:"]
    for feat, val in contrib.head(5).items():
        direction = "↑ increasing" if val > 0 else "↓ decreasing"
        lines.append(f"- `{feat}` {direction} risk (value: {latest[feat]:.2f})")

    summary = "\n".join(lines)
    table = feats[["ICULOS", "risk"] + VITALS].round(2)
    plt.close(fig)
    return fig, summary, table


DISCLAIMER = """
### ⚠️ Research demo -- not a clinical tool
Trained on the PhysioNet/CinC 2019 Challenge ICU dataset only. Not validated on other care settings,
populations, or monitoring equipment. Operating threshold (80%) was chosen from a single train/test
split without cross-validation. **Do not use for actual patient care.**
"""

with gr.Blocks(title="Early Sepsis Risk Demo") as app:
    gr.Markdown("# Early Sepsis Detection -- Vitals-Based Risk Trajectory")
    gr.Markdown(DISCLAIMER)
    gr.Markdown("Upload a CSV with hourly rows and columns: `HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender, ICULOS`")

    with gr.Row():
        file_input = gr.File(label="Patient vitals CSV", file_types=[".csv"])
    gr.Examples(
        examples=[
            [str(BASE_DIR / "sample_patient_stable.csv")],
            [str(BASE_DIR / "sample_patient_deteriorating.csv")],
        ],
        inputs=file_input,
        label="Try an example patient",
    )
    run_btn = gr.Button("Run risk analysis", variant="primary")

    plot_output = gr.Plot(label="Risk trajectory")
    text_output = gr.Markdown()
    table_output = gr.Dataframe(label="Hourly detail")

    run_btn.click(predict, inputs=file_input, outputs=[plot_output, text_output, table_output])

if __name__ == "__main__":
    app.launch()