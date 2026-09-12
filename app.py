"""
Sepsis Early Warning - Gradio app for Hugging Face Spaces.

Takes a CSV of one patient's hourly vitals (HR, O2Sat, Temp, SBP, MAP, DBP,
Resp, Age, Gender, ICULOS columns - matching PhysioNet CinC2019 format) and
returns a model-score trajectory plus an explanation of the latest prediction.

IMPORTANT / HONEST LIMITATIONS:
- Trained on ICU data only; not validated on ED, ward, or outpatient vitals.
- Research/portfolio demo, not a clinical decision tool.
- Requires several hours of sequential readings -- a single timepoint isn't
  enough, because key features are rolling trends, not snapshots.
- The displayed model score is not a calibrated clinical probability.
"""

from pathlib import Path
import pickle

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
THRESHOLD = 0.8  # project operating threshold selected during threshold analysis


def engineer_features(df):
    """Mirror the project's feature engineering for a single patient."""
    df = df.sort_values("ICULOS").reset_index(drop=True)

    for col in VITALS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df[VITALS] = df[VITALS].ffill()
    for col in VITALS:
        df[col] = df[col].fillna(df[col].median())

    n = len(df)
    row_idx = np.arange(n)
    for col in VITALS:
        present = df[f"{col}_missing"].values == 0
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


def _score_tier(score: float) -> str:
    """Convert the model score to a neutral demonstration tier."""
    if score >= THRESHOLD:
        return "ABOVE OPERATING THRESHOLD"
    if score >= 0.5:
        return "INTERMEDIATE MODEL SCORE"
    return "BELOW OPERATING THRESHOLD"


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
    model_scores = MODEL.predict_proba(feats[FEATURE_COLS])[:, 1]
    feats["model_score"] = model_scores

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        feats["ICULOS"],
        feats["model_score"],
        marker="o",
        color="#c0392b",
        label="Model score",
    )
    ax.axhline(
        THRESHOLD,
        color="black",
        linestyle="--",
        alpha=0.6,
        label=f"Operating threshold ({THRESHOLD:.0%})",
    )
    ax.set_xlabel("Hours since ICU admission")
    ax.set_ylabel("Model score (not calibrated probability)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Sepsis model-score trajectory")
    fig.tight_layout()

    latest = feats.iloc[-1]
    latest_score = float(latest["model_score"])
    latest_tier = _score_tier(latest_score)
    shap_values = EXPLAINER.shap_values(feats[FEATURE_COLS].iloc[[-1]])[0]
    contributions = pd.Series(shap_values, index=FEATURE_COLS).sort_values(
        key=abs, ascending=False
    )

    lines = [
        f"**Latest model score: {latest_score:.3f}**",
        f"**Demonstration tier: {latest_tier}**",
        "",
        "This score is an output of the trained classifier, not a calibrated clinical probability.",
        "",
        "Top model features for this prediction:",
    ]
    for feat, value in contributions.head(5).items():
        direction = "↑ model score" if value > 0 else "↓ model score"
        lines.append(f"- `{feat}` {direction} (feature value: {latest[feat]:.2f})")

    summary = "\n".join(lines)
    table = feats[["ICULOS", "model_score"] + VITALS].round(3)
    plt.close(fig)
    return fig, summary, table


DISCLAIMER = """
### ⚠️ Research demo — not a clinical tool
Trained on the PhysioNet/CinC 2019 Challenge ICU dataset only. Not validated on other care settings,
populations, or monitoring equipment. The operating threshold is a project evaluation setting,
not a clinical alert rule. **Do not use for actual patient care.**
"""


with gr.Blocks(title="Early Sepsis Model Demo") as app:
    gr.Markdown("# Early Sepsis Detection — Vitals-Based Model Score")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(
        "Upload a CSV with hourly rows and columns: "
        "`HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender, ICULOS`"
    )

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

    run_btn = gr.Button("Run model analysis", variant="primary")
    plot_output = gr.Plot(label="Model-score trajectory")
    text_output = gr.Markdown()
    table_output = gr.Dataframe(label="Hourly detail")

    run_btn.click(
        predict,
        inputs=file_input,
        outputs=[plot_output, text_output, table_output],
    )


if __name__ == "__main__":
    app.launch()
