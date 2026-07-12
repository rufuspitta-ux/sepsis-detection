## Deployment — COMPLETE

**Status: Live and verified on Hugging Face Spaces.**
Space: https://huggingface.co/spaces/Rufuspitta1/early-sepsis-detection

### What was fixed
- **Root cause of instability:** the entire prediction pipeline (feature
  engineering, XGBoost inference, SHAP explanation, matplotlib rendering)
  was wrapped inside `@spaces.GPU(duration=10)`. None of this needs a GPU
  — XGBoost + SHAP on tabular data runs in <0.4s on CPU — but wrapping it
  routed every click through HF's ZeroGPU worker queue, causing repeated
  `GPU task aborted` errors and quota exhaustion.
- **Fix:** decoupled the GPU requirement from the real pipeline.
  `predict()` is now a plain CPU function called directly by the Gradio
  button. A separate no-op `_touch_gpu()` function, decorated with
  `@spaces.GPU(duration=4)`, exists only to satisfy ZeroGPU's startup
  requirement (a Space must have ≥1 `@spaces.GPU` function) and is never
  called during actual use.
- Also hit and resolved along the way: `requirements.txt` renamed by HF's
  upload system, numpy/numba version conflict, and CPU-Basic hardware
  being gated behind HF PRO for new free accounts.

### What was added
- Two example patient CSVs wired into the app via `gr.Examples`, so
  visitors can try the demo with one click instead of needing their own
  ICU vitals file:
  - `sample_patient_stable.csv` — 72h, normal vitals throughout, no
    deterioration trend.
  - `sample_patient_deteriorating.csv` — 90h, stable through hour ~60,
    then gradual rise in HR/temp/resp with falling BP/O2sat, consistent
    with sepsis onset.

### Verified end-to-end
- Both example patients tested on the live Space: stable patient stays
  below threshold across all 72 hours; deteriorating patient's risk
  trajectory rises and crosses the 80% alert threshold, matching
  expected onset timing.
