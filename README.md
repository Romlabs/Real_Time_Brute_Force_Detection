# 🛡️ Real-Time Brute-Force Detection — Inference Dashboard

A Streamlit app that serves a **pre-trained** Random Forest classifier
for detecting brute-force attacks on network sessions.

## 🔗 Live Demo
https://realtime-brute-force-detection-u8xtgroivwa3o7jsnhujgz.streamlit.app

## 🧠 How It Works

1. **Training** is done once offline via `train_and_save.py`
2. Artifacts (`model.pkl`, `scaler.pkl`, `feature_names.json`, `encoders.json`)
   are committed to this repo
3. The Streamlit app **loads the artifacts** and offers:
   - **🎛️ Single Session Test** — sliders for each feature → live prediction
   - **📂 Batch Prediction** — upload a CSV → score many sessions at once
   - **📊 Model Information** — inspect hyperparameters and feature importances

## 🚀 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py