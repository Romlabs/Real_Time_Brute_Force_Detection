# 🛡️ Real-Time Brute-Force Detection (Streamlit)

Interactive dashboard for detecting brute-force attacks with Random Forest.

## Features
- 📥 Kaggle auto-download or CSV upload
- 🧹 Smart cleaning (NaN handling + label encoding)
- ⚖️ SMOTE visualization
- 🌲 Baseline vs. balanced model comparison
- 🔧 Grid Search + 🎲 Randomized Search (Recall & PR-AUC)
- 📉 Interactive Precision-Recall threshold tuner
- 📊 Feature importance
- 🔮 Prediction on new data

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
