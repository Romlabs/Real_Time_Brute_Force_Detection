# -*- coding: utf-8 -*-
"""
Real-Time Brute-Force Detection via Predictive Machine Learning
Streamlit deployment — includes PR-curve, threshold tuning, and model comparison.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from scipy.stats import randint

warnings.filterwarnings("ignore")

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Real-Time Brute-Force Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CUSTOM CSS
# ==================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 18px; padding: 2.2rem 2.5rem; margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.15);
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ""; position: absolute; top: -50%; right: -10%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 { color: #f8fafc; font-size: 2.4rem; font-weight: 800;
               margin: 0 0 0.5rem 0; letter-spacing: -0.02em;
               position: relative; z-index: 1; }
    .hero p  { color: #94a3b8; font-size: 1.05rem; margin: 0;
               position: relative; z-index: 1; font-weight: 400; }
    .hero .badge {
        display: inline-block; background: rgba(56, 189, 248, 0.15);
        color: #38bdf8; padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 1rem;
        letter-spacing: 0.05em; text-transform: uppercase;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .kpi-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.25s ease; height: 100%;
    }
    .kpi-card:hover {
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.10);
        transform: translateY(-2px); border-color: #cbd5e1;
    }
    .kpi-label { color: #64748b; font-size: 0.75rem; font-weight: 600;
                 letter-spacing: 0.06em; text-transform: uppercase;
                 margin-bottom: 0.35rem; }
    .kpi-value { color: #0f172a; font-size: 1.8rem; font-weight: 700;
                 line-height: 1.1; letter-spacing: -0.02em; }
    .kpi-value.accent { color: #0ea5e9; }
    .kpi-value.good   { color: #10b981; }
    .kpi-value.warn   { color: #f59e0b; }
    .kpi-value.danger { color: #ef4444; }

    .section-header {
        font-size: 1.35rem; font-weight: 700; color: #0f172a;
        margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0; letter-spacing: -0.01em;
    }
    .section-header .emoji { margin-right: 0.4rem; }

    .feature-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.2rem; height: 100%; transition: all 0.2s ease;
    }
    .feature-card:hover { border-color: #38bdf8;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.12); }
    .feature-card h4 { color: #0f172a; font-size: 1rem; font-weight: 700;
                       margin: 0.6rem 0 0.4rem 0; }
    .feature-card p  { color: #64748b; font-size: 0.875rem; margin: 0;
                       line-height: 1.5; }
    .feature-icon { font-size: 1.8rem; display: inline-block; }

    .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em; }
    .pill-blue  { background:#e0f2fe; color:#0369a1; }
    .pill-green { background:#dcfce7; color:#166534; }
    .pill-amber { background:#fef3c7; color:#92400e; }
    .pill-red   { background:#fee2e2; color:#991b1b; }

    .step-list { list-style: none; padding: 0; counter-reset: step; }
    .step-list li {
        counter-increment: step;
        padding: 0.75rem 1rem 0.75rem 3rem; margin-bottom: 0.5rem;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        position: relative; font-size: 0.92rem; color: #334155;
        transition: all 0.2s ease;
    }
    .step-list li::before {
        content: counter(step); position: absolute; left: 0.9rem;
        top: 50%; transform: translateY(-50%); width: 26px; height: 26px;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white; border-radius: 50%; display: flex;
        align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.8rem;
    }
    .step-list li:hover { border-color: #38bdf8; background: #f0f9ff;
        transform: translateX(4px); }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1 { color: #f8fafc !important;
        font-weight: 800 !important; letter-spacing: -0.02em; }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 0.75rem; border-radius: 8px;
        transition: all 0.15s ease; font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56, 189, 248, 0.15);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.2); }
    section[data-testid="stSidebar"] .caption,
    section[data-testid="stSidebar"] small { color: #94a3b8 !important;
        font-size: 0.75rem; }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.55rem 1.4rem; font-weight: 600; font-size: 0.9rem;
        transition: all 0.2s ease; box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.4);
        color: white; background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden;
        border: 1px solid #e2e8f0; }
    .stAlert { border-radius: 10px; border-left-width: 4px; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# SESSION STATE
# ==================================================================
def init_state():
    defaults = {
        "df_raw": None, "df_clean": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "X_res": None, "y_res": None,
        "feature_names": None, "scaler": None,
        "model_baseline": None,
        "model_balanced": None,
        "model_best": None,
        "model_name": None,
        "metrics": None,
        "grid_results": None,
        "rand_recall_results": None,
        "rand_prauc_results": None,
        "y_proba_best": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0f172a",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})
ACCENT = "#0ea5e9"
ACCENT2 = "#f59e0b"

# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;'>
        <span style='font-size:1.6rem;'>🛡️</span>
        <span style='font-size:1.25rem; font-weight:800; letter-spacing:-0.02em;'>
            Brute-Force Detection
        </span>
    </div>
    <div style='color:#94a3b8; font-size:0.78rem; margin-bottom:1.5rem;'>
        Random Forest · SMOTE · Threshold Tuning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧭 Navigation")
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📥 Load Data",
            "🧹 Data Cleaning",
            "⚖️ Class Imbalance (SMOTE)",
            "🌲 Baseline Models",
            "🔧 Grid Search",
            "🎲 Randomized Search (Recall)",
            "🎯 Randomized Search (PR-AUC)",
            "📉 Precision-Recall & Threshold",
            "📊 Feature Importance",
            "🔮 Predict",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#64748b; line-height:1.6;'>
        <b style='color:#94a3b8;'>⚙️ Environment</b><br>
        Python 3.11 · Streamlit Cloud<br>
        n_jobs=1 · cv=2 (cloud-safe)
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# UI HELPERS
# ==================================================================
def hero(title, subtitle, badge=None):
    badge_html = f"<div class='badge'>{badge}</div>" if badge else ""
    st.markdown(f"""
    <div class="hero">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section(title, emoji="📌"):
    st.markdown(
        f"<div class='section-header'><span class='emoji'>{emoji}</span>{title}</div>",
        unsafe_allow_html=True,
    )


def kpi(label, value, variant=""):
    cls = f"kpi-value {variant}".strip()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_row(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Accuracy",  f"{m['Accuracy']:.2f}%",  "accent")
    with c2: kpi("Precision", f"{m['Precision']:.2f}%", "good")
    with c3: kpi("Recall",    f"{m['Recall']:.2f}%",    "good")
    with c4: kpi("F1 Score",  f"{m['F1 Score']:.2f}%",  "accent")
    with c5: kpi("PR-AUC",    f"{m['PR-AUC']:.2f}%",    "warn")


def feature_card(icon, title, body):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
# ML HELPERS
# ==================================================================
@st.cache_data(show_spinner=False)
def load_from_kaggle():
    import kagglehub
    path = kagglehub.dataset_download(
        "dnkumars/cybersecurity-intrusion-detection-dataset"
    )
    csv_file = None
    for f in os.listdir(path):
        if f.endswith(".csv"):
            csv_file = os.path.join(path, f)
            break
    if csv_file is None:
        raise FileNotFoundError("No CSV found.")
    return pd.read_csv(csv_file)


def clean_dataframe(df: pd.DataFrame):
    """Drop session_id, fill NaNs, label-encode categoricals, dedup."""
    df = df.copy()

    # Drop non-predictive identifier
    if "session_id" in df.columns:
        df = df.drop("session_id", axis=1)

    # Handle missing values BEFORE encoding
    if "encryption_used" in df.columns:
        df["encryption_used"] = df["encryption_used"].fillna("None")

    # Label-encode categorical columns
    le = LabelEncoder()
    for col in ["protocol_type", "browser_type", "encryption_used"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_and_scale(df):
    X = df.drop("attack_detected", axis=1)
    y = df["attack_detected"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Encode any remaining object columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, feature_names, scaler


def evaluate_model(model, X_test, y_test, threshold=None):
    """Evaluate with optional custom probability threshold."""
    if threshold is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        proba = (model.predict_proba(X_test)[:, 1]
                 if hasattr(model, "predict_proba") else y_pred)

    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1 Score": f1_score(y_test, y_pred, zero_division=0) * 100,
        "PR-AUC": average_precision_score(y_test, proba) * 100,
    }, y_pred, proba


def plot_confusion(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test),
                cbar=False, ax=ax,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, pad=12)
    plt.tight_layout()
    return fig


def plot_pr_curve(y_test, y_proba, threshold=None, title="Precision-Recall Curve"):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions[:-1], color=ACCENT, lw=2.2, label="Precision")
    ax.plot(thresholds, recalls[:-1], color=ACCENT2, lw=2.2, label="Recall")
    if threshold is not None:
        ax.axvline(threshold, color="#dc2626", ls="--", lw=1.5,
                   label=f"Threshold = {threshold}")
    ax.set_xlabel("Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title(title, pad=12)
    ax.legend(loc="center left", frameon=True)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig


# ==================================================================
# PAGES
# ==================================================================

# ---------------- OVERVIEW ----------------
if page == "🏠 Overview":
    hero(
        "🛡️ Real-Time Brute-Force Detection",
        "Predictive machine learning with PR-curve tuning, threshold "
        "optimization, and model comparison.",
        badge="ML Pipeline · v2.0",
    )

    section("Pipeline Overview", "🔬")
    st.markdown("""
    <ol class="step-list">
        <li><b>Load</b> the Cybersecurity Intrusion Detection dataset</li>
        <li><b>Clean</b> — fill NaNs, label-encode, dedup</li>
        <li><b>Balance</b> with SMOTE</li>
        <li><b>Train</b> baseline + class-balanced Random Forests</li>
        <li><b>Tune</b> via Grid Search, then Randomized Search (Recall &amp; PR-AUC)</li>
        <li><b>Tune the probability threshold</b> on the Precision-Recall curve</li>
        <li><b>Visualize</b> metrics, confusion matrices, and feature importances</li>
    </ol>
    """, unsafe_allow_html=True)

    section("What Makes v2 Different", "✨")
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("🎯", "PR-AUC Optimization",
                     "Randomized Search scored on `average_precision` — "
                     "the right metric for imbalanced attack detection.")
    with c2:
        feature_card("📉", "Threshold Tuning",
                     "Predictions tuned on the Precision-Recall curve to "
                     "balance recall vs. false alarms.")
    with c3:
        feature_card("⚖️", "Class Balancing",
                     "`class_weight='balanced'` + SMOTE to make the model "
                     "care about the minority (attack) class.")

    st.info(
        "⚙️ **Cloud-safe:** searches use small grids, `n_jobs=1`, `cv=2` "
        "to fit the ~1 GB / 1 CPU Streamlit container."
    )

    section("At a Glance", "📊")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Model",       "Random Forest",  "accent")
    with c2: kpi("Imbalance",   "SMOTE + balanced", "good")
    with c3: kpi("Tuning",      "Grid + Rand.",   "accent")
    with c4: kpi("Key Metric",  "PR-AUC",         "warn")


# ---------------- LOAD DATA ----------------
elif page == "📥 Load Data":
    hero("📥 Load Dataset", "Fetch from Kaggle or upload your own CSV.",
         badge="Step 1 / 8")

    source = st.radio(
        "Choose a data source:",
        ["Kaggle (auto-download)", "Upload CSV"],
        horizontal=True,
    )

    if source == "Kaggle (auto-download)":
        if st.button("⬇️ Download from Kaggle"):
            with st.spinner("Downloading dataset..."):
                try:
                    df = load_from_kaggle()
                    st.session_state.df_raw = df
                    st.success(f"✅ Loaded **{len(df):,}** rows.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            st.session_state.df_raw = pd.read_csv(up)
            st.success(f"✅ Loaded **{len(st.session_state.df_raw):,}** rows.")

    df = st.session_state.df_raw
    if df is not None:
        section("Dataset Summary", "📊")
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Rows", f"{df.shape[0]:,}", "accent")
        with c2: kpi("Columns", f"{df.shape[1]}", "accent")
        with c3: kpi("Duplicates", f"{int(df.duplicated().sum())}", "warn")

        section("Preview", "👀")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("🔎 Data types / missing values / describe"):
            t1, t2, t3 = st.tabs(["Data Types", "Missing", "Statistics"])
            with t1:
                st.dataframe(df.dtypes.astype(str).rename("dtype"),
                             use_container_width=True)
            with t2:
                miss = df.isnull().sum().rename("missing")
                st.dataframe(miss, use_container_width=True)
                if miss.sum() > 0:
                    st.warning(f"⚠️ {miss.sum()} missing values detected. "
                               f"They will be handled in the Cleaning step.")
            with t3:
                st.dataframe(df.describe(), use_container_width=True)


# ---------------- DATA CLEANING ----------------
elif page == "🧹 Data Cleaning":
    hero("🧹 Data Cleaning",
         "Fill missing values, drop identifiers, label-encode categoricals, dedup.",
         badge="Step 2 / 8")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Load a dataset first."); st.stop()

    if st.button("▶️ Run Cleaning"):
        with st.spinner("Cleaning..."):
            st.session_state.df_clean = clean_dataframe(st.session_state.df_raw)
        st.success(f"✅ Cleaned: {len(st.session_state.df_raw):,} → "
                   f"{len(st.session_state.df_clean):,} rows")

    df = st.session_state.df_clean
    if df is not None:
        section("Cleaned Preview", "👀")
        st.dataframe(df.head(10), use_container_width=True)

        if "attack_detected" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                section("Target Distribution", "🎯")
                fig, ax = plt.subplots(figsize=(6, 4))
                vc = df["attack_detected"].value_counts()
                ax.bar(vc.index.astype(str), vc.values,
                       color=[ACCENT, ACCENT2], edgecolor="white", linewidth=1.5)
                ax.set_xlabel("attack_detected"); ax.set_ylabel("Count")
                for i, v in enumerate(vc.values):
                    ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                            fontweight="bold")
                st.pyplot(fig)
            with col2:
                section("Correlation Matrix", "🔥")
                num = df.select_dtypes(include=[np.number])
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(num.corr(), annot=True, cmap="coolwarm",
                            fmt=".2f", ax=ax, cbar_kws={"shrink": 0.8},
                            annot_kws={"fontsize": 8})
                st.pyplot(fig)

        if "session_duration" in df.columns:
            section("Outlier Check — session_duration", "📦")
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.boxplot(df["session_duration"].dropna(), vert=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="#bfdbfe", edgecolor=ACCENT),
                       medianprops=dict(color="#dc2626", linewidth=2))
            ax.set_xlabel("session_duration")
            st.pyplot(fig)


# ---------------- SMOTE ----------------
elif page == "⚖️ Class Imbalance (SMOTE)":
    hero("⚖️ Class Imbalance & SMOTE",
         "Balance attack vs. benign classes with synthetic minority oversampling.",
         badge="Step 3 / 8")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Run cleaning first."); st.stop()

    if st.button("▶️ Split & Apply SMOTE"):
        with st.spinner("Splitting + SMOTE..."):
            (X_train, X_test, y_train, y_test,
             names, scaler) = split_and_scale(st.session_state.df_clean)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_res = X_res
            st.session_state.y_res = y_res
            st.session_state.feature_names = names
            st.session_state.scaler = scaler
        st.success("✅ SMOTE applied.")

    if st.session_state.y_res is not None:
        section("Balancing Effect", "📊")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before SMOTE**")
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = st.session_state.y_train.value_counts()
            ax.bar(vc.index.astype(str), vc.values,
                   color=["#94a3b8", "#cbd5e1"], edgecolor="white", linewidth=1.5)
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                        fontweight="bold")
            ax.set_xlabel("attack_detected"); ax.set_ylabel("Count")
            st.pyplot(fig)
        with col2:
            st.markdown("**After SMOTE**")
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = st.session_state.y_res.value_counts()
            ax.bar(vc.index.astype(str), vc.values,
                   color=[ACCENT, ACCENT2], edgecolor="white", linewidth=1.5)
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                        fontweight="bold")
            ax.set_xlabel("attack_detected"); ax.set_ylabel("Count")
            st.pyplot(fig)

        section("Split Overview", "🧮")
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Train rows (pre-SMOTE)",
                     f"{len(st.session_state.y_train):,}", "accent")
        with c2: kpi("Train rows (post-SMOTE)",
                     f"{len(st.session_state.y_res):,}", "good")
        with c3: kpi("Test rows",
                     f"{len(st.session_state.y_test):,}", "accent")
        with c4: kpi("Features",
                     f"{len(st.session_state.feature_names)}", "warn")


# ---------------- BASELINE MODELS ----------------
elif page == "🌲 Baseline Models":
    hero("🌲 Baseline Models",
         "Train two baseline Random Forests: default vs. class_weight='balanced'.",
         badge="Step 4 / 8")

    if st.session_state.X_res is None:
        st.warning("⚠️ Run SMOTE first."); st.stop()

    if st.button("▶️ Train Both Baselines"):
        with st.spinner("Training baselines..."):
            rf_default = RandomForestClassifier(
                random_state=42, n_jobs=1
            )
            rf_balanced = RandomForestClassifier(
                random_state=42, n_jobs=1, class_weight="balanced"
            )
            rf_default.fit(st.session_state.X_res, st.session_state.y_res)
            rf_balanced.fit(st.session_state.X_res, st.session_state.y_res)
            st.session_state.model_baseline = rf_default
            st.session_state.model_balanced = rf_balanced

        st.success("✅ Both baseline models trained.")

    if st.session_state.model_balanced is not None:
        section("Side-by-Side Comparison", "⚖️")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌲 Default")
            m1, _, _ = evaluate_model(
                st.session_state.model_baseline,
                st.session_state.X_test, st.session_state.y_test)
            metric_row(m1)
        with c2:
            st.markdown("#### ⚖️ Balanced")
            m2, _, _ = evaluate_model(
                st.session_state.model_balanced,
                st.session_state.X_test, st.session_state.y_test)
            metric_row(m2)

        section("Confusion Matrices", "🔲")
        c1, c2 = st.columns(2)
        with c1:
            y_pred = st.session_state.model_baseline.predict(
                st.session_state.X_test)
            st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                     "Default RF"))
        with c2:
            y_pred = st.session_state.model_balanced.predict(
                st.session_state.X_test)
            st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                     "Balanced RF"))

        section("Precision-Recall Curve (Balanced)", "📉")
        proba = st.session_state
