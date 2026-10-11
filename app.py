# -*- coding: utf-8 -*-
"""
Real-Time Brute-Force Detection — Inference Dashboard
Loads a pre-trained model and lets users test it with feature inputs.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from textwrap import dedent
from pathlib import Path

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Brute-Force Detection — Inference",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CSS
# ==================================================================
st.markdown(dedent("""
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
               position: relative; z-index: 1; }
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
    .kpi-card:hover { box-shadow: 0 8px 25px rgba(15,23,42,0.10);
        transform: translateY(-2px); border-color: #cbd5e1; }
    .kpi-label { color: #64748b; font-size: 0.75rem; font-weight: 600;
                 letter-spacing: 0.06em; text-transform: uppercase;
                 margin-bottom: 0.35rem; }
    .kpi-value { color: #0f172a; font-size: 1.8rem; font-weight: 700;
                 line-height: 1.1; letter-spacing: -0.02em;
                 font-variant-numeric: tabular-nums; }
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

    .verdict-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981; border-radius: 18px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .verdict-attack {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444; border-radius: 18px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .verdict-safe h2   { color: #065f46; margin: 0; font-size: 2rem; }
    .verdict-attack h2 { color: #991b1b; margin: 0; font-size: 2rem; }
    .verdict-safe p, .verdict-attack p { margin: 0.5rem 0 0 0;
        font-size: 1rem; opacity: 0.75; }

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
    section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,0.2); }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.55rem 1.4rem; font-weight: 600; font-size: 0.9rem;
        transition: all 0.2s ease; box-shadow: 0 2px 8px rgba(14,165,233,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14,165,233,0.4);
        color: white;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 2px 8px rgba(16,185,129,0.3) !important;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden;
        border: 1px solid #e2e8f0; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""").strip(), unsafe_allow_html=True)

# ==================================================================
# LOAD ARTIFACTS (cached)
# ==================================================================
BASE = Path(__file__).parent

@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    model  = joblib.load(BASE / "model.pkl")
    scaler = joblib.load(BASE / "scaler.pkl")
    with open(BASE / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(BASE / "encoders.json") as f:
        encoders = json.load(f)
    return model, scaler, feature_names, encoders

try:
    MODEL, SCALER, FEATURE_NAMES, ENCODERS = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "❌ Required model artifacts are missing. Please ensure "
        "`model.pkl`, `scaler.pkl`, `feature_names.json`, and "
        "`encoders.json` are committed to the repo.\n\n"
        f"Missing file: `{e.filename}`"
    )
    st.stop()

# ==================================================================
# UI HELPERS
# ==================================================================
def hero(title, subtitle, badge=None):
    badge_html = f"<div class='badge'>{badge}</div>" if badge else ""
    html = dedent(f"""
    <div class="hero">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def section(title, emoji="📌"):
    html = dedent(f"""
    <div class='section-header'>
        <span class='emoji'>{emoji}</span>{title}
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def kpi(label, value, variant=""):
    cls = f"kpi-value {variant}".strip()
    html = dedent(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def verdict_card(is_attack, probability):
    if is_attack:
        html = dedent(f"""
        <div class="verdict-attack">
            <h2>⚠️ Brute-Force Attack Detected</h2>
            <p>Predicted attack probability: <b>{probability:.2%}</b></p>
        </div>
        """).strip()
    else:
        html = dedent(f"""
        <div class="verdict-safe">
            <h2>✅ Session Appears Benign</h2>
            <p>Predicted attack probability: <b>{probability:.2%}</b></p>
        </div>
        """).strip()
    st.markdown(html, unsafe_allow_html=True)


# ==================================================================
# PREDICTION HELPERS
# ==================================================================
def prepare_input(input_dict):
    """Convert a raw feature dict into a scaled numpy array."""
    row = {}
    for feat in FEATURE_NAMES:
        val = input_dict.get(feat, 0)

        # Apply encoder for categorical features
        if feat in ENCODERS:
            # val may already be an encoded int (from a selectbox)
            row[feat] = val
        else:
            row[feat] = float(val)

    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    return SCALER.transform(X)


def predict_single(input_dict):
    X_scaled = prepare_input(input_dict)
    proba = MODEL.predict_proba(X_scaled)[0, 1]
    pred  = int(proba >= 0.5)
    return pred, proba


def predict_batch(df):
    """Batch prediction on a cleaned DataFrame."""
    X = df.reindex(columns=FEATURE_NAMES, fill_value=0)
    X_scaled = SCALER.transform(X)
    probas = MODEL.predict_proba(X_scaled)[:, 1]
    preds  = (probas >= 0.5).astype(int)
    return preds, probas


# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown(dedent("""
    <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;'>
        <span style='font-size:1.6rem;'>🛡️</span>
        <span style='font-size:1.25rem; font-weight:800; letter-spacing:-0.02em;'>
            Brute-Force Detection
        </span>
    </div>
    <div style='color:#94a3b8; font-size:0.78rem; margin-bottom:1.5rem;'>
        Inference Dashboard · Pre-trained Model
    </div>
    """).strip(), unsafe_allow_html=True)

    st.markdown("##### 🧭 Navigation")
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🎛️ Single Session Test",
            "📂 Batch Prediction (CSV)",
            "📊 Model Information",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(dedent(f"""
    <div style='font-size:0.72rem; color:#64748b; line-height:1.6;'>
        <b style='color:#94a3b8;'>⚙️ Model Status</b><br>
        ✅ Loaded: Random Forest<br>
        ✅ Features: {len(FEATURE_NAMES)}<br>
        ✅ Scaler: StandardScaler<br>
    </div>
    """).strip(), unsafe_allow_html=True)

# ==================================================================
# PAGES
# ==================================================================

# ---------------- OVERVIEW ----------------
if page == "🏠 Overview":
    hero(
        "🛡️ Real-Time Brute-Force Detection",
        "Test a pre-trained Random Forest classifier by adjusting "
        "network session features.",
        badge="Inference · v1.0",
    )

    section("How to Use This System", "💡")
    st.markdown(dedent("""
    <ol style='font-size:1rem; line-height:1.8; color:#334155;'>
        <li><b>Single Session Test</b> — adjust feature sliders to simulate
            a network session, then click <i>Predict</i> to see if the model
            classifies it as benign or an attack.</li>
        <li><b>Batch Prediction</b> — upload a CSV of many sessions to score
            them all at once.</li>
        <li><b>Model Information</b> — inspect the loaded model's structure
            and feature importances.</li>
    </ol>
    """).strip(), unsafe_allow_html=True)

    section("Loaded Model", "🧠")
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Algorithm", "Random Forest", "accent")
    with c2: kpi("Features",  f"{len(FEATURE_NAMES)}", "accent")
    with c3: kpi("Estimators", f"{MODEL.n_estimators}", "good")

    section("About the Model", "📖")
    st.markdown(dedent(f"""
    This dashboard serves a **Random Forest classifier** trained on the
    *Cybersecurity Intrusion Detection* dataset. The model:

    - Was trained with **{MODEL.n_estimators} decision trees**
    - Uses `class_weight='balanced'` to handle class imbalance
    - Consumes **{len(FEATURE_NAMES)} engineered features**
    - Outputs the probability that a session is a brute-force attack

    **Training happened offline once.** This app only performs inference.
    """).strip())


# ---------------- SINGLE SESSION TEST ----------------
elif page == "🎛️ Single Session Test":
    hero("🎛️ Single Session Test",
         "Adjust the features below and click Predict to test the model.",
         badge="Interactive Inference")

    section("Session Features", "🎚️")

    # ---------- Feature input form ----------
    with st.form("session_form"):
        # --- Row 1: Numeric features ---
        c1, c2, c3 = st.columns(3)

        with c1:
            network_packet_size = st.slider(
                "network_packet_size",
                min_value=0, max_value=2000, value=500, step=10,
                help="Size of the network packet in bytes.")

            login_attempts = st.slider(
                "login_attempts",
                min_value=0, max_value=20, value=1, step=1,
                help="Number of login attempts in this session.")

        with c2:
            session_duration = st.slider(
                "session_duration",
                min_value=0.0, max_value=2000.0, value=60.0, step=1.0,
                help="Duration of the session in seconds.")

            failed_logins = st.slider(
                "failed_logins",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of failed login attempts.")

        with c3:
            ip_reputation_score = st.slider(
                "ip_reputation_score",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="Reputation score of the source IP (0 = bad, 1 = good).")

            unusual_time_access = st.selectbox(
                "unusual_time_access",
                options=[0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                help="Was the access at an unusual time?")

        # --- Row 2: Categorical features ---
        c4, c5, c6 = st.columns(3)

        with c4:
            protocol_type = st.selectbox(
                "protocol_type",
                options=list(ENCODERS.get("protocol_type", {"tcp": 0}).keys()),
                help="Network protocol used.")

        with c5:
            browser_type = st.selectbox(
                "browser_type",
                options=list(ENCODERS.get("browser_type", {"chrome": 0}).keys()),
                help="Browser used in the session.")

        with c6:
            encryption_used = st.selectbox(
                "encryption_used",
                options=list(ENCODERS.get("encryption_used", {"None": 0}).keys()),
                help="Encryption protocol used.")

        submit = st.form_submit_button("🔮 Predict")

    # ---------- Handle submission ----------
    if submit:
        input_dict = {
            "network_packet_size":  network_packet_size,
            "protocol_type":        ENCODERS["protocol_type"][protocol_type],
            "login_attempts":       login_attempts,
            "session_duration":     session_duration,
            "encryption_used":      ENCODERS["encryption_used"][encryption_used],
            "ip_reputation_score":  ip_reputation_score,
            "failed_logins":        failed_logins,
            "browser_type":         ENCODERS["browser_type"][browser_type],
            "unusual_time_access":  unusual_time_access,
        }

        pred, proba = predict_single(input_dict)

        section("Prediction Result", "🎯")
        verdict_card(pred == 1, proba)

        c1, c2, c3 = st.columns(3)
        with c1: kpi("Prediction", "ATTACK" if pred == 1 else "BENIGN",
                     "danger" if pred == 1 else "good")
        with c2: kpi("Attack Probability", f"{proba:.2%}", "warn")
        with c3: kpi("Confidence", f"{max(proba, 1-proba):.2%}", "accent")

        # Probability gauge
        section("Probability Breakdown", "📊")
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([""], [1 - proba], color="#10b981", edgecolor="white",
                label="Benign", height=0.5)
        ax.barh([""], [proba], left=[1 - proba], color="#ef4444",
                edgecolor="white", label="Attack", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks([])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4),
                  ncol=2, frameon=False)
        st.pyplot(fig)

        # Input summary
        with st.expander("📋 Submitted values"):
            st.dataframe(pd.DataFrame([input_dict]).T.rename(
                columns={0: "Value"}), use_container_width=True)


# ---------------- BATCH PREDICTION ----------------
elif page == "📂 Batch Prediction (CSV)":
    hero("📂 Batch Prediction",
         "Upload a CSV with the same feature columns to score many sessions.",
         badge="Bulk Inference")

    st.info(
        f"📌 Expected columns: **{', '.join(FEATURE_NAMES)}**  \n"
        "Extra columns (like `session_id`) are ignored."
    )

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)

        # Apply same cleaning as training
        clean = raw.copy()
        if "session_id" in clean.columns:
            clean = clean.drop(columns=["session_id"])
        if "encryption_used" in clean.columns:
            clean["encryption_used"] = clean["encryption_used"].fillna("None")

        # Encode categoricals using saved mappings
        for col, mapping in ENCODERS.items():
            if col in clean.columns:
                clean[col] = clean[col].astype(str).map(mapping).fillna(-1).astype(int)

        # Predict
        preds, probas = predict_batch(clean)

        out = raw.copy()
        out["attack_probability"] = probas.round(4)
        out["prediction"] = preds

        n_attacks = int((preds == 1).sum())
        n_safe = int((preds == 0).sum())

        section("Summary", "📊")
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Sessions Scored", f"{len(out):,}", "accent")
        with c2: kpi("Attacks Detected", f"{n_attacks:,}", "danger")
        with c3: kpi("Benign Sessions", f"{n_safe:,}", "good")
        with c4: kpi("Attack Rate",
                     f"{n_attacks / len(out) * 100:.1f}%", "warn")

        section("Predictions Preview", "👀")
        st.dataframe(out.head(50), use_container_width=True)

        section("Prediction Distribution", "📈")
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = out["prediction"].value_counts()
        colors = ["#10b981" if v == 0 else "#ef4444" for v in vc.index]
        ax.bar(vc.index.astype(str), vc.values, color=colors,
               edgecolor="white", linewidth=1.5)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                    fontweight="bold")
        ax.set_xlabel("Prediction (0 = benign, 1 = attack)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        section("Probability Distribution", "🎯")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(probas, bins=40, color="#0ea5e9",
                edgecolor="white", alpha=0.85)
        ax.axvline(0.5, color="#dc2626", ls="--", lw=2, label="Threshold = 0.50")
        ax.set_xlabel("Predicted attack probability")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "⬇️ Download Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


# ---------------- MODEL INFORMATION ----------------
elif page == "📊 Model Information":
    hero("📊 Model Information",
         "Inspect the loaded model's structure and feature importances.",
         badge="Model Card")

    section("Model Summary", "🧠")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Algorithm", "Random Forest", "accent")
    with c2: kpi("Trees", f"{MODEL.n_estimators}", "accent")
    with c3: kpi("Max Depth", str(MODEL.max_depth), "accent")
    with c4: kpi("Features", f"{len(FEATURE_NAMES)}", "good")

    section("Hyperparameters", "⚙️")
    params = {
        "n_estimators":      MODEL.n_estimators,
        "max_depth":         MODEL.max_depth,
        "min_samples_split": MODEL.min_samples_split,
        "min_samples_leaf":  MODEL.min_samples_leaf,
        "max_features":      MODEL.max_features,
        "class_weight":      MODEL.class_weight,
        "random_state":      MODEL.random_state,
        "criterion":         MODEL.criterion,
    }
    st.dataframe(pd.DataFrame([params]).T.rename(columns={0: "Value"}),
                 use_container_width=True)

    section("Feature Importances", "🏅")
    imps = MODEL.feature_importances_
    imp_df = (pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": imps})
              .sort_values("Importance", ascending=False).reset_index(drop=True))

    top_n = st.slider("Top N features", 3, len(imp_df),
                      min(15, len(imp_df)))

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    palette = sns.color_palette("Blues_r", top_n)
    sns.barplot(data=imp_df.head(top_n), x="Importance", y="Feature",
                ax=ax, palette=palette)
    for i, v in enumerate(imp_df.head(top_n)["Importance"].values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9,
                fontweight="bold", color="#334155")
    ax.set_xlabel("Relative importance")
    st.pyplot(fig)

    with st.expander("📄 Full feature list"):
        st.dataframe(imp_df, use_container_width=True)

    section("Encoder Mappings", "🔤")
    for col, mapping in ENCODERS.items():
        st.markdown(f"**{col}**")
        st.dataframe(pd.DataFrame([mapping]).T.rename(
            columns={0: "Encoded Value"}), use_container_width=True)