"""
Credit Card Fraud Detection — Premium Streamlit Dashboard
Run with:  streamlit run streamlit_app.py
Place this file in your project root (same level as app.py / requirements.txt).
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
    
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection | Command Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "dataset/creditcard.csv"
MODEL_PATH = "models/fraud_detection_model.pkl"

DATA_URL = "https://drive.google.com/file/d/1lGPuRgXTnc2_r1GofpxUXjDiZVSzcOji/view?usp=drive_link"
# ----------------------------------------------------------------------------
# PREMIUM CSS THEME
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    html { scroll-behavior: smooth; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #16192b 0%, #0d0f1a 45%, #08090f 100%);
        color: #e7e9f5;
    }

    /* ---------- SCROLL REVEAL SYSTEM ---------- */
    .reveal {
        opacity: 0;
        transform: translateY(46px) scale(0.96);
        filter: blur(14px);
        transition: opacity 0.9s cubic-bezier(.22,.68,0,1),
                    transform 0.9s cubic-bezier(.22,.68,0,1),
                    filter 0.9s cubic-bezier(.22,.68,0,1);
        will-change: opacity, transform, filter;
    }
    .reveal.in-view {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes floaty {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11121c 0%, #0a0b12 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #d8dae8 !important; }

    h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }

    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(37,99,235,0.10));
        border: 1px solid rgba(148,163,255,0.18);
        margin-bottom: 26px;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #a78bfa, #60a5fa 45%, #34d399, #a78bfa);
        background-size: 200% auto;
        animation: shimmer 6s ease-in-out infinite;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero p { color: #9aa0c0; margin: 0; font-size: 0.95rem; }
    .hero { backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); }

    .kpi-card {
        padding: 20px 22px;
        border-radius: 16px;
        background: linear-gradient(155deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(167,139,250,0.45);
        box-shadow: 0 16px 34px rgba(124,58,237,0.22);
    }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9aa0c0; margin-bottom: 8px; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #f4f5fb; font-family: 'JetBrains Mono', monospace; }
    .kpi-sub { font-size: 0.8rem; margin-top: 6px; }
    .up { color: #34d399; } .down { color: #f87171; } .neutral { color: #9aa0c0; }

    .glass-panel {
        padding: 22px 24px;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 18px;
        transition: border-color .3s ease, box-shadow .3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(96,165,250,0.28);
        box-shadow: 0 14px 38px rgba(37,99,235,0.16);
    }

    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: .03em;
    }
    .badge-safe { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.35); }
    .badge-fraud { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.35); }

    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.4rem; font-weight: 700; letter-spacing: .02em;
        box-shadow: 0 6px 18px rgba(124,58,237,0.35);
    }
    .stButton>button:hover { filter: brightness(1.1); }

    /* ---------------- AMBIENT ANIMATED BACKGROUND ---------------- */
    .stApp::before {
        content: "";
        position: fixed; inset: 0; z-index: -1; pointer-events: none;
        background:
            radial-gradient(circle at 20% 20%, rgba(124,58,237,0.16), transparent 40%),
            radial-gradient(circle at 80% 30%, rgba(37,99,235,0.14), transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(52,211,153,0.10), transparent 45%);
        background-size: 200% 200%;
        animation: auroraDrift 22s ease-in-out infinite alternate;
    }
    @keyframes auroraDrift {
        0%   { background-position: 0% 0%, 100% 0%, 50% 100%; }
        100% { background-position: 30% 30%, 70% 40%, 40% 70%; }
    }

    /* ---------------- GLASSMORPHISM ---------------- */
    section[data-testid="stSidebar"] {
        background: rgba(10,11,18,0.65) !important;
        backdrop-filter: blur(18px) saturate(140%);
        -webkit-backdrop-filter: blur(18px) saturate(140%);
    }
    .kpi-card, .glass-panel, .hero {
        backdrop-filter: blur(14px) saturate(130%);
        -webkit-backdrop-filter: blur(14px) saturate(130%);
    }

    /* ---------------- HOVER GLOW / TILT ---------------- */
    .kpi-card {
        transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease, border-color .35s ease, filter .35s ease;
    }
    .kpi-card:hover {
        transform: translateY(-6px) scale(1.015);
        border-color: rgba(167,139,250,0.55);
        box-shadow: 0 18px 40px rgba(124,58,237,0.28), 0 0 0 1px rgba(167,139,250,0.15) inset;
    }
    .glass-panel {
        transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
    }
    .glass-panel:hover {
        border-color: rgba(96,165,250,0.35);
        box-shadow: 0 14px 34px rgba(37,99,235,0.16);
    }

    /* ---------------- SHIMMER ON BADGES ---------------- */
    .badge { position: relative; overflow: hidden; }
    .badge::after {
        content: ""; position: absolute; top: 0; left: -60%;
        width: 40%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
        animation: shimmer 2.6s ease-in-out infinite;
    }
    @keyframes shimmer { 0% { left: -60%; } 100% { left: 130%; } }

    /* ---------------- SCROLL-DRIVEN REVEAL ANIMATIONS ---------------- */
    /* Ties opacity/transform/blur directly to scroll position, so the
       effect replays every time an element re-enters the viewport —
       scrolling down plays it forward, scrolling back up reverses it,
       scrolling down again replays it. Supported in Chrome/Edge 115+. */
    @supports (animation-timeline: view()) {
        .hero {
            opacity: 0; transform: translateY(-30px) scale(0.97); filter: blur(10px);
            animation: revealFade linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 30%;
        }
        .kpi-card {
            opacity: 0; transform: translateY(50px) scale(0.94); filter: blur(8px);
            animation: revealUp linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 45%;
        }
        .glass-panel {
            opacity: 0; transform: translateY(60px) scale(0.96); filter: blur(10px);
            animation: revealUp linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 40%;
        }
        div[data-testid="stPlotlyChart"] {
            opacity: 0; transform: scale(0.92); filter: blur(6px);
            animation: revealScale linear both;
            animation-timeline: view();
            animation-range: entry 0% cover 50%;
        }
    }
    /* Fallback for browsers without scroll-timeline support: fully visible, no animation */
    @supports not (animation-timeline: view()) {
        .hero, .kpi-card, .glass-panel { opacity: 1; transform: none; filter: none; }
    }

    @keyframes revealUp {
        to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }
    @keyframes revealFade {
        to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }
    @keyframes revealScale {
        to { opacity: 1; transform: scale(1); filter: blur(0); }
    }

    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    ::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: #3b3f5c; border-radius: 8px; }

    /* Top toolbar / header bar */
    header[data-testid="stHeader"] {
        background: rgba(10,11,18,0.75) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    header[data-testid="stHeader"] * { color: #e7e9f5 !important; fill: #e7e9f5 !important; }
    div[data-testid="stToolbar"] { background: transparent !important; }
    div[data-testid="stDecoration"] { background: linear-gradient(90deg, #7c3aed, #2563eb, #34d399) !important; }
    #MainMenu { visibility: visible; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading transaction data...")
def load_data(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, None
    with open(path, "rb") as f:
        saved = pickle.load(f)
    if isinstance(saved, dict):
        return saved.get("model"), saved.get("scaler")
    return saved, None

df = load_data(DATA_PATH)
model, scaler = load_model(MODEL_PATH)

# ----------------------------------------------------------------------------
# SCROLL-TRIGGERED REVEAL ENGINE
# Replays the fade+blur animation every time an element re-enters the
# viewport (not just once), and re-scans for new elements after each
# Streamlit rerender (e.g. switching sidebar pages).
# ----------------------------------------------------------------------------
components.html("""
<script>
function initScrollReveal() {
    const doc = window.parent.document;
    if (!doc) return;

    if (!window.__revealObserver) {
        window.__revealObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                } else {
                    entry.target.classList.remove('in-view');
                }
            });
        }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });
    }

    const elements = doc.querySelectorAll('.reveal');
    elements.forEach((el) => {
        if (!el.dataset.revealBound) {
            el.dataset.revealBound = "true";
            window.__revealObserver.observe(el);
        }
    });
}
initScrollReveal();
setInterval(initScrollReveal, 700);
</script>
""", height=0, width=0)

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💳 Fraud Command Center")
    st.caption("Premium analytics dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Data Explorer", "🔍 Live Prediction", "🧠 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Status")
    st.markdown(
        f"Dataset: {'🟢 Loaded' if df is not None else '🔴 Missing'}  \n"
        f"Model: {'🟢 Loaded' if model is not None else '🔴 Missing'}  \n"
        f"Scaler: {'🟢 Loaded' if scaler is not None else '🟡 Not found'}"
    )

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero reveal">
    <h1>Credit Card Fraud Detection</h1>
    <p>Real-time transaction risk monitoring, model diagnostics, and instant fraud scoring — all in one place.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    if df is None:
        st.warning("⚠️ No dataset found at `dataset/creditcard.csv`. Add the Kaggle dataset to unlock this view.")
    else:
        total_tx = len(df)
        fraud_tx = int(df["Class"].sum())
        legit_tx = total_tx - fraud_tx
        fraud_rate = fraud_tx / total_tx * 100
        total_amount = df["Amount"].sum()
        fraud_amount = df.loc[df["Class"] == 1, "Amount"].sum()

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "Total Transactions", f"{total_tx:,}", "neutral", "Full dataset"),
            (c2, "Fraudulent Cases", f"{fraud_tx:,}", "down", f"{fraud_rate:.3f}% of all transactions"),
            (c3, "Legitimate Cases", f"{legit_tx:,}", "up", f"{100 - fraud_rate:.3f}% of all transactions"),
            (c4, "Fraud Amount Exposure", f"${fraud_amount:,.0f}", "down", f"of ${total_amount:,.0f} total volume"),
        ]
        for i, (col, label, value, tone, sub) in enumerate(cards):
            with col:
                st.markdown(f"""
                <div class="kpi-card reveal" style="transition-delay:{i*0.08:.2f}s;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub {tone}">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
            st.subheader("Transaction Volume Over Time")
            df_time = df.copy()
            df_time["Hour"] = (df_time["Time"] // 3600) % 24
            hourly = df_time.groupby(["Hour", "Class"]).size().reset_index(name="Count")
            hourly["Class"] = hourly["Class"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.area(
                hourly, x="Hour", y="Count", color="Class",
                color_discrete_map={"Legitimate": "#60a5fa", "Fraud": "#f87171"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360, legend_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
            st.subheader("Class Distribution")
            dist = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
            fig2 = px.pie(
                names=dist.index, values=dist.values, hole=0.62,
                color=dist.index, color_discrete_map={"Legitimate": "#60a5fa", "Fraud": "#f87171"},
                template="plotly_dark",
            )
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
        st.subheader("Transaction Amount Distribution (log scale)")
        fig3 = px.histogram(
            df, x="Amount", color=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=60, log_y=True, template="plotly_dark",
            color_discrete_map={"Legitimate": "#60a5fa", "Fraud": "#f87171"},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340, legend_title="", bargap=0.05)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "📊 Data Explorer":
    if df is None:
        st.warning("⚠️ No dataset found at `dataset/creditcard.csv`.")
    else:
        st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
        st.subheader("Filter Transactions")
        col1, col2, col3 = st.columns(3)
        with col1:
            class_filter = st.selectbox("Class", ["All", "Legitimate only", "Fraud only"])
        with col2:
            amt_range = st.slider("Amount range ($)", 0.0, float(df["Amount"].max()), (0.0, float(df["Amount"].max())))
        with col3:
            sample_n = st.number_input("Rows to display", min_value=10, max_value=5000, value=200, step=10)

        filtered = df.copy()
        if class_filter == "Legitimate only":
            filtered = filtered[filtered["Class"] == 0]
        elif class_filter == "Fraud only":
            filtered = filtered[filtered["Class"] == 1]
        filtered = filtered[(filtered["Amount"] >= amt_range[0]) & (filtered["Amount"] <= amt_range[1])]

        st.dataframe(filtered.head(int(sample_n)), use_container_width=True, height=340)
        st.caption(f"Showing {min(sample_n, len(filtered)):,} of {len(filtered):,} matching rows")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
        st.subheader("Feature Correlation Heatmap")
        sample_for_corr = df.sample(min(20000, len(df)), random_state=42)
        corr = sample_for_corr.corr(numeric_only=True)
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark", aspect="auto",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=560)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: LIVE PREDICTION
# ============================================================================
elif page == "🔍 Live Prediction":
    if model is None:
        st.error("🔴 No trained model found at `models/fraud_detection_model.pkl`. Train and save a model first.")
    else:
        st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
        st.subheader("Score a Transaction")

        mode = st.radio("Input method", ["Paste feature values", "Pick a random transaction from dataset"], horizontal=True)

        feature_values = None
        if mode == "Paste feature values":
            raw = st.text_area(
                "30 comma-separated values: Time, V1..V28, Amount",
                placeholder="0, -1.36, -0.07, 2.54, ..., 149.62",
                height=90,
            )
            if raw.strip():
                try:
                    feature_values = [float(x.strip()) for x in raw.split(",")]
                    if len(feature_values) != 30:
                        st.error(f"Expected 30 values, got {len(feature_values)}.")
                        feature_values = None
                except ValueError:
                    st.error("Could not parse values — make sure they're all numbers separated by commas.")
        else:
            if df is not None:
                draw_type = st.radio(
                    "Draw from",
                    ["Any transaction", "Known fraud only", "Known legitimate only"],
                    horizontal=True,
                )
                if st.button("🎲 Draw random transaction"):
                    if draw_type == "Known fraud only":
                        pool = df[df["Class"] == 1]
                    elif draw_type == "Known legitimate only":
                        pool = df[df["Class"] == 0]
                    else:
                        pool = df
                    row = pool.drop(columns=["Class"]).sample(1).iloc[0]
                    st.session_state["picked_row"] = row.tolist()
                if "picked_row" in st.session_state:
                    feature_values = st.session_state["picked_row"]
                    st.code(", ".join(f"{v:.4f}" for v in feature_values), language="text")
            else:
                st.info("Dataset not available — use manual paste instead.")

        st.markdown('</div>', unsafe_allow_html=True)

        if feature_values is not None:
            arr = np.array(feature_values, dtype=float).reshape(1, -1)
            if scaler is not None:
                try:
                    arr[0, 0] = scaler.transform([[arr[0, 0]]])[0][0]
                    arr[0, -1] = scaler.transform([[arr[0, -1]]])[0][0]
                except Exception:
                    st.info("Scaler present but could not be applied to Time/Amount — using raw values.")

            pred = model.predict(arr)[0]
            prob = float(model.predict_proba(arr)[0][1]) if hasattr(model, "predict_proba") else float(pred)

            col1, col2 = st.columns([1, 1.4])
            with col1:
                badge_class = "badge-fraud" if pred == 1 else "badge-safe"
                badge_text = "⚠️ FRAUD DETECTED" if pred == 1 else "✅ LEGITIMATE"
                st.markdown(f"""
                <div class="kpi-card reveal" style="text-align:center; padding:30px;">
                    <div class="badge {badge_class}" style="font-size:1rem; padding:10px 20px;">{badge_text}</div>
                    <div style="margin-top:18px; font-size:0.85rem; color:#9aa0c0;">Fraud Probability</div>
                    <div class="kpi-value" style="font-size:2.6rem;">{prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#9aa0c0"},
                        "bar": {"color": "#f87171" if prob > 0.5 else "#34d399"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(52,211,153,0.25)"},
                            {"range": [30, 70], "color": "rgba(251,191,36,0.25)"},
                            {"range": [70, 100], "color": "rgba(248,113,113,0.25)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                    },
                ))
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "🧠 Model Performance":
    if model is None or df is None:
        st.warning("⚠️ Need both a trained model and the dataset to compute performance metrics.")
    else:
        with st.spinner("Evaluating model on a held-out split..."):
            X = df.drop(columns=["Class"]).copy()
            y = df["Class"]
            if scaler is not None:
                try:
                    X["Time"] = scaler.transform(X[["Time"]])
                    X["Amount"] = scaler.transform(X[["Amount"]])
                except Exception:
                    pass
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Accuracy", report["accuracy"]),
            ("Precision (Fraud)", report["1"]["precision"]),
            ("Recall (Fraud)", report["1"]["recall"]),
            ("ROC-AUC", roc_auc),
        ]
        for i, (col, (label, val)) in enumerate(zip([c1, c2, c3, c4], metrics)):
            with col:
                st.markdown(f"""
                <div class="kpi-card reveal" style="transition-delay:{i*0.08:.2f}s;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
            st.subheader("Confusion Matrix")
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
            st.subheader("ROC Curve")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color="#a78bfa", width=3)))
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#4b5563", dash="dash")))
            fig2.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
        st.subheader("Precision–Recall Curve")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color="#34d399", width=3), fill="tozeroy"))
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=340, xaxis_title="Recall", yaxis_title="Precision",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if hasattr(model, "feature_importances_"):
            st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
            st.subheader("Top Feature Importances")
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig4 = px.bar(
                x=imp.values, y=imp.index, orientation="h",
                template="plotly_dark", color=imp.values, color_continuous_scale="Purples",
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=440, coloraxis_showscale=False, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="glass-panel reveal">', unsafe_allow_html=True)
    st.subheader("About This Project")
    st.write("""
This dashboard visualizes and interacts with a machine-learning model trained to detect
fraudulent credit card transactions from anonymized, PCA-transformed features (`V1`–`V28`),
alongside `Time` and `Amount`.

**Pages**
- **Overview** — headline KPIs and dataset-wide trends
- **Data Explorer** — filterable transaction table and correlation heatmap
- **Live Prediction** — score a transaction manually or by sampling the dataset
- **Model Performance** — confusion matrix, ROC/PR curves, and feature importances

**Stack:** Python, scikit-learn, Streamlit, Plotly
""")
    st.markdown('</div>', unsafe_allow_html=True)
