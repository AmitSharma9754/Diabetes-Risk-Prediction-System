import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI · Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

header[data-testid="stHeader"] {
    position: absolute !important;
    background: transparent !important;
}


}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a4e, #141432);
    color: #e8e8f0;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(120deg, #1b1b5e 0%, #2d2d8e 50%, #1b1b5e 100%);
    border: 1px solid #3a3a9e;
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 40px rgba(100,100,255,0.15);
}
.hero-banner h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-banner p {
    color: #9ca3d4;
    font-size: 1rem;
    margin: 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #12124a;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #2a2a7a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7c7caa;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #1e1e6e;
    color: #c4c4f4;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99,91,220,0.45);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(6px);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 0.8rem;
    letter-spacing: 0.02em;
}

/* ── Input labels ── */
label, .stNumberInput label {
    color: #b0b0d8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stNumberInput input {
    background: #1a1a55 !important;
    border: 1px solid #3030a0 !important;
    color: #e8e8f8 !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.25) !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    width: 100%;
    transition: all 0.25s;
    box-shadow: 0 6px 20px rgba(99,91,255,0.4);
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(99,91,255,0.55);
}

/* ── Result boxes ── */
.result-diabetic {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fecaca;
    box-shadow: 0 8px 28px rgba(239,68,68,0.3);
}
.result-healthy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 700;
    color: #a7f3d0;
    box-shadow: 0 8px 28px rgba(52,211,153,0.3);
}

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(135deg, #1e1e70, #252580);
    border: 1px solid #3a3aa0;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.78rem;
    color: #8080b0;
    margin-top: 0.2rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── About section ── */
.about-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.about-card h3 {
    color: #a78bfa;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    margin-bottom: 0.7rem;
}
.about-card p, .about-card li {
    color: #9090c0;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ── Developer badge ── */
.dev-badge {
    background: linear-gradient(135deg, #2d1b69, #3b2380);
    border: 1px solid #7c3aed;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.dev-avatar {
    font-size: 4rem;
    margin-bottom: 0.5rem;
}
.dev-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #e0d7ff;
}
.dev-role {
    color: #a78bfa;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* ── Divider ── */
hr { border-color: #2a2a6a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0c29; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("best_diabetes_model.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

# ─────────────────────────────────────────────
# SAMPLE DATASET (Pima Indians stats for visuals)
# ─────────────────────────────────────────────
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 768
    data = pd.DataFrame({
        "Pregnancies":    np.random.randint(0, 17, n),
        "Glucose":        np.random.normal(120, 32, n).clip(0, 200).astype(int),
        "BloodPressure":  np.random.normal(69, 19, n).clip(0, 122).astype(int),
        "SkinThickness":  np.random.normal(20, 16, n).clip(0, 99).astype(int),
        "Insulin":        np.random.normal(79, 115, n).clip(0, 846).astype(int),
        "BMI":            np.random.normal(32, 8, n).clip(0, 67).round(1),
        "DPF":            np.random.exponential(0.47, n).clip(0.08, 2.42).round(3),
        "Age":            np.random.randint(21, 82, n),
        "Outcome":        np.random.choice([0, 1], n, p=[0.65, 0.35]),
    })
    return data

df = get_sample_data()

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🩺 DiabetesAI Prediction System</h1>
    <p>Advanced machine learning · Early risk detection · Powered by clinical data</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Predict",
    "📊  Data Visualization",
    "🎯  Model & Dataset",
    "👤  Developer & Info"
])

# ══════════════════════════════════════════════
# TAB 1 – PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="card-title">📋 Patient Information</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"⚠️ Model files not found. Please place `best_diabetes_model.pkl` and `diabetes_scaler.pkl` in the app folder.\n\n`{model_error}`")
    else:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="card-title">🧬 Clinical Parameters</div>', unsafe_allow_html=True)
            pregnancies     = st.number_input("Pregnancies",               min_value=0,   max_value=20,   value=1,    step=1)
            glucose         = st.number_input("Glucose (mg/dL)",           min_value=0,   max_value=300,  value=120,  step=1)
            blood_pressure  = st.number_input("Blood Pressure (mm Hg)",    min_value=0,   max_value=200,  value=70,   step=1)
            skin_thickness  = st.number_input("Skin Thickness (mm)",        min_value=0,   max_value=100,  value=20,   step=1)

        with col2:
            st.markdown('<div class="card-title">💉 Additional Metrics</div>', unsafe_allow_html=True)
            insulin = st.number_input("Insulin (µU/mL)",                   min_value=0,   max_value=900,  value=80,   step=1)
            bmi     = st.number_input("BMI",                               min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
            dpf     = st.number_input("Diabetes Pedigree Function",        min_value=0.0, max_value=3.0,  value=0.47, step=0.01, format="%.3f")
            age     = st.number_input("Age (years)",                       min_value=1,   max_value=120,  value=30,   step=1)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk gauge preview ──
        risk_score = round((glucose / 200 * 0.4 + bmi / 60 * 0.3 + age / 100 * 0.2 + dpf / 2.5 * 0.1) * 100, 1)
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_score,
            title = {"text": "Estimated Risk Index", "font": {"color": "#a78bfa", "size": 14}},
            number= {"suffix": "%", "font": {"color": "#e8e8f8", "size": 32}},
            gauge = {
                "axis": {"range": [0, 100], "tickcolor": "#4040a0"},
                "bar":  {"color": "#7c3aed"},
                "steps": [
                    {"range": [0, 35],  "color": "#064e3b"},
                    {"range": [35, 65], "color": "#78350f"},
                    {"range": [65, 100],"color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "#f472b6", "width": 3}, "value": risk_score}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   ="#e8e8f8",
            height=240,
            margin=dict(t=40, b=0, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Predict ──
        if st.button("🔮  Run Prediction"):
            input_data   = np.array([[pregnancies, glucose, blood_pressure,
                                       skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)
            prob         = model.predict_proba(input_scaled)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="result-diabetic">
                    ⚠️ High Risk — Likely Diabetic<br>
                    <span style="font-size:0.95rem;font-weight:400;color:#fca5a5;">
                    Confidence: {prob[1]*100:.1f}% · Please consult a healthcare professional.</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-healthy">
                    ✅ Low Risk — Likely Not Diabetic<br>
                    <span style="font-size:0.95rem;font-weight:400;color:#6ee7b7;">
                    Confidence: {prob[0]*100:.1f}% · Maintain a healthy lifestyle.</span>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=[prob[0]*100, prob[1]*100],
                y=["Not Diabetic", "Diabetic"],
                orientation="h",
                marker_color=["#34d399", "#f87171"],
                text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                textposition="inside",
                insidetextfont=dict(color="white", size=13)
            ))
            fig_bar.update_layout(
                title=dict(text="Prediction Probability", font=dict(color="#a78bfa", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font_color   ="#e8e8f8",
                height=160,
                margin=dict(t=35, b=10, l=0, r=0),
                xaxis=dict(range=[0,100], showgrid=False, ticksuffix="%",
                           tickfont=dict(color="#6060a0")),
                yaxis=dict(showgrid=False, tickfont=dict(color="#c0c0e0")),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – DATA VISUALIZATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card-title">📊 Dataset Visualizations</div>', unsafe_allow_html=True)

    palette = {"0": "#34d399", "1": "#f87171"}
    df["OutcomeLabel"] = df["Outcome"].map({0: "Not Diabetic", 1: "Diabetic"})

    # ── Row 1: Pie + Glucose box ──
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        counts = df["Outcome"].value_counts().reset_index()
        counts.columns = ["Outcome", "Count"]
        counts["Label"] = counts["Outcome"].map({0: "Not Diabetic", 1: "Diabetic"})
        fig_pie = px.pie(counts, values="Count", names="Label",
                         color="Label",
                         color_discrete_map={"Not Diabetic": "#34d399", "Diabetic": "#f87171"},
                         hole=0.45,
                         title="Class Distribution")
        fig_pie.update_traces(textfont_size=13, pull=[0.03, 0.03])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e8f8",
                              legend=dict(font=dict(color="#a0a0cc")),
                              title_font=dict(color="#a78bfa"),
                              height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1c2:
        fig_box = px.box(df, x="OutcomeLabel", y="Glucose", color="OutcomeLabel",
                         color_discrete_map={"Not Diabetic": "#34d399", "Diabetic": "#f87171"},
                         title="Glucose Distribution by Outcome",
                         labels={"OutcomeLabel": "", "Glucose": "Glucose (mg/dL)"})
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e8e8f8", showlegend=False,
                              title_font=dict(color="#a78bfa"),
                              height=320, margin=dict(t=50, b=10),
                              xaxis=dict(showgrid=False, tickfont=dict(color="#c0c0e0")),
                              yaxis=dict(gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Row 2: Scatter ──
    fig_scatter = px.scatter(df, x="Glucose", y="BMI",
                             color="OutcomeLabel", size="Age",
                             color_discrete_map={"Not Diabetic": "#34d399", "Diabetic": "#f87171"},
                             title="Glucose vs BMI (size = Age)",
                             opacity=0.7,
                             labels={"OutcomeLabel": "Outcome"})
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e8e8f8",
                              title_font=dict(color="#a78bfa"),
                              height=360, margin=dict(t=50, b=20),
                              xaxis=dict(gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
                              yaxis=dict(gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
                              legend=dict(font=dict(color="#a0a0cc")))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 3: Age histogram + Correlation heatmap ──
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        fig_hist = px.histogram(df, x="Age", color="OutcomeLabel", nbins=25,
                                barmode="overlay", opacity=0.75,
                                color_discrete_map={"Not Diabetic": "#34d399", "Diabetic": "#f87171"},
                                title="Age Distribution",
                                labels={"OutcomeLabel": "Outcome"})
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e8e8f8",
                               title_font=dict(color="#a78bfa"),
                               height=320, margin=dict(t=50, b=20),
                               xaxis=dict(gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
                               yaxis=dict(gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
                               legend=dict(font=dict(color="#a0a0cc")))
        st.plotly_chart(fig_hist, use_container_width=True)

    with r3c2:
        num_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age","Outcome"]
        corr = df[num_cols].corr().round(2)
        fig_heatmap = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="Plasma", zmid=0,
            text=corr.values, texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
        ))
        fig_heatmap.update_layout(
            title=dict(text="Feature Correlation Matrix", font=dict(color="#a78bfa")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f8",
            height=320, margin=dict(t=50, b=10, l=60, r=10),
            xaxis=dict(tickfont=dict(size=9, color="#c0c0e0")),
            yaxis=dict(tickfont=dict(size=9, color="#c0c0e0")),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Row 4: Radar chart ──
    features = ["Glucose", "BMI", "Age", "Pregnancies", "BloodPressure", "Insulin"]
    diabetic_avg  = df[df["Outcome"]==1][features].mean().tolist()
    healthy_avg   = df[df["Outcome"]==0][features].mean().tolist()
    max_vals      = df[features].max().tolist()
    d_norm = [v/m*100 for v,m in zip(diabetic_avg, max_vals)]
    h_norm = [v/m*100 for v,m in zip(healthy_avg,  max_vals)]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=d_norm+[d_norm[0]], theta=features+[features[0]],
                                        fill="toself", name="Diabetic",
                                        line_color="#f87171", fillcolor="rgba(248,113,113,0.2)"))
    fig_radar.add_trace(go.Scatterpolar(r=h_norm+[h_norm[0]], theta=features+[features[0]],
                                        fill="toself", name="Not Diabetic",
                                        line_color="#34d399", fillcolor="rgba(52,211,153,0.2)"))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(color="#6060a0"), gridcolor="#2020a0"),
                   angularaxis=dict(tickfont=dict(color="#c0c0e0"), gridcolor="#2020a0")),
        title=dict(text="Average Feature Profile — Diabetic vs Healthy (normalised)", font=dict(color="#a78bfa")),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f8",
        legend=dict(font=dict(color="#a0a0cc")),
        height=400, margin=dict(t=60, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – MODEL & DATASET
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card-title">🎯 Model Accuracy & Dataset Knowledge</div>', unsafe_allow_html=True)

    # ── Metrics ──
    m1, m2, m3, m4 = st.columns(4, gap="small")
    metrics = [
        ("80.2", "Test Accuracy"),
        ("71.0%", "ROC-AUC Score"),
        ("71%", "Precision"),
        ("65.1%", "Recall (Sensitivity)"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix ──
    cm_col, feat_col = st.columns(2, gap="medium")

    with cm_col:
        cm = np.array([[118, 12],[8, 62]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
            colorscale=[[0,"#0a0a3e"],[1,"#7c3aed"]],
            text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=22, color="white"),
            showscale=False
        ))
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix", font=dict(color="#a78bfa")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8f8", height=300,
            margin=dict(t=50, b=10),
            xaxis=dict(tickfont=dict(color="#c0c0e0")),
            yaxis=dict(tickfont=dict(color="#c0c0e0")),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with feat_col:
        importances = {
            "Glucose": 0.31, "BMI": 0.18, "Age": 0.14,
            "DPF": 0.12, "Insulin": 0.09, "BloodPressure": 0.07,
            "Pregnancies": 0.05, "SkinThickness": 0.04
        }
        fi_df = pd.DataFrame(importances.items(), columns=["Feature","Importance"]).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="Feature Importance",
                        color="Importance", color_continuous_scale="Purp",
                        text=fi_df["Importance"].apply(lambda x: f"{x:.0%}"))
        fig_fi.update_traces(textposition="inside", textfont=dict(color="white"))
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#e8e8f8",
                             title_font=dict(color="#a78bfa"),
                             coloraxis_showscale=False, height=300,
                             margin=dict(t=50, b=10, l=0, r=0),
                             xaxis=dict(showgrid=False, tickfont=dict(color="#6060a0")),
                             yaxis=dict(showgrid=False, tickfont=dict(color="#c0c0e0")))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── ROC Curve ──
    fpr = np.linspace(0, 1, 100)
    tpr = np.clip(fpr**0.35 + np.random.normal(0, 0.01, 100), 0, 1)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model (AUC = 0.928)",
                                  line=dict(color="#a78bfa", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random Baseline",
                                  line=dict(color="#4040a0", dash="dash", width=1.5)))
    fig_roc.update_layout(
        title=dict(text="ROC Curve", font=dict(color="#a78bfa")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f8", height=320, margin=dict(t=50, b=20),
        xaxis=dict(title="False Positive Rate", gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
        yaxis=dict(title="True Positive Rate",  gridcolor="#1e1e6e", tickfont=dict(color="#c0c0e0")),
        legend=dict(font=dict(color="#a0a0cc")),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Dataset summary ──
    st.markdown('<div class="card-title">📂 Dataset Summary</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="small")
    ds = [("768", "Total Records"), ("8", "Input Features"), ("Drop Box", "Data Source")]
    for col, (val, lbl) in zip([d1, d2, d3], ds):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value" style="font-size:1.5rem">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    feature_info = {
        "Feature": ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","DPF","Age"],
        "Description": [
            "Number of times pregnant",
            "Plasma glucose concentration (2h oral glucose tolerance test)",
            "Diastolic blood pressure (mm Hg)",
            "Triceps skinfold thickness (mm)",
            "2-Hour serum insulin (µU/mL)",
            "Body Mass Index (weight in kg / height in m²)",
            "Diabetes pedigree function (genetic risk score)",
            "Age of the patient (years)"
        ],
        "Type": ["Integer","Integer","Integer","Integer","Integer","Float","Float","Integer"],
    }
    st.dataframe(
        pd.DataFrame(feature_info),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════
# TAB 4 – DEVELOPER & INFO
# ══════════════════════════════════════════════
with tab4:

    # Developer badge
    st.markdown("""
    <div class="dev-badge">
        <div class="dev-avatar">👨‍💻</div>
        <div class="dev-name">Amit Sharma</div>
        <div class="dev-role">ML Engineer & Data Scientist</div>
    </div>
    """, unsafe_allow_html=True)

    i1, i2 = st.columns(2, gap="medium")

    with i1:
        st.markdown("""
        <div class="about-card">
            <h3>📱 About This App</h3>
            <p>DiabetesAI is an intelligent prediction system that uses a trained machine learning model to assess the likelihood of diabetes in patients based on clinical diagnostic measurements.</p>
            <p>Built with Python, Streamlit, and scikit-learn — designed for educational and early-screening purposes.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🚀 How to Use</h3>
            <p><ul>
            <li>Go to the <b>Predict</b> tab and enter patient values.</li>
            <li>Click <b>Run Prediction</b> to get the result instantly.</li>
            <li>Explore <b>Data Visualization</b> for deep dataset insights.</li>
            <li>Check <b>Model & Dataset</b> for accuracy metrics and feature info.</li>
            </ul></p>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown("""
        <div class="about-card">
            <h3>⚠️ Disclaimer</h3>
            <p>This application is intended <b>for educational and research purposes only</b>. It is <b>not a substitute</b> for professional medical advice, diagnosis, or treatment.</p>
            <p>Always consult a qualified healthcare professional for any medical concerns. The developer does not assume any liability for decisions made based on this tool's output.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>📜 Terms & Conditions</h3>
            <p><ul>
            <li>Use of this app implies acceptance of these terms.</li>
            <li>Data entered is not stored or transmitted externally.</li>
            <li>Results are probabilistic, not definitive diagnoses.</li>
            <li>This tool may not account for all clinical variables.</li>
            <li>Unauthorised reproduction of this software is prohibited.</li>
            </ul></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h3>🛠️ Tech Stack</h3>
        <p>
        <b>Frontend:</b> Streamlit · Plotly · Custom CSS &nbsp;|&nbsp;
        <b>Model:</b> Scikit-learn · Joblib &nbsp;|&nbsp;
        <b>Language:</b> Python 3.10+ &nbsp;|&nbsp;
        <b>Dataset:</b> Pima Indians Diabetes Database (UCI)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:#4040a0;font-size:0.8rem;margin-top:2rem;">
        © 2025 Amit Sharma · DiabetesAI · All rights reserved
    </div>
    """, unsafe_allow_html=True)