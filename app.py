"""
app.py
Production-Grade Cyber-Medical Streamlit Web Application
Automatic Diagnosis of Six Twelve-Lead ECG Abnormalities Using a Deep Residual Neural Network
Based on the Nature Communications / IEEE Audit Report by Prashant Kumar Moharana, Krish Kumar Kedia, Vansh Kukkar (VIT Chennai)
Framework:
- Model: 1D Deep Residual Neural Network (ResNet-1D, 4 stages, 16-320 channels, 4096x12 input tensor at 400 Hz)
- Multi-label sigmoid head with class-specific optimal F1 thresholds (tau_k)
- 4 Publishable Novel Research Contributions:
  1. 1D Temporal Grad-CAM & Multi-Lead Saliency Attention Map (Explainable AI / Saliency)
  2. Multi-Label Conformal Prediction Sets (Certified (1 - alpha) marginal coverage)
  3. Latent Representation Space & Electrophysiological Feature Projection
  4. Clinical Hemodynamic Triage & Automated Publication-Grade PDF Generator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dnn_engine.dnn_pipeline import DNNECGPipeline
from dnn_engine.resnet_dnn import TARGET_CLASSES, NOMINAL_THRESHOLDS, CLASS_DESCRIPTIONS
from dnn_engine.generate_dnn_samples import LEAD_ORDER

# Page Configuration
st.set_page_config(
    page_title="CardioResNet-1D | 12-Lead Multi-Label Deep Diagnosis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Medical Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main {
        background: radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.06) 0%, transparent 40%),
                    radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.06) 0%, transparent 45%),
                    #090d16;
        color: #e2e8f0;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }

    .badge-dnn {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-pos {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-neg {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
    }

    .triage-banner {
        border-radius: 10px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }

    @keyframes ecgPulse {
        0% { filter: drop-shadow(0 0 2px #38bdf8); }
        50% { filter: drop-shadow(0 0 10px #38bdf8); }
        100% { filter: drop-shadow(0 0 2px #38bdf8); }
    }
    .heart-pulse {
        animation: ecgPulse 2.2s infinite ease-in-out;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(51, 65, 85, 0.6);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize DNN Pipeline Cache
@st.cache_resource
def get_dnn_pipeline():
    return DNNECGPipeline()

pipeline = get_dnn_pipeline()

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dnn_engine", "samples")

# SIDEBAR: Control Panel & Data Ingestion
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 32px;" class="heart-pulse">🫀</span>
            <div>
                <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">CARDIORESNET-1D</h2>
                <span style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;">Deep Neural Network Diagnostic Suite</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🗂️ Data Ingestion Mode")
    source_mode = st.radio("Select Input Source:", ["Pre-Loaded Clinical Cases (400 Hz)", "Upload Patient ECG File (.npy, .csv, .mat)"])
    
    loaded_signal = None
    record_label = ""
    
    if source_mode == "Pre-Loaded Clinical Cases (400 Hz)":
        sample_options = {
            "Case 1: Multi-Label Co-occurrence (LBBB + AF)": "case_MULTI_LBBB_AF.npy",
            "Case 2: 1st Degree AV Block (1dAVb — Prolonged PR)": "case_1dAVb.npy",
            "Case 3: Right Bundle Branch Block (RBBB — rsR' in V1)": "case_RBBB.npy",
            "Case 4: Left Bundle Branch Block (LBBB — Broad Notched R)": "case_LBBB.npy",
            "Case 5: Sinus Bradycardia (SB — Rate < 60 BPM)": "case_SB.npy",
            "Case 6: Atrial Fibrillation (AF — Chaotic Baseline)": "case_AF.npy",
            "Case 7: Sinus Tachycardia (ST — Rate > 100 BPM)": "case_ST.npy",
            "Case 8: Normal Sinus Tracing (No Abnormality)": "case_NORMAL.npy"
        }
        chosen_sample = st.selectbox("Choose Verified Benchmark Strip:", list(sample_options.keys()))
        file_path = os.path.join(SAMPLES_DIR, sample_options[chosen_sample])
        
        if os.path.exists(file_path):
            loaded_signal = np.load(file_path)
            record_label = sample_options[chosen_sample]
            st.success(f"Loaded `{record_label}` (12 leads, 4,096 samples @ 400 Hz)")
            
    else:
        uploaded_file = st.file_uploader("Drop Patient ECG file:", type=["npy", "csv", "mat"])
        if uploaded_file is not None:
            record_label = uploaded_file.name
            fname = uploaded_file.name.lower()
            if fname.endswith(".npy"):
                loaded_signal = np.load(uploaded_file)
            elif fname.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
                loaded_signal = df_up.values.T if df_up.shape[1] == 12 else df_up.values
            elif fname.endswith(".mat"):
                mat_dict = sio.loadmat(uploaded_file)
                for k in ['val', 'data', 'ecg', 'tracings']:
                    if k in mat_dict and isinstance(mat_dict[k], np.ndarray):
                        loaded_signal = mat_dict[k]
                        break
                        
            if loaded_signal is not None:
                st.success(f"Uploaded `{record_label}` successfully!")
                
    st.markdown("---")
    st.markdown("### ⚙️ Operating Parameters")
    st.markdown("<b>Input Tensor Shape:</b> `(4096, 12)`", unsafe_allow_html=True)
    st.markdown("<b>Sampling Rate:</b> `400 Hz (~10.24 s)`", unsafe_allow_html=True)
    st.markdown("<b>Voltage Scaling:</b> `10^-4 V`", unsafe_allow_html=True)
    use_calibrated_thresh = st.checkbox("Apply Class-Specific F1 Thresholds (tau_k)", value=True)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
            <b>Report Authors:</b><br>
            • Prashant Kumar Moharana<br>
            • Krish Kumar Kedia<br>
            • Vansh Kukkar<br>
            <i>Vellore Institute of Technology, Chennai</i><br><br>
            <b>Companion Publication:</b><br>
            Ribeiro et al., <i>Nature Communications</i> 11, 1760 (2020)
        </div>
    """, unsafe_allow_html=True)

# MAIN INTERFACE
if loaded_signal is None:
    st.info("👈 Please select a pre-loaded clinical case or upload an ECG file from the sidebar.")
    st.stop()

# Reorder/orient signal to (12, 4096)
ecg_array = np.array(loaded_signal, dtype=np.float32)
if ecg_array.shape == (4096, 12):
    ecg_array = ecg_array.T
elif ecg_array.ndim == 1:
    ecg_array = np.tile(ecg_array, (12, 1))

# Run Inference through 1D Deep Residual Neural Network
with st.spinner("⚡ Propagating 4,096×12 tensor through 4 Residual Stages (16-320 channels)..."):
    results = pipeline.predict(ecg_array, record_id=record_label)

verdict = results["overall_verdict"]
detected_list = results["detected_abnormalities"]
res_by_cls = results["results_by_class"]
raw_scores = results["raw_scores"]
thresholds = results["thresholds"]
novelty = results["novelty"]

# TOP HERO SECTION
triage_color = novelty.get("triage_color", "#10b981")
st.markdown(f"""
    <div class="triage-banner" style="background: {triage_color}18; border: 1px solid {triage_color}60;">
        <span style="font-size: 20px;">🚨</span>
        <div style="flex-grow: 1;">
            <b style="color: {triage_color}; font-size: 0.95rem;">CLINICAL TRIAGE STATUS: {novelty.get('triage_level')}</b>
            <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 12px;">• {novelty.get('risk_flags', ['Stable'])[0]}</span>
        </div>
        <span style="font-size: 0.8rem; background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 6px; color: #cbd5e1;">
            Multi-Label Sigmoid Independent Head
        </span>
    </div>
""", unsafe_allow_html=True)

col_hero1, col_hero2 = st.columns([2.3, 1])

with col_hero1:
    st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">12-Lead Multi-Label DNN Diagnosis</span>
                <span class="badge-dnn {'badge-pos' if detected_list else 'badge-neg'}">
                    {f"{len(detected_list)} Findings" if detected_list else "Normal"}
                </span>
            </div>
            <h1 style="margin: 8px 0 4px 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
                {verdict}
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.88rem;">
                Conformal Guaranteed Set: <b style="color: #38bdf8;">{', '.join(novelty.get('conformal_set', [])) or 'None'}</b> | 
                Evaluated against class-specific F1 thresholds (Listing 2 & 4)
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_hero2:
    st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Companion Benchmark</span>
            <h1 style="margin: 8px 0 4px 0; font-size: 2.3rem; font-weight: 800; color: #38bdf8;">
                0.951 <span style="font-size: 1.0rem; color: #64748b;">AP_micro</span>
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.82rem;">
                Specificity ≥ <b>0.995</b> across all 6 targets on 827-exam test set
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# 6 MULTI-LABEL TARGET CARDS
cols = st.columns(6)
for i, cls_name in enumerate(TARGET_CLASSES):
    item = res_by_cls[cls_name]
    prob = item['probability']
    th = item['threshold']
    is_det = item['detected']
    
    with cols[i]:
        st.markdown(f"""
            <div class="metric-card" style="padding: 14px; text-align: center; border-color: {'#ef4444' if is_det else 'rgba(51, 65, 85, 0.8)'};">
                <span style="font-size: 0.85rem; font-weight: 800; color: {'#f87171' if is_det else '#94a3b8'};">{cls_name}</span>
                <h3 style="margin: 4px 0; font-size: 1.4rem; color: #f8fafc;">{prob:.3f}</h3>
                <div style="font-size: 0.72rem; color: #64748b;">Thresh: {th:.3f}</div>
                <span class="badge-dnn {'badge-pos' if is_det else 'badge-neg'}" style="margin-top: 6px; font-size: 0.7rem; padding: 2px 8px;">
                    {'POSITIVE' if is_det else 'NEGATIVE'}
                </span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# INTERACTIVE TABS
tab_leads, tab_saliency, tab_conformal, tab_benchmark, tab_report = st.tabs([
    "📈 12-Lead Interactive Waveforms",
    "🔥 1D Temporal Grad-CAM & Lead Saliency",
    "🎯 Conformal Prediction & Calibration",
    "📊 Benchmark Comparison (Table I)",
    "📄 Clinical PDF Diagnostic Report"
])

# TAB 1: 12-LEAD WAVEFORMS
with tab_leads:
    st.markdown("#### Standard Twelve-Lead Surface Electrocardiogram (400 Hz)")
    layout_mode = st.radio("Display Layout:", ["Lead II Rhythm Strip (Zoomable)", "12-Lead 3x4 Diagnostic Grid"], horizontal=True)
    
    t_axis = np.arange(ecg_array.shape[1]) / 400.0
    
    if layout_mode == "Lead II Rhythm Strip (Zoomable)":
        lead2_sig = ecg_array[1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_axis, y=lead2_sig, mode='lines', name='Lead DII', line=dict(color='#38bdf8', width=1.8)
        ))
        fig.update_layout(
            title=f"Lead DII Standard Rhythm Strip ({record_label}) — Continuous Sampling @ 400 Hz",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (10^-4 V)",
            template="plotly_dark",
            paper_bgcolor='rgba(15, 23, 42, 0.7)',
            plot_bgcolor='rgba(10, 15, 29, 0.9)',
            height=420,
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.08)')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig_12 = make_subplots(
            rows=6, cols=2,
            subplot_titles=[f"Lead {name}" for name in LEAD_ORDER],
            vertical_spacing=0.06, horizontal_spacing=0.05
        )
        for i in range(12):
            r = (i % 6) + 1
            c = (i // 6) + 1
            fig_12.add_trace(
                go.Scatter(x=t_axis, y=ecg_array[i], mode='lines', line=dict(color='#10b981', width=1.2), name=LEAD_ORDER[i]),
                row=r, col=c
            )
        fig_12.update_layout(
            height=880, showlegend=False, template="plotly_dark",
            paper_bgcolor='rgba(15, 23, 42, 0.7)', plot_bgcolor='rgba(10, 15, 29, 0.9)',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_12, use_container_width=True)

# TAB 2: 1D TEMPORAL GRAD-CAM & LEAD SALIENCY (NOVELTY 1)
with tab_saliency:
    st.markdown("""
        ### 🔥 1D Temporal Grad-CAM & Deep Lead Saliency Attention
        **Novelty Contribution:** End-to-end deep neural networks are often treated as 'black boxes'. 
        By backpropagating the gradient of target abnormality **{}** through all 4 residual stages to the input tensor $\mathbf{{X}}$, we compute the exact temporal cardiac cycles and lead channels responsible for the diagnosis.
    """.format(novelty.get('salient_class', 'Target')))
    
    col_sal1, col_sal2 = st.columns([2, 1])
    
    with col_sal1:
        st.markdown("#### Temporal Saliency Heatmap (Overlaid with Lead DII)")
        sal_t = novelty.get("time_axis", [])
        sal_val = novelty.get("temporal_saliency", [])
        
        fig_sal = go.Figure()
        # Scaled Lead II
        lead2_sub = ecg_array[1][::8][:len(sal_t)]
        fig_sal.add_trace(go.Scatter(
            x=sal_t, y=lead2_sub, mode='lines', name='ECG Signal (Lead II)', line=dict(color='#cbd5e1', width=1.5)
        ))
        # Saliency highlight area
        fig_sal.add_trace(go.Scatter(
            x=sal_t, y=np.array(sal_val) * (np.max(lead2_sub) - np.min(lead2_sub)) + np.min(lead2_sub),
            mode='lines', fill='tozeroy', name='DNN Attention / Grad-CAM Energy',
            line=dict(color='#f43f5e', width=2), fillcolor='rgba(244, 63, 94, 0.25)'
        ))
        fig_sal.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(15, 23, 42, 0.7)', plot_bgcolor='rgba(10, 15, 29, 0.9)',
            height=340, xaxis_title="Time (seconds)", yaxis_title="Amplitude & Saliency Energy"
        )
        st.plotly_chart(fig_sal, use_container_width=True)
        
    with col_sal2:
        st.markdown("#### Lead Saliency Energy Breakdown")
        lead_sal = novelty.get("lead_saliency", [0]*12)
        df_lead_sal = pd.DataFrame({
            "Lead": LEAD_ORDER,
            "Saliency": lead_sal
        }).sort_values(by="Saliency", ascending=True)
        
        fig_lead_bar = go.Figure()
        fig_lead_bar.add_trace(go.Bar(
            y=df_lead_sal["Lead"], x=df_lead_sal["Saliency"], orientation='h',
            marker=dict(color=df_lead_sal["Saliency"], colorscale='Magma')
        ))
        fig_lead_bar.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(15, 23, 42, 0.7)', plot_bgcolor='rgba(10, 15, 29, 0.9)',
            height=340, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Normalized Energy"
        )
        st.plotly_chart(fig_lead_bar, use_container_width=True)

# TAB 3: CONFORMAL PREDICTION (NOVELTY 2)
with tab_conformal:
    st.markdown("""
        ### 🎯 Multi-Label Conformal Prediction Sets & Risk Calibration
        **Novelty Contribution:** Instead of arbitrary binary cutoffs, we construct **Split-Conformal Prediction Sets** guaranteeing that all true patient abnormalities are enclosed within the predicted set with certified **90% finite-sample confidence**.
    """)
    
    col_c1, col_c2 = st.columns([1.3, 1])
    
    with col_c1:
        st.markdown("#### Probability vs Class-Specific F1 Threshold Margin")
        df_thresh = pd.DataFrame([
            {
                "Abnormality": k,
                "Sigmoid Score (p_k)": res_by_cls[k]['probability'],
                "F1 Threshold (tau_k)": res_by_cls[k]['threshold'],
                "Margin (p_k - tau_k)": res_by_cls[k]['margin'],
                "Status": "POSITIVE" if res_by_cls[k]['detected'] else "NEGATIVE"
            }
            for k in TARGET_CLASSES
        ])
        st.dataframe(df_thresh, height=260)
        
    with col_c2:
        st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Conformal Guarantee</span>
                <h3 style="margin: 6px 0; color: #f8fafc;">90% Coverage Level</h3>
                <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
                    • <b>Certified Multi-Label Set:</b><br>
                    <code>{novelty.get('conformal_set', [])}</code><br><br>
                    • <b>Clinical Significance:</b> Conformal calibration adjusts for label shift and low class prevalence, preventing high false-negative rates in rare conduction blocks.
                </p>
            </div>
        """, unsafe_allow_html=True)

# TAB 4: BENCHMARK AUDIT (TABLE I FROM PAPER)
with tab_benchmark:
    st.markdown("### 📊 Benchmark Comparison on 827-Exam Test Set (Table I from Report)")
    st.markdown("Companion paper (*Ribeiro et al., Nature Communications 2020*) performance metrics audited in Section VIII:")
    
    bench_df = pd.DataFrame({
        "Abnormality": ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
        "Precision": [0.867, 0.895, 1.000, 0.833, 1.000, 0.947],
        "Recall": [0.929, 1.000, 1.000, 0.938, 0.769, 0.973],
        "Specificity": [0.995, 0.995, 1.000, 0.996, 1.000, 0.997],
        "F1 Score": [0.897, 0.944, 1.000, 0.882, 0.870, 0.960]
    })
    
    col_b1, col_b2 = st.columns([1.2, 1])
    with col_b1:
        st.dataframe(bench_df, height=260)
    with col_b2:
        fig_f1 = go.Figure(go.Bar(
            x=bench_df["Abnormality"], y=bench_df["F1 Score"],
            marker=dict(color=bench_df["F1 Score"], colorscale='Blues', showscale=False),
            text=[f"{v:.3f}" for v in bench_df["F1 Score"]], textposition='auto'
        ))
        fig_f1.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(15, 23, 42, 0.7)', plot_bgcolor='rgba(10, 15, 29, 0.9)',
            height=260, yaxis=dict(range=[0.7, 1.02]), margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_f1, use_container_width=True)

# TAB 5: CLINICAL PDF REPORT (NOVELTY 4)
with tab_report:
    st.markdown("#### 📄 Executive Clinical Diagnostic Audit Report (PDF)")
    col_rep1, col_rep2 = st.columns([1.2, 1])
    
    with col_rep1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top: 0; color: #38bdf8;">Report Contents</h4>
                <ul style="color: #94a3b8; font-size: 0.88rem; line-height: 1.6;">
                    <li><b>Exam ID:</b> <code>{record_label}</code></li>
                    <li><b>Architecture:</b> 1D Deep ResNet (4 Residual Stages)</li>
                    <li><b>Multi-Label Findings:</b> <code>{verdict}</code></li>
                    <li><b>Conformal 90% Set:</b> <code>{novelty.get('conformal_set', [])}</code></li>
                    <li><b>Triage Priority:</b> <code>{novelty.get('triage_level')}</code></li>
                    <li><b>Regulatory Disclaimer:</b> Pre-screening & clinical decision-support</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with col_rep2:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        pdf_bytes = pipeline.generate_pdf_report(record_label, results)
        st.download_button(
            label="⬇️ Download Official Clinical Audit PDF",
            data=pdf_bytes,
            file_name=f"CardioResNet_Report_{record_label.replace('.npy', '')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.success("✅ Diagnostic PDF generated and ready for instant export.")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.78rem;">
        CardioResNet-1D Clinical Decision-Support Suite • 1D Deep Residual Neural Network (Nature Communications Architecture) • Built for Academic & Pre-Screening Research
    </div>
""", unsafe_allow_html=True)
