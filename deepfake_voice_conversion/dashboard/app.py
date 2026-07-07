import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.fft import dct
import os
import io
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #0D9488, #2563EB);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 1rem 0 0.2rem 0;
}
.sub-title {
    font-size: 1.1rem; color: #64748B;
    text-align: center; margin-bottom: 2rem;
}
.real-result {
    background: linear-gradient(135deg, #16A34A, #15803D);
    color: white; padding: 1.5rem 2rem; border-radius: 16px;
    font-size: 1.8rem; font-weight: 700; text-align: center;
    margin: 1rem 0; box-shadow: 0 8px 25px rgba(22,163,74,0.35);
    letter-spacing: 1px;
}
.fake-result {
    background: linear-gradient(135deg, #EF4444, #B91C1C);
    color: white; padding: 1.5rem 2rem; border-radius: 16px;
    font-size: 1.8rem; font-weight: 700; text-align: center;
    margin: 1rem 0; box-shadow: 0 8px 25px rgba(239,68,68,0.35);
    letter-spacing: 1px;
}
.stat-card {
    background: white; border-radius: 14px; padding: 1.2rem;
    text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    border-top: 4px solid; margin: 0.3rem;
}
.insight-box {
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4);
    border-left: 5px solid #2563EB; border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem; margin: 0.8rem 0;
}
.warning-box {
    background: linear-gradient(135deg, #FEF9C3, #FEF2F2);
    border-left: 5px solid #F59E0B; border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem; margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────
BASE       = r'D:\Projects\DeepFake\deepfake_voice_conversion'
MODEL_PATH = os.path.join(BASE, 'models', 'cnn_lstm_final.keras')
PLOTS_DIR  = os.path.join(BASE, 'results', 'plots')

# ── MODEL LOADING ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# ── AUDIO PROCESSING ──────────────────────────────────────
SR, DURATION = 16000, 4
N_SAMPLES    = SR * DURATION

def load_audio_from_bytes(audio_bytes):
    import tempfile
    import soundfile as sf

    # Write to temp file first — fixes BytesIO signature error
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SR,
                                  duration=DURATION, mono=True)
    except Exception:
        # Fallback for flac files
        data, samplerate = sf.read(tmp_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        import resampy
        audio = resampy.resample(data, samplerate, SR)
        audio = audio[:N_SAMPLES]
    finally:
        os.unlink(tmp_path)

    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    audio = audio[:N_SAMPLES]
    m = np.max(np.abs(audio))
    return audio / m if m > 0 else audio

def extract_features(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=40, n_fft=2048, hop_length=512)
    mfcc = (mfcc - np.mean(mfcc,axis=1,keepdims=True)) / \
           (np.std(mfcc,axis=1,keepdims=True)+1e-8)
    mel  = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio, sr=SR, n_mels=128, n_fft=2048, hop_length=512),
        ref=np.max)
    mel  = (mel-mel.mean())/(mel.std()+1e-8)
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    fb   = np.zeros((40, stft.shape[0]))
    for i in range(40):
        s=int(i*stft.shape[0]/40); e=int((i+1)*stft.shape[0]/40)
        fb[i,s:e]=1.0
    lfcc = dct(np.log(np.dot(fb,stft)+1e-8), axis=0, norm='ortho')[:40]
    lfcc = (lfcc-np.mean(lfcc,axis=1,keepdims=True)) / \
           (np.std(lfcc,axis=1,keepdims=True)+1e-8)
    min_T = min(mfcc.shape[1], mel.shape[1], lfcc.shape[1])
    feat  = np.vstack([mfcc[:,:min_T], mel[:,:min_T], lfcc[:,:min_T]])
    return feat[...,np.newaxis][np.newaxis,...]

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Navigation")
    page = st.radio("", [
        "🔍  Detect Audio",
        "📊  Model Performance",
        "🔬  How It Works",
        "📋  About"
    ])
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown("**CNN-LSTM Hybrid**")
    st.markdown("**1.93M parameters**")
    st.markdown("---")
    st.markdown("### 📈 Performance")
    cols = st.columns(2)
    cols[0].metric("AUC",      "96.30%")
    cols[1].metric("Accuracy", "86.62%")
    cols[0].metric("F1",       "87.51%")
    cols[1].metric("EER",      "8.62%")
    st.markdown("---")
    st.markdown("###  Key Finding")
    st.info("LFCC is 13× more important than MFCC for detecting fake audio")
   

# ══════════════════════════════════════════════════════════
# PAGE 1 — DETECT
# ══════════════════════════════════════════════════════════
if page == "🔍  Detect Audio":
    st.markdown('<div class="main-title">🎙️ Deepfake Audio Detector</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an audio file — the AI will tell you if it is REAL human speech or AI-GENERATED in seconds</div>',
                unsafe_allow_html=True)

    # Upload zone
    uploaded = st.file_uploader(
        "**Drop your audio file here**",
        type=['wav','flac'],
        help="Upload .wav or .flac audio files"
    )

    if uploaded is not None:
        audio_bytes = uploaded.read()
        audio       = load_audio_from_bytes(audio_bytes)

        col1, col2 = st.columns([1,1], gap="large")

        with col1:
            st.markdown("### 🎵 Your Audio")
            st.audio(audio_bytes)

            # Waveform with plotly
            times = np.linspace(0, DURATION, len(audio))
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=times, y=audio,
                fill='tozeroy',
                fillcolor='rgba(13,148,136,0.15)',
                line=dict(color='#0D9488', width=1.5),
                name='Waveform'
            ))
            fig_wave.update_layout(
                title='Audio Waveform',
                xaxis_title='Time (seconds)',
                yaxis_title='Amplitude',
                height=250,
                margin=dict(l=20,r=20,t=40,b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,250,252,1)',
                showlegend=False,
                xaxis=dict(gridcolor='#E2E8F0'),
                yaxis=dict(gridcolor='#E2E8F0')
            )
            st.plotly_chart(fig_wave, use_container_width=True)

            # Frequency spectrum
            fft_vals  = np.abs(np.fft.rfft(audio))
            fft_freqs = np.fft.rfftfreq(len(audio), 1/SR)
            fig_freq  = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=fft_freqs[:len(fft_freqs)//4],
                y=fft_vals[:len(fft_vals)//4],
                fill='tozeroy',
                fillcolor='rgba(37,99,235,0.12)',
                line=dict(color='#2563EB', width=1.5),
            ))
            fig_freq.update_layout(
                title='Frequency Spectrum',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Magnitude',
                height=220,
                margin=dict(l=20,r=20,t=40,b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,250,252,1)',
                showlegend=False,
                xaxis=dict(gridcolor='#E2E8F0'),
                yaxis=dict(gridcolor='#E2E8F0')
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        with col2:
            st.markdown("### 🔍 Detection Result")
            with st.spinner("Analyzing audio... extracting features..."):
                try:
                    model = load_model()
                    feat  = extract_features(audio)
                    score = float(model.predict(feat, verbose=0)[0][0])
                    conf  = score*100 if score>=0.5 else (1-score)*100
                    label = "FAKE / AI-GENERATED" if score>=0.5 else "REAL"

                    if label == "REAL":
                        st.markdown('<div class="real-result">✅ REAL AUDIO</div>',
                                    unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown('<div class="fake-result">🚨 FAKE / AI-GENERATED</div>',
                                    unsafe_allow_html=True)

                    st.markdown(f"**Confidence: {conf:.1f}%**")

                    # Confidence gauge with plotly
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf,
                        domain={'x':[0,1],'y':[0,1]},
                        title={'text': "Confidence Level"},
                        gauge={
                            'axis': {'range':[0,100]},
                            'bar':  {'color': '#16A34A' if label=='REAL' else '#EF4444'},
                            'steps': [
                                {'range':[0,50], 'color':'#F0FDF4'},
                                {'range':[50,80],'color':'#DCFCE7'},
                                {'range':[80,100],'color':'#BBF7D0'},
                            ],
                            'threshold': {
                                'line':{'color':'#0D9488','width':4},
                                'thickness':0.75,'value':80
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=280,
                        margin=dict(l=20,r=20,t=40,b=20),
                        paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Detail metrics
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Raw Score",  f"{score:.4f}")
                    c2.metric("Confidence", f"{conf:.1f}%")
                    c3.metric("Verdict",    "REAL" if score<0.5 else "FAKE")

                    if label != "REAL":
                        st.markdown("""
                        <div class="warning-box">
                        ⚠️ <strong>Synthetic artifacts detected.</strong><br>
                        This audio shows patterns consistent with AI-generated speech.
                        It may have been produced using tools like ElevenLabs, RVC,
                        or similar voice synthesis systems. Such audio is used in
                        fraud, impersonation, and disinformation attacks.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="insight-box">
                        ✅ <strong>Genuine human speech detected.</strong><br>
                        No synthetic artifacts found. The acoustic patterns in this
                        audio are consistent with real human voice recordings.
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info(f"Model path: {MODEL_PATH}")

        # Feature Visualizations
        st.markdown("---")
        st.markdown("### 📊 Acoustic Feature Analysis")
        st.caption("These are the three features your CNN-LSTM model analyzed to make its decision")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**🔴 MFCC — Vocal Tract Features**")
            fig_m, ax_m = plt.subplots(figsize=(6,3.5))
            mfcc_vis = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
            librosa.display.specshow(mfcc_vis, sr=SR, hop_length=512,
                                      x_axis='time', ax=ax_m, cmap='RdYlGn')
            ax_m.set_ylabel('Coefficient')
            ax_m.set_title('MFCC (40 coefficients)', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close()
            st.caption("Captures how the vocal tract shapes sound. Less important for fake detection (3% drop if removed)")

        with c2:
            st.markdown("**🟠 Mel Spectrogram — Time-Frequency Image**")
            fig_mel, ax_mel = plt.subplots(figsize=(6,3.5))
            mel_vis = librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128))
            librosa.display.specshow(mel_vis, sr=SR, hop_length=512,
                                      x_axis='time', y_axis='mel',
                                      ax=ax_mel, cmap='magma')
            ax_mel.set_title('Mel Spectrogram (128 bins)', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_mel)
            plt.close()
            st.caption("Converts audio to a 2D image. CNN reads this like a photo (7.5% drop if removed)")

        with c3:
            st.markdown("**⭐ LFCC — Most Important Feature**")
            fig_l, ax_l = plt.subplots(figsize=(6,3.5))
            stft_vis  = np.abs(librosa.stft(audio))
            lfcc_vis  = librosa.power_to_db(stft_vis[:40,:])
            librosa.display.specshow(lfcc_vis, sr=SR,
                                      hop_length=512, ax=ax_l, cmap='plasma')
            ax_l.set_title('LFCC (40 coefficients) ⭐', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_l)
            plt.close()
            st.caption("⭐ KEY FINDING: 40% accuracy drop if removed. Linear frequency artifacts reveal AI generation")

    else:
        # Landing state
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        steps = [
            ("1️⃣", "#0D9488", "Upload Audio", "Drop a .wav or .flac file above"),
            ("2️⃣", "#2563EB", "Feature Extraction", "Extracts MFCC + Mel + LFCC features automatically"),
            ("3️⃣", "#F97316", "CNN-LSTM Analysis", "Deep learning model analyzes 208 acoustic features"),
            ("4️⃣", "#16A34A", "Get Result", "Instant REAL or FAKE verdict with confidence score"),
        ]
        for col, (num, color, title, desc) in zip([col1,col2,col3,col4], steps):
            with col:
                st.markdown(f"""
                <div style='text-align:center; padding:1.5rem; background:white;
                border-radius:14px; box-shadow:0 4px 15px rgba(0,0,0,0.07);
                border-top:4px solid {color};'>
                <div style='font-size:2rem'>{num}</div>
                <div style='font-weight:700; color:{color}; margin:0.5rem 0'>{title}</div>
                <div style='font-size:0.9rem; color:#64748B'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

       

# ══════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown('<div class="main-title">📊 Model Performance</div>',
                unsafe_allow_html=True)

    # Animated metrics
    st.markdown("### ASVspoof 2021 LA — Test Results (800 samples)")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    data = [
        (c1,"Accuracy","86.62%","↑","#0D9488"),
        (c2,"Precision","82.06%","↑","#2563EB"),
        (c3,"Recall","93.75%","↑","#16A34A"),
        (c4,"F1 Score","87.51%","↑","#F97316"),
        (c5,"AUC","96.30%","↑","#7C3AED"),
        (c6,"EER","8.62%","↓ lower=better","#EF4444"),
    ]
    for col, name, val, delta, color in data:
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-top-color:{color}">
            <div style="font-size:1.6rem;font-weight:700;color:{color}">{val}</div>
            <div style="font-size:0.85rem;color:#64748B;margin-top:0.3rem">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Plotly radar chart for metrics
    categories = ['Accuracy','Precision','Recall','F1 Score','AUC']
    values     = [86.62, 82.06, 93.75, 87.51, 96.30]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(13,148,136,0.2)',
        line=dict(color='#0D9488', width=2),
        name='CNN-LSTM Model'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[70,100])),
        title='Performance Radar Chart',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Images side by side
    col1, col2 = st.columns(2)
    with col1:
        cm_path = os.path.join(PLOTS_DIR, 'cm_ASVspoof_2021_LA.png')
        if os.path.exists(cm_path):
            st.markdown("### Confusion Matrix")
            st.image(cm_path)
            st.markdown("""
            <div class="insight-box">
            ✅ <strong>318</strong> real samples correctly identified<br>
            ✅ <strong>375</strong> fake samples correctly identified<br>
            ⚠️ 82 false alarms (real called fake)<br>
            ⚠️ 25 missed fakes (fake called real)
            </div>
            """, unsafe_allow_html=True)

    with col2:
        tc_path = os.path.join(PLOTS_DIR, 'training_curves.png')
        if os.path.exists(tc_path):
            st.markdown("### Training History")
            st.image(tc_path)
            st.markdown("""
            <div class="insight-box">
            📈 Model trained for <strong>25 epochs</strong><br>
            🎯 Best validation AUC: <strong>97.40%</strong><br>
            ✅ No overfitting — train and val curves track together
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture diagram
    st.markdown("### 🏗️ Model Architecture")
    arch_data = {
        'Layer': ['Input', 'Conv2D(32)', 'Conv2D(64)', 'Conv2D(128)',
                  'Reshape', 'LSTM(128)', 'LSTM(64)', 'Dense(128)', 'Dense(1)'],
        'Output Shape': ['(208,126,1)', '(208,126,32)', '(104,63,64)', '(52,31,128)',
                         '(31,3328)', '(31,128)', '(64,)', '(128,)', '(1,)'],
        'Parameters': [0, 320, 18496, 73856, 0, 1769984, 49408, 8320, 65],
        'Purpose': ['Combined MFCC+Mel+LFCC features',
                    'Detect local spectral patterns',
                    'Detect complex frequency patterns',
                    'Detect high-level audio features',
                    'Prepare for temporal analysis',
                    'Learn patterns over time (sequence)',
                    'Compress temporal representation',
                    'Final classification layer',
                    'Real(0) or Fake(1) output']
    }
    import pandas as pd
    df_arch = pd.DataFrame(arch_data)
    st.dataframe(df_arch, use_container_width=True, hide_index=True)
    st.caption("Total parameters: 1,930,113 (7.36 MB)")

# ══════════════════════════════════════════════════════════
# PAGE 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════
elif page == "🔬  How It Works":
    st.markdown('<div class="main-title">🔬 How It Works</div>',
                unsafe_allow_html=True)

    # Key finding callout
    st.success("""
    🔑 **KEY RESEARCH FINDING — LFCC is the most important feature**

    Ablation study shows:
    - Remove MFCC → accuracy drops 3% (least important)
    - Remove Mel Spectrogram → accuracy drops 7.5% (moderately important)
    - Remove LFCC → accuracy drops **40%** (MOST important — 13× more than MFCC)

    **Interpretation:** AI-generated speech leaves its most detectable artifacts
    in the linear frequency domain — not the mel-scaled domain that most papers use.
    This is an original finding not present in the base paper (Asuai et al. 2025).
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance",
        "🔍 Real vs Fake",
        "🎯 GradCAM",
        "🌐 Cross-Dataset"
    ])

    with tab1:
        st.markdown("### Ablation Study — Which Feature Matters Most?")
        st.markdown("""
        An **ablation study** removes one feature at a time and measures how much
        accuracy drops. Higher drop = that feature is more important.
        """)

        # Plotly bar chart for feature importance
        features = ['MFCC (40 coeff)', 'Mel Spectrogram (128 bins)', 'LFCC (40 coeff)']
        drops    = [3.0, 7.5, 40.0]
        colors   = ['#EF4444', '#F97316', '#0D9488']

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            x=features, y=drops,
            marker_color=colors,
            text=[f'{d:.1f}%' for d in drops],
            textposition='outside',
            textfont=dict(size=16, color='black')
        ))
        fig_fi.update_layout(
            title='Accuracy Drop When Feature is Removed<br><sub>Higher = More Important</sub>',
            yaxis_title='Accuracy Drop (%)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,1)',
            yaxis=dict(gridcolor='#E2E8F0', range=[0,50])
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Show the saved plot too
        fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-card" style="border-top-color:#EF4444">
            <div style="font-size:2rem;font-weight:700;color:#EF4444">3%</div>
            <div style="font-weight:600">MFCC</div>
            <div style="font-size:0.85rem;color:#64748B">Least important.<br>Vocal tract shape is similar in real and fake audio</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card" style="border-top-color:#F97316">
            <div style="font-size:2rem;font-weight:700;color:#F97316">7.5%</div>
            <div style="font-weight:600">Mel Spectrogram</div>
            <div style="font-size:0.85rem;color:#64748B">Moderately important.<br>Time-frequency patterns differ somewhat</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-card" style="border-top-color:#0D9488">
            <div style="font-size:2rem;font-weight:700;color:#0D9488">40% ⭐</div>
            <div style="font-weight:600">LFCC</div>
            <div style="font-size:0.85rem;color:#64748B">MOST IMPORTANT.<br>AI artifacts are strongest in linear frequency domain</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Visual Comparison — Real vs Fake Audio Features")
        st.markdown("""
        These images show what the three features look like for a real audio clip (top row)
        vs a fake AI-generated clip (bottom row). Notice the differences — especially in the
        LFCC column (rightmost) where the texture patterns are visibly different.
        """)
        rv_path = os.path.join(PLOTS_DIR, 'real_vs_fake_features.png')
        if os.path.exists(rv_path):
            st.image(rv_path)
            st.caption("Top row = REAL audio | Bottom row = FAKE audio. Each column shows MFCC, Mel Spectrogram, and LFCC respectively")
        else:
            st.warning("Plot not found — run the notebook first")

    with tab3:
        st.markdown("### GradCAM — What the CNN Focuses On")
        st.markdown("""
        **GradCAM (Gradient-weighted Class Activation Mapping)** shows which parts of the
        acoustic features the CNN relied on most when making its decision.

        Think of it like a heat map — the CNN highlights which time-frequency regions were
        most important for its REAL vs FAKE decision.
        """)
        gc_path = os.path.join(PLOTS_DIR, 'gradcam_explanation.png')
        if os.path.exists(gc_path):
            st.image(gc_path)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="insight-box">
            🔴 <strong>Bright red/yellow areas</strong><br>
            The CNN focused heavily here. These are the most discriminative
            time-frequency regions for the real/fake decision.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="warning-box">
            🔵 <strong>Dark blue areas</strong><br>
            The CNN mostly ignored these regions. They contribute little
            to distinguishing real from fake audio.
            </div>
            """, unsafe_allow_html=True)

        st.info("REAL and FAKE audio show **different GradCAM patterns** — proving the model learned genuine acoustic differences, not dataset artifacts.")

    with tab4:
        st.markdown("### Cross-Dataset Generalization")
        st.markdown("""
        One of the key challenges in deepfake detection is **generalization** —
        does the model work on audio from sources it was never trained on?

        We tested the model on voice-modified audio (pitch-shifted and formant-changed
        audio from the real dataset) to simulate an unseen distribution.
        """)

        cd_path = os.path.join(PLOTS_DIR, 'cross_dataset_comparison.png')
        if os.path.exists(cd_path):
            st.image(cd_path)

        # Plotly comparison
        datasets = ['ASVspoof 2021 LA<br>(Training)', 'Voice-Modified<br>(Unseen)']
        acc_vals = [86.62, 53.33]
        f1_vals  = [87.51, 21.33]
        auc_vals = [96.30, 55.0]

        fig_cross = go.Figure()
        for name, vals, color in [
            ('Accuracy', acc_vals, '#0D9488'),
            ('F1', f1_vals, '#F97316'),
            ('AUC', auc_vals, '#7C3AED')
        ]:
            fig_cross.add_trace(go.Bar(
                name=name, x=datasets, y=vals,
                marker_color=color, opacity=0.85
            ))

        fig_cross.update_layout(
            barmode='group',
            title='Performance Drop from Training to Unseen Distribution',
            yaxis_title='Score (%)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,1)',
            yaxis=dict(gridcolor='#E2E8F0', range=[0,115])
        )
        st.plotly_chart(fig_cross, use_container_width=True)

        st.warning("""
        **This performance drop IS the research finding.**
        It confirms what 2025-2026 literature says — models trained on one distribution
        struggle on unseen tools. This is the generalization challenge your project
        identifies and measures.
        """)

# ══════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════
elif page == "📋  About":
    st.markdown('<div class="main-title">📋 About This Project</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1.3,1])

    with col1:
        st.markdown("""
        ### 🎯 Project Title
        **Explainable Deepfake Audio Detection with Cross-Dataset Generalization Analysis**

        ### 🚨 Why This Matters
        In February 2024, fraudsters cloned executive voices and tricked a finance
        worker into wiring **$25 million**. Deepfake fraud attempts surged **1,300%**
        in 2024. Losses in contact centers reached **$12.5 billion** globally.

        Humans cannot reliably detect fake audio. Automated detection is critical.

        ### 🔬 What We Built
        A CNN-LSTM hybrid deep learning model that:
        1. Extracts **MFCC + Mel Spectrogram + LFCC** features from audio
        2. Classifies audio as **REAL or FAKE** with 86.62% accuracy
        3. Achieves **96.30% AUC** on ASVspoof 2021 LA
        4. Provides **GradCAM explainability** — shows WHY it made the decision
        5. Demonstrates **cross-dataset generalization analysis**

        ### 📊 Key Results
        | Metric | Score |
        |---|---|
        | Accuracy | 86.62% |
        | F1 Score | 87.51% |
        | Recall | 93.75% |
        | AUC | 96.30% |
        | EER | 8.62% |

        ### 🔑 Original Research Finding
        **LFCC is 13× more important than MFCC** for detecting fake audio.
        Removing it causes a 40% accuracy drop — revealing that AI-generated
        speech artifacts primarily manifest in the linear frequency domain.
        This was not found in the base paper (Asuai et al. 2025).
        """)

    with col2:
        st.markdown("""
        ### 📚 Base Paper
        **Asuai et al. (2025)**
        *Hybrid CNN-LSTM Architectures for Deepfake Audio Detection
        Using MFCC and Spectrogram Analysis*
        American Journal of Mathematical and Computer Modelling

        ### 🔍 Research Gap We Address
        The base paper achieved **94.7% on TTS deepfakes** but:
        - ❌ No LFCC features — missing the most important feature
        - ❌ No explainability — black box decisions
        - ❌ No cross-dataset evaluation

        **This project adds all three.**

        ### 📖 References
        1. Asuai et al. (2025) — Base paper (CNN-LSTM)
        2. Frank & Schönherr (2021) — WaveFake dataset
        3. Yi et al. (2024) — Survey of 200+ papers
        4. Borzi et al. (2025) — MDPI Sensors survey
        5. Chinchmalatpure et al. (2026) — RVC detection

        ### 👩‍💻 Developed By
        **Narjeena Thanveen P K**
        MSc Computer Science (AI/ML/Data Science) — 4th Semester
        S-VYASA Deemed to be University, Bengaluru
        MCAP481 — Capstone Project

        ### 🏗️ Tech Stack
        Python · TensorFlow · Keras · Librosa · Streamlit
        NumPy · Scikit-learn · Matplotlib · Seaborn · Plotly
        """)

        st.markdown("""
        <div class="insight-box">
        <strong>Dataset:</strong> ASVspoof 2021 LA<br>
        <strong>Training:</strong> 2000 real + 2000 fake audio samples<br>
        <strong>Model size:</strong> 1,930,113 parameters (7.36 MB)<br>
        <strong>Training time:</strong> 25 epochs (~45 mins on CPU)
        </div>
        """, unsafe_allow_html=True)