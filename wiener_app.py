import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz

# --- Page Config ---
st.set_page_config(
    page_title="Signal Operations Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 🎨 THEME CONFIGURATION
# ==============================================================================
# Define color palettes for different "Scenes"
themes = {
    "Navy Command ⚓": {
        "bg": "#0e1b2e",           # Deep Navy Background
        "text": "#e0e0e0",         # Light Grey Text
        "accent": "#00d4ff",       # Cyan Radar color
        "plot_bg": "#13233a",      # Slightly lighter navy for plots
        "plot_element": "#00d4ff", # Cyan lines
        "grid": "#1f3a5f"          # Navy grid lines
    },
    "Scientific Light 🔬": {
        "bg": "#ffffff",           # White Background
        "text": "#333333",         # Dark Grey Text
        "accent": "#0066cc",       # Professional Blue
        "plot_bg": "#f0f2f6",      # Light Grey plot background
        "plot_element": "#0066cc", # Blue lines
        "grid": "#d9d9d9"          # Light Grey grid
    },
    "Industrial Amber ⚠️": {
        "bg": "#2b2b2b",           # Dark Grey Background
        "text": "#f0f0f0",         # White Text
        "accent": "#ffae00",       # Safety Orange
        "plot_bg": "#363636",      # Lighter Grey plots
        "plot_element": "#ffae00", # Orange lines
        "grid": "#4d4d4d"          # Grey grid
    },
    "Black Ops 🌑": { # The original one
        "bg": "#0e1117",
        "text": "#fafafa",
        "accent": "#00ff00",
        "plot_bg": "#0e1117",
        "plot_element": "#00ff00",
        "grid": "#333333"
    }
}

# --- Sidebar "Mission Control" ---
with st.sidebar:
    st.title("📡 OPS CENTER")
    
    # Theme Selector
    st.subheader("🖥️ Interface Theme")
    selected_theme_name = st.selectbox("Select Scene:", list(themes.keys()), index=0)
    current_theme = themes[selected_theme_name]
    
    st.divider()
    st.markdown("### 🚀 Mission Select")
    mission = st.radio(
        "",
        ["OP: CLARITY (Wiener)", "OP: HUNTER (Matched)", "OP: ORACLE (Prediction)"],
        index=0
    )

# --- Inject CSS Dynamic Variables ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {current_theme['bg']}; 
        color: {current_theme['text']};
    }}
    .metric-container {{
        border: 1px solid {current_theme['grid']};
        padding: 10px;
        border-radius: 5px;
        background-color: {current_theme['plot_bg']};
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Courier New', Courier, monospace;
        color: {current_theme['accent']};
    }}
    h1, h2, h3, p {{
        font-family: 'Helvetica', sans-serif;
        color: {current_theme['text']} !important;
    }}
    .stButton>button {{
        border-radius: 20px;
        border: 1px solid {current_theme['grid']};
        background-color: {current_theme['plot_bg']};
        color: {current_theme['text']};
    }}
    .stButton>button:hover {{
        border-color: {current_theme['accent']};
        color: {current_theme['accent']};
    }}
</style>
""", unsafe_allow_html=True)

# --- Plotting Helper to Apply Theme ---
def apply_theme_to_plot(fig, ax, theme):
    fig.patch.set_facecolor(theme['bg'])
    
    # Handle single ax or array of axes
    if isinstance(ax, np.ndarray):
        axes = ax.flat
    else:
        axes = [ax]
        
    for a in axes:
        a.set_facecolor(theme['plot_bg'])
        a.tick_params(colors=theme['text'])
        a.spines['bottom'].set_color(theme['text'])
        a.spines['left'].set_color(theme['text'])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.grid(True, color=theme['grid'], linestyle=':')
        
        # Update text labels
        a.xaxis.label.set_color(theme['text'])
        a.yaxis.label.set_color(theme['text'])
        a.title.set_color(theme['text'])
        
        # Update legend if it exists
        legend = a.get_legend()
        if legend:
            plt.setp(legend.get_texts(), color=theme['text'])
            legend.get_frame().set_facecolor(theme['plot_bg'])
            legend.get_frame().set_edgecolor(theme['grid'])

# --- Helper Functions ---
def design_wiener(noisy, clean, L):
    r_yx = np.correlate(clean, noisy, mode='full')
    center = len(r_yx) // 2
    r_yx = r_yx[center:center+L]
    
    r_yy = np.correlate(noisy, noisy, mode='full')
    center_y = len(r_yy) // 2
    r_yy = r_yy[center_y:center_y+L]
    R_yy = toeplitz(r_yy)
    
    try:
        h = np.linalg.solve(R_yy + np.eye(L)*1e-6, r_yx)
        return h
    except:
        return np.zeros(L)

def generate_ar_signal(n_samples):
    np.random.seed(42)
    w = np.random.normal(0, 1, n_samples)
    x = np.zeros(n_samples)
    for n in range(2, n_samples):
        x[n] = 0.75*x[n-1] - 0.5*x[n-2] + w[n]
    return x / np.max(np.abs(x))

# ==============================================================================
# MISSION 1: OPERATION CLARITY (Wiener Filter)
# ==============================================================================
if mission == "OP: CLARITY (Wiener)":
    st.header("Signal Restoration")
    
    col_ctrl, col_vis = st.columns([1, 3])
    
    with col_ctrl:
        # st.subheader("Filter Params")
        L_taps = st.number_input("Filter Taps (L)", 2, 100, 15)
        snr_setting = st.select_slider("Jamming Level (Noise)", options=["Low", "Medium", "High", "Critical"])
        
        noise_map = {"Low": 0.1, "Medium": 0.4, "High": 0.8, "Critical": 1.5}
        noise_std = noise_map[snr_setting]
    
    # Simulation
    t = np.linspace(0, 1, 500)
    clean_sig = np.sin(2*np.pi*5*t) * np.exp(-2*t) + 0.5 * np.sin(2*np.pi*20*t)
    noise = np.random.normal(0, noise_std, 500)
    received_sig = clean_sig + noise
    
    h_opt = design_wiener(received_sig, clean_sig, L_taps)
    restored_sig = signal.lfilter(h_opt, 1, received_sig)
    
    mse_raw = np.mean((clean_sig - received_sig)**2)
    mse_filt = np.mean((clean_sig - restored_sig)**2)
    
    with col_vis:
        m1, m2, m3 = st.columns(3)
        m1.metric("Input Quality (MSE)", f"{mse_raw:.4f}")
        m2.metric("Restored Quality (MSE)", f"{mse_filt:.4f}")
        m3.metric("Signal Boost", f"+{10 * np.log10(mse_raw / mse_filt):.1f} dB")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(t, received_sig, color=current_theme['text'], alpha=0.3, label='Corrupted Input')
        ax.plot(t, clean_sig, color='green', linewidth=1, alpha=0.6, label='True Target')
        ax.plot(t, restored_sig, color=current_theme['accent'], linewidth=2, label='Wiener Estimate')
        
        ax.set_title("Live Signal Feed")
        ax.legend()
        
        apply_theme_to_plot(fig, ax, current_theme)
        st.pyplot(fig)

# ==============================================================================
# MISSION 2: OPERATION HUNTER (Matched Filter)
# ==============================================================================
elif mission == "OP: HUNTER (Matched)":
    st.header("Target Detection")
    
    col_game, col_radar = st.columns([1, 2])
    
    with col_game:
        target_shape = st.selectbox("Target Signature", ["Chirp (Sonar)", "Gaussian (Radar)", "Rect (Pulse)"])
        noise_pwr = st.slider("Atmospheric Noise", 0.5, 3.0, 1.0)
        
        if 'target_loc' not in st.session_state: st.session_state['target_loc'] = 150
        if st.button("🎲 Scramble Location"):
            st.session_state['target_loc'] = np.random.randint(50, 450)
            
    # Simulation
    N = 500
    L_sig = 60
    if target_shape == "Chirp (Sonar)":
        sig = signal.chirp(np.linspace(0, 1, L_sig), f0=1, f1=10, t1=1, method='linear')
    elif target_shape == "Gaussian (Radar)":
        sig = signal.gaussian(L_sig, std=10)
    else:
        sig = np.ones(L_sig)
        
    env_noise = np.random.normal(0, noise_pwr, N)
    received = env_noise.copy()
    loc = st.session_state['target_loc']
    received[loc:loc+L_sig] += sig
    
    # Matched Filter
    h_matched = sig[::-1] 
    detection_score = signal.convolve(received, h_matched, mode='same')
    detected_loc = np.argmax(np.abs(detection_score)) - L_sig//2
    
    with col_radar:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 1. Raw Input
        ax1.plot(received, color=current_theme['text'], lw=0.8, alpha=0.5)
        ax1.set_title("Raw Sensor Data")
        
        # 2. Matched Filter Output
        ax2.plot(detection_score, color=current_theme['accent'], lw=1.5)
        ax2.set_title("Matched Filter Output (Correlation Energy)")
        
        # Draw Peak
        peak_val = np.max(np.abs(detection_score))
        peak_idx = np.argmax(np.abs(detection_score))
        ax2.axvline(peak_idx, color='red', linestyle='--', alpha=0.8)
        ax2.text(peak_idx+10, peak_val, "TARGET LOCK", color='red', fontweight='bold')
        
        apply_theme_to_plot(fig, [ax1, ax2], current_theme)
        st.pyplot(fig)
        
        if abs(detected_loc - loc) < 5:
            st.success(f"🎯 **TARGET ACQUIRED!**")
        else:
            st.error(f"❌ **TARGET LOST.** Increase Signal Power!")

# ==============================================================================
# MISSION 3: OPERATION ORACLE (Prediction)
# ==============================================================================
elif mission == "OP: ORACLE (Prediction)":
    st.header("Future Prediction")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        p_order = st.slider("Prediction Order", 1, 20, 5)
        horizon = st.slider("Forecast Horizon", 10, 100, 50)
        st.caption("System: AR(2) Process")
        
    with col_p2:
        data_len = 300
        full_signal = generate_ar_signal(data_len + horizon)
        train_sig = full_signal[:data_len]
        truth_future = full_signal[data_len:]
        
        # LPC
        corr = np.correlate(train_sig, train_sig, 'full')
        mid = len(corr)//2
        r = corr[mid:mid+p_order+1]
        
        try:
            a_coeffs = np.linalg.solve(toeplitz(r[:-1]), r[1:])
            
            # Predict
            buffer = list(train_sig[-p_order:])
            predictions = []
            for _ in range(horizon):
                pred_val = np.dot(a_coeffs, buffer[::-1])
                predictions.append(pred_val)
                buffer.pop(0)
                buffer.append(pred_val)
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(range(data_len-50, data_len), train_sig[-50:], color=current_theme['text'], label='Observed History')
            ax.plot(range(data_len, data_len+horizon), truth_future, color='gray', linestyle=':', label='Actual Future')
            ax.plot(range(data_len, data_len+horizon), predictions, color=current_theme['accent'], linewidth=2, marker='o', markersize=3, label='LPC Prediction')
            
            ax.set_title(f"Linear Prediction (Order p={p_order})")
            ax.legend()
            
            apply_theme_to_plot(fig, ax, current_theme)
            st.pyplot(fig)
            
        except:
            st.error("Matrix Singularity. Try adjusting parameters.")
        with st.expander("📂 Mission Intel: How LPC Works"):
            st.markdown(r"""
            **Linear Prediction:**
            We assume the current sample is a linear combination of past samples:
            $$ \hat{x}(n) = \sum_{k=1}^{p} a_k x(n-k) $$
        
            The coefficients $a_k$ are found by matching the autocorrelation of the predictor to the signal. This is why it works well for speech and AR processes.
            """)



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.linalg import toeplitz

# # --- Page Config ---
# st.set_page_config(
#     page_title="Signal Operations Center",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Custom CSS for "Operations" Look ---
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #0e1117; 
#         color: #fafafa;
#     }
#     .metric-container {
#         border: 1px solid #333;
#         padding: 10px;
#         border-radius: 5px;
#         background-color: #262730;
#     }
#     div[data-testid="stMetricValue"] {
#         font-family: 'Courier New', Courier, monospace;
#         color: #00ff00; /* Radar Green */
#     }
#     h1, h2, h3 {
#         font-family: 'Helvetica', sans-serif;
#         color: #e0e0e0;
#     }
#     .stButton>button {
#         border-radius: 20px;
#         border: 1px solid #4b4b4b;
#         background-color: #262730;
#         color: white;
#     }
#     .stButton>button:hover {
#         border-color: #00ff00;
#         color: #00ff00;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- Helper Functions ---
# def design_wiener(noisy, clean, L):
#     # Cross-correlation r_yx
#     r_yx = np.correlate(clean, noisy, mode='full')
#     center = len(r_yx) // 2
#     r_yx = r_yx[center:center+L]
    
#     # Auto-correlation R_yy
#     r_yy = np.correlate(noisy, noisy, mode='full')
#     center_y = len(r_yy) // 2
#     r_yy = r_yy[center_y:center_y+L]
#     R_yy = toeplitz(r_yy)
    
#     # Solve
#     try:
#         h = np.linalg.solve(R_yy + np.eye(L)*1e-6, r_yx)
#         return h
#     except:
#         return np.zeros(L)

# def generate_ar_signal(n_samples):
#     np.random.seed(42)
#     w = np.random.normal(0, 1, n_samples)
#     x = np.zeros(n_samples)
#     # AR(2) process: x[n] = 0.75x[n-1] - 0.5x[n-2] + w[n]
#     for n in range(2, n_samples):
#         x[n] = 0.75*x[n-1] - 0.5*x[n-2] + w[n]
#     return x / np.max(np.abs(x))

# # --- Sidebar "Mission Control" ---
# with st.sidebar:
#     st.title("📡 OPS CENTER")
#     st.markdown("Select Active Mission:")
    
#     mission = st.radio(
#         "",
#         ["OP: HUNTER (Matched)", "OP: CLARITY (Wiener)", "OP: ORACLE (Prediction)"],
#         index=0
#     )
    
#     st.divider()
#     st.info("""
#     **Mission Status:**
#     Active
#     **System:** Online
#     """)

# # ==============================================================================
# # MISSION 1: OPERATION CLARITY (Wiener Filter)
# # ==============================================================================
# if mission == "OP: CLARITY (Wiener)":
#     st.header("Operation CLARITY: Signal Restoration")
#     st.markdown("""
#     **Briefing:** We have intercepted a corrupted transmission. 
#     **Objective:** Configure the **Wiener Filter** to minimize Mean Squared Error (MSE) and recover the intelligence.
#     """)
    
#     # --- Mission Controls ---
#     col_ctrl, col_vis = st.columns([1, 3])
    
#     with col_ctrl:
#         st.subheader("Filter Params")
#         L_taps = st.number_input("Filter Taps (L)", 2, 100, 15)
#         snr_setting = st.select_slider("Jamming Level (Noise)", options=["Low", "Medium", "High", "Critical"])
        
#         noise_map = {"Low": 0.1, "Medium": 0.4, "High": 0.8, "Critical": 1.5}
#         noise_std = noise_map[snr_setting]
        
#         if st.button("🔄 New Transmission"):
#             st.session_state['noise_seed'] = np.random.randint(0, 1000)
    
#     # --- Simulation ---
#     if 'noise_seed' not in st.session_state: st.session_state['noise_seed'] = 42
#     np.random.seed(st.session_state['noise_seed'])
    
#     t = np.linspace(0, 1, 500)
#     # "Intelligence" signal (unknown to receiver typically, but known for training)
#     clean_sig = np.sin(2*np.pi*5*t) * np.exp(-2*t) + 0.5 * np.sin(2*np.pi*20*t)
#     noise = np.random.normal(0, noise_std, 500)
#     received_sig = clean_sig + noise
    
#     # Apply Wiener
#     h_opt = design_wiener(received_sig, clean_sig, L_taps)
#     restored_sig = signal.lfilter(h_opt, 1, received_sig)
    
#     # Calc Metrics
#     mse_raw = np.mean((clean_sig - received_sig)**2)
#     mse_filt = np.mean((clean_sig - restored_sig)**2)
#     improvement = 10 * np.log10(mse_raw / mse_filt)
    
#     # --- Visuals ---
#     with col_vis:
#         # Dashboard Metrics
#         m1, m2, m3 = st.columns(3)
#         m1.metric("Input Quality (MSE)", f"{mse_raw:.4f}")
#         m2.metric("Restored Quality (MSE)", f"{mse_filt:.4f}")
#         m3.metric("Signal Boost", f"+{improvement:.1f} dB", delta_color="normal")
        
#         # Plot
#         fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
#         ax.set_facecolor('#0e1117')
        
#         ax.plot(t, received_sig, color='#444444', alpha=0.6, label='Corrupted Input')
#         ax.plot(t, clean_sig, color='#00ff00', linewidth=2, alpha=0.4, label='True Intel (Target)')
#         ax.plot(t, restored_sig, color='#00ccff', linewidth=1.5, linestyle='--', label='Wiener Estimate')
        
#         ax.set_title("Live Signal Feed", color='white')
#         ax.tick_params(colors='white')
#         ax.spines['bottom'].set_color('white')
#         ax.spines['left'].set_color('white')
#         ax.legend(facecolor='#262730', labelcolor='white')
#         ax.grid(True, color='#333333', linestyle=':')
        
#         st.pyplot(fig)
        
#         with st.expander("📂 Access Technical Schematics"):
#             st.markdown(r"""
#             **Wiener Filter Logic:**
#             The filter coefficients $\mathbf{h}$ are calculated by solving the Wiener-Hopf equation:
#             $$ \mathbf{h} = \mathbf{R}_{yy}^{-1} \mathbf{r}_{yx} $$
#             This finds the mathematically optimal filter that separates the signal from the noise based on their statistical correlation.
#             """)

# # ==============================================================================
# # MISSION 2: OPERATION HUNTER (Matched Filter)
# # ==============================================================================
# elif mission == "OP: HUNTER (Matched)":
#     st.header("Operation HUNTER: Target Detection")
#     st.markdown("""
#     **Briefing:** A stealth target is hidden somewhere in the noise floor. 
#     **Objective:** Use the **Matched Filter** to maximize SNR and identify the target's location index.
#     """)
    
#     # --- Game Controls ---
#     col_game, col_radar = st.columns([1, 2])
    
#     with col_game:
#         st.subheader("Sonar Settings")
#         target_shape = st.selectbox("Target Signature", ["Chirp (Sonar)", "Gaussian (Radar)", "Rect (Pulse)"])
#         noise_pwr = st.slider("Atmospheric Noise", 0.5, 3.0, 1.0)
        
#         if 'target_loc' not in st.session_state: st.session_state['target_loc'] = 150
        
#         if st.button("🎲 Scramble Target Location"):
#             st.session_state['target_loc'] = np.random.randint(50, 450)
            
#     # --- Simulation ---
#     N = 500
#     t = np.linspace(0, 1, N)
    
#     # Define Target Signature
#     L_sig = 60
#     if target_shape == "Chirp (Sonar)":
#         sig = signal.chirp(np.linspace(0, 1, L_sig), f0=1, f1=10, t1=1, method='linear')
#     elif target_shape == "Gaussian (Radar)":
#         sig = signal.gaussian(L_sig, std=10)
#     else:
#         sig = np.ones(L_sig)
        
#     # Create Environment
#     env_noise = np.random.normal(0, noise_pwr, N)
#     received = env_noise.copy()
    
#     # Plant Target
#     loc = st.session_state['target_loc']
#     received[loc:loc+L_sig] += sig
    
#     # MATCHED FILTERING
#     # Filter is time-reversed signal
#     h_matched = sig[::-1] 
#     # Valid convolution to avoid boundary artifacts
#     detection_score = signal.convolve(received, h_matched, mode='same')
    
#     # Detect
#     detected_loc = np.argmax(np.abs(detection_score)) - L_sig//2 # Adjust for lag
    
#     # Check Success (tolerance of +/- 5 samples)
#     success = abs(detected_loc - loc) < 5
    
#     with col_radar:
#         # Visualizing as a "Radar Screen"
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e1117')
        
#         # 1. Raw Input
#         ax1.set_facecolor('black')
#         ax1.plot(received, color='#00ff00', lw=0.8, alpha=0.7)
#         ax1.set_title("Raw Sensor Data (High Noise)", color='#00ff00')
#         ax1.set_yticks([])
#         ax1.tick_params(colors='gray')
#         ax1.grid(True, color='#003300')
        
#         # 2. Matched Filter Output
#         ax2.set_facecolor('black')
#         ax2.plot(detection_score, color='#ff00ff', lw=1.5)
#         ax2.set_title("Matched Filter Output (Correlation Energy)", color='#ff00ff')
        
#         # Draw Crosshair at peak
#         peak_val = np.max(np.abs(detection_score))
#         peak_idx = np.argmax(np.abs(detection_score))
#         ax2.axvline(peak_idx, color='white', linestyle='--', alpha=0.5)
#         ax2.text(peak_idx+10, peak_val, "TARGET LOCK", color='white', fontweight='bold')
        
#         ax2.tick_params(colors='gray')
#         ax2.grid(True, color='#330033')
        
#         st.pyplot(fig)
        
#         if success:
#             st.success(f"🎯 **TARGET ACQUIRED!** True Loc: {loc} | Detected: {detected_loc}")
#         else:
#             st.error(f"❌ **TARGET LOST.** True Loc: {loc} | Detected: {detected_loc}. Increase Signal Power!")

# # ==============================================================================
# # MISSION 3: OPERATION ORACLE (Prediction)
# # ==============================================================================
# elif mission == "OP: ORACLE (Prediction)":
#     st.header("Operation ORACLE: Future Prediction")
#     st.markdown("""
#     **Briefing:** We need to predict the next steps of a chaotic system to prevent failure.
#     **Objective:** Use **Linear Predictive Coding (LPC)** to predict samples $x[n]$ based on past $p$ samples.
#     """)
    
#     col_p1, col_p2 = st.columns([1, 2])
    
#     with col_p1:
#         # st.subheader("Predictor Config")
#         p_order = st.slider("Prediction Order (Past Samples used)", 1, 20, 5)
#         horizon = st.slider("Forecast Horizon (Samples)", 10, 100, 50)
        
#         # st.markdown("---")
#         st.write("System: AR(2) Process")
#         st.caption("$x[n] = 0.75x[n-1] - 0.5x[n-2] + noise$")
        
#     with col_p2:
#         # Generate Data
#         data_len = 300
#         full_signal = generate_ar_signal(data_len + horizon)
        
#         # Training Data (observed)
#         train_sig = full_signal[:data_len]
#         # Future Data (to predict)
#         truth_future = full_signal[data_len:]
        
#         # Train LPC (Yule-Walker)
#         # 1. Autocorrelation
#         corr = np.correlate(train_sig, train_sig, 'full')
#         mid = len(corr)//2
#         r = corr[mid:mid+p_order+1]
        
#         # 2. Levinson-Durbin or simple solve
#         R = toeplitz(r[:-1])
#         b = r[1:]
#         try:
#             # coeffs a = [a1, a2, ... ap]
#             a_coeffs = np.linalg.solve(R, b)
            
#             # Predict Future (Recursive)
#             # Start with last p samples from training
#             buffer = list(train_sig[-p_order:])
#             predictions = []
            
#             for _ in range(horizon):
#                 # x_hat[n] = sum(a_k * x[n-k])
#                 # buffer is [x[n-p], ... x[n-1]]
#                 # we need to reverse buffer to match a_coeffs [a1...ap]
#                 # a1 multiplies x[n-1] (last in buffer)
#                 pred_val = np.dot(a_coeffs, buffer[::-1])
#                 predictions.append(pred_val)
                
#                 # Update buffer
#                 buffer.pop(0)
#                 buffer.append(pred_val)
                
#             # Plotting
#             fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
#             ax.set_facecolor('#0e1117')
            
#             # Plot History
#             ax.plot(range(data_len-50, data_len), train_sig[-50:], color='white', label='Observed History')
            
#             # Plot Truth
#             ax.plot(range(data_len, data_len+horizon), truth_future, color='gray', linestyle=':', label='Actual Future')
            
#             # Plot Prediction
#             ax.plot(range(data_len, data_len+horizon), predictions, color='#ffcc00', linewidth=2, marker='o', markersize=3, label='LPC Prediction')
            
#             ax.set_title(f"Linear Prediction (Order p={p_order})", color='white')
#             ax.legend(facecolor='#262730', labelcolor='white')
#             ax.grid(True, color='#333333')
#             ax.tick_params(colors='white')
#             ax.spines['bottom'].set_color('white')
#             ax.spines['left'].set_color('white')
            
#             st.pyplot(fig)
            
#             # Error Metric
#             mse_pred = np.mean((truth_future - predictions)**2)
#             st.caption(f"Prediction MSE: {mse_pred:.5f}")
            
#         except np.linalg.LinAlgError:
#             st.error("Singular Matrix: Try different order or regenerate signal.")

#     with st.expander("📂 Mission Intel: How LPC Works"):
#         st.markdown(r"""
#         **Linear Prediction:**
#         We assume the current sample is a linear combination of past samples:
#         $$ \hat{x}(n) = \sum_{k=1}^{p} a_k x(n-k) $$
        
#         The coefficients $a_k$ are found by matching the autocorrelation of the predictor to the signal. This is why it works well for speech and AR processes.
#         """)
