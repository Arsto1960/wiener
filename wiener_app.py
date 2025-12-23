import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz, inv

# --- Page Config ---
st.set_page_config(
    page_title="Wiener & Matched Filter Explorer",
    page_icon="🎯",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Optimal Filtering & Prediction")
st.markdown("""
Explore optimal filters that minimize error (**Wiener**), maximize SNR (**Matched**), or predict future values (**LPC**).
""")

# --- Helper Functions ---

def design_wiener_filter(y, x, L):
    """
    Design Wiener filter h to estimate x from y.
    Solve: R_yy * h = r_yx
    """
    # Estimate Autocorrelation of y (R_yy)
    # Using toeplitz matrix approach for simplicity on short signals
    # Or correlation method. Let's use correlation method.
    
    # 1. Auto-correlation of y
    r_yy = np.correlate(y, y, mode='full')
    center = len(r_yy) // 2
    r_yy = r_yy[center:center+L]
    R_yy = toeplitz(r_yy)
    
    # 2. Cross-correlation of y and x (r_yx)
    # We want to estimate x[n] from y[n], y[n-1]...
    r_yx = np.correlate(x, y, mode='full')
    center_yx = len(r_yx) // 2
    # We need lags 0 to L-1
    r_yx = r_yx[center_yx:center_yx+L]
    
    # 3. Solve R_yy * h = r_yx
    # Add small regularization to diagonal for stability
    R_yy = R_yy + np.eye(L) * 1e-6
    h = np.linalg.solve(R_yy, r_yx)
    
    return h

def design_lpc_predictor(x, order):
    """
    Compute LPC coefficients using Autocorrelation method (Yule-Walker).
    """
    # 1. Autocorrelation of x
    r_xx = np.correlate(x, x, mode='full')
    center = len(r_xx) // 2
    r_xx = r_xx[center:center+order+1]
    
    # R * a = r
    # R is Toeplitz of r_xx[0...p-1]
    # r is r_xx[1...p]
    R = toeplitz(r_xx[:order])
    r = r_xx[1:]
    
    # Solve
    a = np.linalg.solve(R, r)
    return a

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Wiener Filter (Denoising)",
    "2️⃣ Matched Filter (Detection)",
    "3️⃣ Linear Prediction (LPC)"
])

# ==============================================================================
# TAB 1: WIENER FILTER
# ==============================================================================
with tab1:
    st.header("1. Wiener Filter: Minimum Mean Square Error")
    st.markdown(r"""
    The Wiener filter $h_W$ minimizes $E[(x(n) - \hat{x}(n))^2]$.
    Solution: $\mathbf{h} = \mathbf{R}_{yy}^{-1} \mathbf{r}_{yx}$.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Settings")
        snr_db = st.slider("SNR (dB)", -5, 20, 5)
        L_wiener = st.slider("Filter Length (L)", 2, 50, 10)
        
        # Signal Generation
        N_samples = 1000
        t = np.linspace(0, 1, N_samples)
        # Clean signal: sum of sinusoids (simulating speech-like harmonic structure)
        x_clean = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*25*t)
        
        # Add Noise
        sig_power = np.mean(x_clean**2)
        noise_power = sig_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), N_samples)
        y_noisy = x_clean + noise
        
    with col2:
        # Train Wiener Filter
        # In practice, we estimate statistics. Here we use the whole signal.
        h_opt = design_wiener_filter(y_noisy, x_clean, L_wiener)
        
        # Filter
        x_est = signal.lfilter(h_opt, 1, y_noisy)
        
        # MSE Calculation
        mse_noisy = np.mean((x_clean - y_noisy)**2)
        mse_wiener = np.mean((x_clean - x_est)**2)
        
        # --- Plots ---
        fig1, ax = plt.subplots(2, 1, figsize=(10, 8))
        fig1.patch.set_alpha(0)
        
        # Time Domain
        ax[0].plot(t[:200], x_clean[:200], 'k', linewidth=2, label="Original $x(n)$")
        ax[0].plot(t[:200], y_noisy[:200], 'gray', alpha=0.5, label="Noisy $y(n)$")
        ax[0].plot(t[:200], x_est[:200], 'r--', linewidth=1.5, label="Wiener Estimate $\hat{x}(n)$")
        ax[0].legend(loc="upper right")
        ax[0].set_title("Denoising Performance (First 200 samples)")
        ax[0].grid(True, alpha=0.3)
        
        # Filter Response
        w, H = signal.freqz(h_opt)
        ax[1].plot(w/np.pi, 20*np.log10(np.abs(H)+1e-10))
        ax[1].set_title(f"Wiener Filter Frequency Response (L={L_wiener})")
        ax[1].set_ylabel("Magnitude (dB)")
        ax[1].set_xlabel("Normalized Frequency ($\times \pi$)")
        ax[1].grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Noisy MSE", f"{mse_noisy:.4f}")
        c2.metric("Wiener MSE", f"{mse_wiener:.4f}", delta=f"{-(mse_noisy-mse_wiener)/mse_noisy*100:.1f}%")
        c3.metric("Improvement (dB)", f"{10*np.log10(mse_noisy/mse_wiener):.2f} dB")

# ==============================================================================
# TAB 2: MATCHED FILTER
# ==============================================================================
with tab2:
    st.header("2. Matched Filter: Signal Detection")
    st.markdown(r"""
    The Matched Filter maximizes SNR at the detection instant.
    Optimal Impulse Response: $h(n) = x(L-1-n)$ (Time-Reversed Signal).
    """)
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        # Pattern to detect
        pattern_type = st.selectbox("Signal Pattern", ["Rectangle Pulse", "Chirp", "Gaussian Pulse"])
        noise_lvl = st.slider("Noise Level (Std Dev)", 0.1, 2.0, 0.8)
        
        # Create Pattern
        L_pattern = 50
        t_pat = np.linspace(0, 1, L_pattern)
        
        if pattern_type == "Rectangle Pulse":
            pattern = np.ones(L_pattern)
        elif pattern_type == "Chirp":
            pattern = signal.chirp(t_pat, f0=1, f1=10, t1=1, method='linear')
        else:
            pattern = signal.gaussian(L_pattern, std=7)
            
        # Create Received Signal (Embedded in Noise)
        N_rx = 500
        rx_signal = np.random.normal(0, noise_lvl, N_rx)
        
        # Embed pattern at random location
        # np.random.seed(42) # Keep random for "game" feel or fix it? Let's fix for demo stability
        true_pos = 200
        rx_signal[true_pos:true_pos+L_pattern] += pattern
        
    with col_m2:
        # Design Matched Filter: Time Reverse the pattern
        h_matched = pattern[::-1]
        
        # Apply Filter (Convolution)
        # 'valid' would shrink it, 'same' keeps size, 'full' expands. 
        # Detection peak should be at end of pattern in 'full' convolution logic
        # or center depending on alignment.
        output = signal.lfilter(h_matched, 1, rx_signal)
        
        # Find Peak
        peak_idx = np.argmax(np.abs(output))
        
        # --- Plots ---
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10))
        fig2.patch.set_alpha(0)
        
        # 1. Pattern
        ax2[0].plot(pattern, 'b')
        ax2[0].set_title("Target Signal Pattern $s(n)$")
        ax2[0].grid(True, alpha=0.3)
        
        # 2. Noisy Input
        ax2[1].plot(rx_signal, 'gray', label="Received $r(n)$")
        # Highlight true position
        ax2[1].axvspan(true_pos, true_pos+L_pattern, color='green', alpha=0.2, label="True Location")
        ax2[1].set_title(f"Received Noisy Signal (Noise $\sigma={noise_lvl}$)")
        ax2[1].legend(loc="upper right")
        ax2[1].grid(True, alpha=0.3)
        
        # 3. Filter Output
        ax2[2].plot(output, 'purple', linewidth=1.5, label="Matched Filter Output")
        ax2[2].plot(peak_idx, output[peak_idx], 'rx', markersize=10, markeredgewidth=3, label="Detected Peak")
        ax2[2].set_title("Matched Filter Output (Correlation)")
        ax2[2].set_xlabel("Sample Index")
        ax2[2].legend()
        ax2[2].grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Detection Logic
        # Peak should be at (True Pos + Length - 1) roughly due to convolution delay
        expected_peak = true_pos + L_pattern - 1
        st.info(f"**Detection Result:** Peak at index {peak_idx}. (Expected approx {expected_peak}). The sharp peak indicates the presence of the signal!")

# ==============================================================================
# TAB 3: LINEAR PREDICTION (LPC)
# ==============================================================================
with tab3:
    st.header("3. Linear Predictive Coding (LPC)")
    st.markdown(r"""
    Predict the next sample based on past $p$ samples: $\hat{x}(n) = \sum a_k x(n-k)$.
    Commonly used for **Speech Compression**.
    """)
    
    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        st.subheader("Predictor Settings")
        order_p = st.slider("Prediction Order (p)", 1, 20, 10)
        
        # Generate Synthetic AR Signal (Speech-like)
        # Filter white noise through resonances
        np.random.seed(10)
        noise_exc = np.random.normal(0, 1, 300)
        # Formants filter (AR process)
        b_ar = [1]
        a_ar = [1, -1.2, 0.9, -0.4] # Arbitrary poles
        x_speech = signal.lfilter(b_ar, a_ar, noise_exc)
        x_speech = x_speech / np.max(np.abs(x_speech))
        
    with col_l2:
        # 1. Compute LPC Coefficients
        a_coeffs = design_lpc_predictor(x_speech, order_p)
        
        # 2. Predict Signal
        # Use lfilter with these coeffs to predict
        # Error filter A(z) = 1 - sum(ak z^-k) ? 
        # Definition: x_hat = sum(ak * x(n-k)). 
        # Prediction Error e(n) = x(n) - x_hat(n)
        # This is equivalent to filtering x with A(z) = [1, -a1, -a2...]
        
        coeffs_analysis = np.concatenate(([1], -a_coeffs))
        error_signal = signal.lfilter(coeffs_analysis, 1, x_speech)
        
        # Prediction signal x_hat = x - error
        x_hat = x_speech - error_signal
        
        # Prediction Gain
        pow_x = np.mean(x_speech**2)
        pow_e = np.mean(error_signal**2)
        pred_gain = 10 * np.log10(pow_x / pow_e)
        
        # --- Plots ---
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 8))
        fig3.patch.set_alpha(0)
        
        # Time Domain Prediction
        ax3[0].plot(x_speech, 'k', alpha=0.5, label="Original Signal")
        ax3[0].plot(x_hat, 'r--', label=f"LPC Prediction (p={order_p})")
        ax3[0].set_title("Signal vs Prediction")
        ax3[0].legend()
        ax3[0].grid(True, alpha=0.3)
        
        # Prediction Error
        ax3[1].plot(error_signal, 'g')
        ax3[1].set_title("Prediction Error (Residual)")
        ax3[1].grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        
        # Metrics
        st.success(f"""
        **Performance:**
        * **Prediction Gain:** {pred_gain:.2f} dB
        * **Interpretation:** The error signal has much lower energy than the original. In speech coding, we only transmit the coefficients $a_k$ and the (quantized) error, saving massive bandwidth!
        """)
