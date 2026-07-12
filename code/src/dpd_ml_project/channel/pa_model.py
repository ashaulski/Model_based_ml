import numpy as np
from scipy.signal import lfilter


def apply_pa_model(x: np.ndarray, fs=640e6, bypass: bool = False) -> np.ndarray:
    """
    Behavioral PA model approximating Skyworks SKY85355-class FEM.
    
    Parameters:
        tx_iq : list of complex (baseband IQ)
        fs: sampling frequency (Hz), default 640 MHz
    
    Returns:
        y : complex numpy array (PA output)
    """

    
    x = np.asarray(x, dtype=complex).reshape(-1)

    if bypass:
        return x.copy()

    snr_db = 70
    

    # --- AM-AM: Rapp model (typical WLAN PA compression) ---
    # Smooth saturation, no hard clipping
    p = 3.0             # smoothness factor
    A_sat = 1.0         # saturation amplitude
    r = np.abs(x)

    am_am = r / (1 + (r / A_sat)**(2 * p))**(1 / (2 * p))

    # --- AM-PM: mild phase distortion (typical WiFi PA) ---
    # few degrees near compression
    phi_max = np.deg2rad(8)   # max ~8 deg
    am_pm = phi_max * (r / A_sat)**2 / (1 + (r / A_sat)**3)

    # Apply AM-AM + AM-PM
    y = am_am * np.exp(1j * (np.angle(x) + am_pm))

    # --- Memory effects (weak, thermal/bias dynamics) ---
    # Simple IIR envelope memory
    env = np.abs(y)
    alpha = 0.98  # strong memory (slow thermal effect)
    env_mem = lfilter([1-alpha], [1, -alpha], env)

    # Gain droop with memory (supply/bias compression)
    k_mem = 0.15
    y = y * (1 - k_mem * env_mem)

    # --- Optional: mild spectral regrowth shaping ---
    # add small cubic memory polynomial term
    beta = 0.08
    delay = int(fs * 1e-9)  # ~1 ns electrical memory

    if delay > 0:
        x_del = np.concatenate([np.zeros(delay, dtype=complex), x[:-delay]])
        y += beta * x_del * np.abs(x_del)**2

    return y
