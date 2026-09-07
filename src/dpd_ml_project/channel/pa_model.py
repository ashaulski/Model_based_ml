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


def apply_pa_model_torch(pred):
    """Differentiable PyTorch mirror of :func:`apply_pa_model`.

    Takes/returns real I/Q tensors of shape ``(batch, N, 2)`` so it can sit
    inside a training graph; the NumPy version stays the canonical evaluator.
    The cubic regrowth term is omitted because its canonical delay is 0 samples
    at the default 640 MHz rate.
    """
    import torch

    x = torch.complex(pred[..., 0], pred[..., 1])
    r = torch.abs(x)

    p, a_sat = 3.0, 1.0
    am_am = r / torch.pow(1 + torch.pow(r / a_sat, 2 * p), 1 / (2 * p))
    phi_max = np.deg2rad(8)
    am_pm = phi_max * torch.pow(r / a_sat, 2) / (1 + torch.pow(r / a_sat, 3))
    y = am_am * torch.exp(1j * (torch.angle(x) + am_pm))

    env = torch.abs(y)
    alpha = 0.98
    env_mem_steps, prev = [], None
    for idx in range(env.shape[1]):
        current = (1 - alpha) * env[:, idx] + (alpha * prev if prev is not None else 0.0)
        env_mem_steps.append(current)
        prev = current
    env_mem = torch.stack(env_mem_steps, dim=1)
    y = y * (1 - 0.15 * env_mem)

    return torch.stack((y.real, y.imag), dim=-1)
