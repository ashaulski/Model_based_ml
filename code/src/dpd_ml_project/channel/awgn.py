import numpy as np
from scipy.signal import lfilter


def apply_awgn(tx_iq: np.ndarray, snr_db: float = 70, bypass: bool = False) -> np.ndarray:
    """
    Additive White Gaussian Noise (AWGN) channel model.
    
    Parameters:
        tx_iq : list of complex (baseband IQ)
        snr_db: signal-to-noise ratio in dB (default 70)
    
    Returns:
        y : complex numpy array (AWGN channel output)
    """

    
    tx_iq = np.asarray(tx_iq, dtype=complex).reshape(-1)

    if bypass:
        return tx_iq.copy()

    signal_rms_db = 10 * np.log10(np.mean(np.abs(tx_iq)**2));

    noise_rms_db = signal_rms_db - snr_db
    noise = 10**(noise_rms_db/20) * (np.random.randn(len(tx_iq)) + 1j * np.random.randn(len(tx_iq)))
    y = tx_iq + noise

    return y
