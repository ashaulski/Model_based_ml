import numpy as np
from dpd_ml_project import config



_FFT_SIZE = config.Config.FFT_SIZE
_CP_LEN = config.Config.CP_LEN
_PILOT_CARRIERS = config.Config.PILOT_CARRIERS
_PILOT_VALUES = config.Config.PILOT_VALUES
_DATA_CARRIERS = config.Config.DATA_CARRIERS


def _bpsk_map(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=int)
    return np.where(bits == 0, 1 + 0j, -1 + 0j).astype(complex)


def _carrier_to_index(k: int) -> int:
    return k % _FFT_SIZE


def gen_lsig(
    bypass: bool = False,
    signal_rms_dbp: float = config.Config.signal_rms_dbp,
    repeat_bits_every_call: bool = False,
    bits_seed: int = 0,
) -> np.ndarray:
    """Generate one OFDM symbol with BPSK data (80 samples with CP).

    If repeat_bits_every_call=True, the same random bit pattern is generated on each
    function call using bits_seed. If False, a fresh random bit pattern is generated.
    """
    _ = bypass  # source module keeps same behavior; flag kept for unified API

    # L-SIG is generated with random BPSK data
    # Note: in real Wi-Fi, L-SIG is encoded with a fixed rate-1/2 code and known bits, 
    # but for our purposes random bits are sufficient to test the signal processing chain.
    if repeat_bits_every_call:
        rng = np.random.default_rng(bits_seed)
        random_bits = rng.integers(0, 2, size=48)
    else:
        random_bits = np.random.randint(0, 2, size=48)
    data_symbols = _bpsk_map(random_bits)

    # place data and pilots in the frequency domain
    xf = np.zeros(_FFT_SIZE, dtype=complex)
    for carrier, sym in zip(_DATA_CARRIERS, data_symbols):
        xf[_carrier_to_index(carrier)] = sym
    for carrier, pilot in zip(_PILOT_CARRIERS, _PILOT_VALUES):
        xf[_carrier_to_index(carrier)] = complex(pilot, 0.0)

    # IFFT to time domain
    x_time = np.fft.ifft(np.asarray(xf, dtype=np.complex128), norm="ortho")
    # add cyclic prefix
    cp = x_time[-_CP_LEN:]
    x_time_w_cp = np.concatenate((cp, x_time))

    # --- Normalize input power ---
    x_time_w_cp = x_time_w_cp / (np.sqrt(np.mean(np.abs(x_time_w_cp)**2)) + 1e-12)
    rms_db = 10 * np.log10(np.mean(np.abs(x_time_w_cp)**2) + 1e-12)
    x_time_w_cp = x_time_w_cp * 10**(signal_rms_dbp/20)

    return x_time_w_cp


def generate_lsig_stub(num_samples: int = 1500) -> np.ndarray:
    """Backward-compatible helper: repeat L-SIG symbol to requested length."""
    if num_samples <= 0:
        return np.array([], dtype=complex)
    symbol = gen_lsig()
    reps = int(np.ceil(num_samples / len(symbol)))
    out = np.tile(symbol, reps)
    return out[:num_samples]
