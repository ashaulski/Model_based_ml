import numpy as np
from scipy.signal import upfirdn
from dpd_ml_project import config



_FFT_SIZE = config.Config.FFT_SIZE
_CP_LEN = config.Config.CP_LEN
_PILOT_CARRIERS = config.Config.PILOT_CARRIERS
_PILOT_VALUES = config.Config.PILOT_VALUES
_DATA_CARRIERS = config.Config.DATA_CARRIERS

# Half-band FIR filter coefficients for 2x upsampling (computed once at module load)
hb_coef = [0.000337282995741736,0,-0.000924412931855515,0,0.0021000195146104,0,-0.00414379676607774,0,0.00744079197463427,0,-0.0124846122449978,0,0.0199348019109818,0,-0.0307606060726599,0,0.0466096805827025,0,-0.0708564626549998,0,0.112212178127729,0,-0.20281864831572,0,0.633435329860412,1,0.633435329860412,0,-0.20281864831572,0,0.112212178127729,0,-0.0708564626549998,0,0.0466096805827025,0,-0.0307606060726599,0,0.0199348019109818,0,-0.0124846122449978,0,0.00744079197463427,0,-0.00414379676607774,0,0.0021000195146104,0,-0.000924412931855515,0,0.000337282995741736]
# hb_coef = [0,-0.000042083934478,0,0.000033107525775,0,-0.000045878242574,0,0.000061705492505,0,-0.000081049786215,0,0.000104461316851,0,-0.000132424341069,0,0.000165665552783,0,-0.000204704756985,0,0.000250317618126,0,-0.000303278226915,0,0.000364328381252,0,-0.000434357417162,0,0.000514290282690,0,-0.000605123229032,0,0.000707862047762,0,-0.000823617711710,0,0.000953615819749,0,-0.001099116356354,0,0.001261485055629,0,-0.001442192116396,0,0.001642848980775,0,-0.001865217376954,0,0.002111230289153,0,-0.002383045397211,0,0.002683052106942,0,-0.003013966408123,0,0.003378912625083,0,-0.003781505075232,0,0.004225973182802,0,-0.004717322742088,0,0.005261585371873,0,-0.005866104521191,0,0.006539951962746,0,-0.007294516538107,0,0.008144306320844,0,-0.009108130625885,0,0.010210837242623,0,-0.011485975965138,0,0.012979968734568,0,-0.014758911775743,0,0.016920202052852,0,-0.019613422516790,0,0.023080519087792,0,-0.027739814587013,0,0.034382040994120,0,-0.044702415148599,0,0.063109653620542,0,-0.105771089660046,0,0.318199015488867,0.500000000000000,0.318199015488867,0,-0.105771089660046,0,0.063109653620542,0,-0.044702415148599,0,0.034382040994120,0,-0.027739814587013,0,0.023080519087792,0,-0.019613422516790,0,0.016920202052852,0,-0.014758911775743,0,0.012979968734568,0,-0.011485975965138,0,0.010210837242623,0,-0.009108130625885,0,0.008144306320844,0,-0.007294516538107,0,0.006539951962746,0,-0.005866104521191,0,0.005261585371873,0,-0.004717322742088,0,0.004225973182802,0,-0.003781505075232,0,0.003378912625083,0,-0.003013966408123,0,0.002683052106942,0,-0.002383045397211,0,0.002111230289153,0,-0.001865217376954,0,0.001642848980775,0,-0.001442192116396,0,0.001261485055629,0,-0.001099116356354,0,0.000953615819749,0,-0.000823617711710,0,0.000707862047762,0,-0.000605123229032,0,0.000514290282690,0,-0.000434357417162,0,0.000364328381252,0,-0.000303278226915,0,0.000250317618126,0,-0.000204704756985,0,0.000165665552783,0,-0.000132424341069,0,0.000104461316851,0,-0.000081049786215,0,0.000061705492505,0,-0.000045878242574,0,0.000033107525775,0,-0.000042083934478,0]*2
def apply_usx32(x, factor=32):
    """Upsample by 32x using 5 stages of 2x upsampling with half-band FIR filters."""
    h = hb_coef
    x_up = x
    for _ in range(5):
        x_up = upfirdn(h, x_up, up=2)
    return np.asarray(x_up, dtype=complex)

def _bpsk_map(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=int)
    return np.where(bits == 0, 1 + 0j, -1 + 0j).astype(complex)


def _data_subcarriers() -> list[int]:
    used = [k for k in range(-26, 26) if k != 0]
    return [k for k in used if k not in _PILOT_CARRIERS]


def _carrier_to_index(k: int) -> int:
    return k % _FFT_SIZE


def apply_usx32(x, factor=32):
    """Upsample by 32x using 5 stages of 2x upsampling with half-band FIR filters."""
    h = hb_coef
    x_up = x
    for _ in range(5):
        x_up = upfirdn(h, x_up, up=2) # compensate for gain of FIR filter (up by 2, so gain is 2 per stage)
    # # remove filter delay (half the filter length) to align with original signal timing
    # filter_gdelay = (len(h) - 1) // 2
    # delay = filter_gdelay * (factor)-filter_gdelay  # total delay after 5 stages   
    # signal_len = len(x)*factor
    # x_up_no_dly = x_up[delay:delay+signal_len]  # compensate for delay

    # return x_up_no_dly.tolist()
    return np.asarray(x_up, dtype=complex)

def apply_dsx32(x, factor=32):
    """Downsample by 32x using 5 stages of 2x downsampling with half-band FIR filters."""
    # normelize filter coefficients to ensure unity gain at DC (important for downsampling)
    h = hb_coef
    x_down = x
    for _ in range(5):
        x_down = upfirdn(h, x_down, down=2)/2 # compensate for gain of FIR filter

    
    return np.asarray(x_down, dtype=complex)


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
