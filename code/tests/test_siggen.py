import numpy as np

from dpd_ml_project.siggen.SigGen import generate_lsig_stub, gen_lsig


def test_lsig_symbol_length() -> None:
    iq = gen_lsig()
    assert len(iq) > 80  # upsampled 32x via cascaded half-band filters


def test_lsig_has_non_zero_energy() -> None:
    iq = gen_lsig()
    assert sum(abs(x) ** 2 for x in iq) > 0.0


def test_lsig_repeat_bits_every_call_enabled() -> None:
    iq1 = gen_lsig(repeat_bits_every_call=True, bits_seed=123)
    iq2 = gen_lsig(repeat_bits_every_call=True, bits_seed=123)
    assert np.array_equal(iq1, iq2)


def test_lsig_repeat_bits_every_call_disabled() -> None:
    iq1 = gen_lsig(repeat_bits_every_call=False)
    iq2 = gen_lsig(repeat_bits_every_call=False)
    assert not np.array_equal(iq1, iq2)


def test_lsig_stub_length_default() -> None:
    iq = generate_lsig_stub()
    assert len(iq) == 1500
