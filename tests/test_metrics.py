from dpd_ml_project.metrics.evm import compute_evm


def test_evm_zero_for_identical_vectors() -> None:
    ref = [1 + 0j, 1 - 1j]
    meas = [1 + 0j, 1 - 1j]
    assert compute_evm(ref, meas) == 0.0
