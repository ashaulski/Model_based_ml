def build_qm_basis(tx_iq: list[complex], order: int = 5) -> list[list[complex]]:
    """Build quasi-memoryless polynomial basis terms."""
    return [[x * (abs(x) ** (p - 1)) for p in range(1, order + 1)] for x in tx_iq]
