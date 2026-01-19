"""Base utilities for privacy accountants.

This module provides shared constants and helper functions used by
concrete accountant implementations (RDP, GDP, zCDP).

No abstract base class — we use Protocol from _typing.py for the interface.
"""

import jax.numpy as jnp

from jax_prism._typing import Array

# =============================================================================
# Default Alpha Grid for RDP
# =============================================================================

# Standard grid covering a wide range of privacy regimes.
# Small α: better for high-privacy (large σ), many steps
# Large α: better for low-privacy (small σ), few steps
# The accountant optimizes over this grid when converting to (ε,δ)-DP.
DEFAULT_RDP_ORDERS: Array = jnp.array([
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0,
    48.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
])
"""Default Rényi orders for RDP accounting.

This grid is designed to give good coverage across different privacy
regimes. The conversion to (ε,δ)-DP optimizes over all orders.

References:
    - Mironov, "Rényi Differential Privacy", 2017
    - Abadi et al., "Deep Learning with Differential Privacy", 2016
"""


# =============================================================================
# Numerical Constants
# =============================================================================

MIN_ALPHA: float = 1.0 + 1e-10
"""Minimum valid α for RDP (must be > 1)."""

MAX_EPSILON: float = float("inf")
"""Returned when privacy is completely broken."""

MIN_SAMPLE_RATE: float = 1e-10
"""Minimum sample rate to avoid numerical issues."""
