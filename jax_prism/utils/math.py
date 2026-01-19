"""Math utilities for JAX-Prism."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def binom(n, k):
    """Compute binomial coefficient C(n, k).

    Warning: Overflows for large n. Use log_binom for numerical stability.

    Args:
        n: Total count.
        k: Selection count.

    Returns:
        C(n, k) = n! / (k! * (n-k)!)
    """
    return jnp.exp(log_binom(n, k))


def log_binom(n, k):
    """Compute log of binomial coefficient: log(C(n, k)).

    Numerically stable for large n using log-gamma.

    Args:
        n: Total count (can be array).
        k: Selection count (can be array).

    Returns:
        log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    """
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
