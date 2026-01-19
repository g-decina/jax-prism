"""Shared pytest fixtures for JAX-Prism tests."""

import jax
import jax.numpy as jnp
import pytest

from jax_prism.data.batch import TimeSeriesBatch


@pytest.fixture
def rng_key():
    """Provide a deterministic PRNG key for reproducible tests."""
    return jax.random.key(42)


@pytest.fixture
def sample_batch():
    """Provide a minimal TimeSeriesBatch for testing.

    Creates a batch with:
    - 4 sequences
    - 32 past time steps
    - 8 future time steps
    - 2 known covariates
    - 3 observed covariates
    - 4 static features
    """
    B, T_past, T_future = 4, 32, 8
    F_known, F_obs, F_static = 2, 3, 4

    return TimeSeriesBatch(
        past_targets=jnp.ones((B, T_past)),
        future_targets=jnp.ones((B, T_future)),
        past_observed_covariates=jnp.ones((B, T_past, F_obs)),
        past_known_covariates=jnp.ones((B, T_past, F_known)),
        future_known_covariates=jnp.ones((B, T_future, F_known)),
        static_covariates=jnp.ones((B, F_static)),
        mask=jnp.ones((B,)),
    )


@pytest.fixture
def minimal_batch():
    """Provide the simplest possible TimeSeriesBatch.

    Just past_targets—no covariates, no future.
    """
    return TimeSeriesBatch(
        past_targets=jnp.ones((2, 16)),
    )
