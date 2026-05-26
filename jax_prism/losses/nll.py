"""Negative Log-Likelihood loss for probabilistic forecasting."""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, DistributionHead


class NLLLoss:
    """Negative Log-Likelihood loss.

    Computes the negative log probability of targets under the predicted
    distribution. This is the standard loss for maximum likelihood training
    of probabilistic models.

    Optionally includes sigma regularization to prevent scale collapse,
    penalizing predicted σ values below a minimum threshold.

    Attributes:
        distribution: Distribution head to interpret raw predictions.
        sigma_reg_weight: Weight for sigma regularization term. None disables.
        min_sigma: Minimum sigma threshold for regularization.
    """

    def __init__(
        self,
        distribution: DistributionHead,
        sigma_reg_weight: float | None = None,
        min_sigma: float = 0.15,
        fixed_sigma: float | None = None,
    ):
        """Initialize with a distribution head.

        Args:
            distribution: DistributionHead instance (e.g., GaussianHead).
            sigma_reg_weight: Weight for regularization. None disables regularization.
            min_sigma: Target minimum sigma (in normalized space). Default 0.15.
            fixed_sigma: Fixed sigma value used during phased training.
        """
        self.distribution = distribution
        self.sigma_reg_weight = sigma_reg_weight
        self.min_sigma = min_sigma
        self.fixed_sigma = fixed_sigma

    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None,
    ) -> Array:
        """Compute mean negative log-likelihood.

        Args:
            predictions: Raw model output, shape (..., num_params).
            targets: Ground truth values, shape (..., 1).
            mask: Optional mask, shape (..., 1). 1 = valid, 0 = ignore.

        Returns:
            Scalar mean NLL loss (plus regularization if enabled).
        """
        # 1. Convert raw predictions to distribution parameters
        params = self.distribution.params_from_raw(predictions)

        # Optional: if fixed_sigma is provided, override scale (for two-phase training)
        if self.fixed_sigma is not None:
            params = {**params, "scale": jnp.full_like(params["scale"], self.fixed_sigma)}
        
        # 2. Compute log probabilities
        log_probs = self.distribution.log_prob(params, targets)

        # 3. Negate
        nll = -log_probs

        # 4. Apply mask if provided
        if mask is not None:
            nll = nll * mask
            loss = jnp.sum(nll) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            loss = jnp.mean(nll)

        # 5. Add sigma regularization if enabled
        if self.sigma_reg_weight is not None:
            # Penalize sigma values below min_sigma with L2 penalty
            # We penalize in log-space to avoid exploding gradients
            # at very small values of σ.
            log_ratio = jnp.log(self.min_sigma) - jnp.log(params["scale"])
            sigma_penalty = jnp.mean(jax.nn.relu(log_ratio) ** 2)
            return loss + self.sigma_reg_weight * sigma_penalty

        return loss