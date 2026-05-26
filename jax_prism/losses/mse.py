"""Mean Squared Error for point forecasting."""

import jax.numpy as jnp

from jax_prism._typing import Array, DistributionHead

class MSELoss:
    """Mean Squared Error loss.
    
    Computes the mean squared error between predicted and target values.
    This is the standard loss for point-estimate regression models.
    
    Attributes:
        distribution: If provided, extracts μ via params_from_raw()["loc"].
                    If None, uses predictions[..., 0] directly.
    """
    
    def __init__(self, distribution: DistributionHead | None = None):
        """Initialize MSE loss.
        
        Args:
            distribution: Optional distribution head to extract μ from raw
                predictions. If None, assumes predictions[..., 0] is μ.
        """
        self.distribution = distribution
        
        
    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None
    ) -> Array:
        """Compute mean squared error loss.

        Args:
            predictions: Raw model output, shape (..., num_params).
            targets: Ground truth values, shape (..., 1).
            mask: Optional mask, shape (..., 1). 1 = valid, 0 = ignore.

        Returns:
            Scalar MSE loss.
        """
        
        if self.distribution is not None:
            params = self.distribution.params_from_raw(predictions)
            mu = params["loc"]
        else:
            mu = predictions[..., 0]
        
        mse = (mu - targets) ** 2
        
        if mask is not None:
            mse = mse * mask
            loss = jnp.sum(mse) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            loss = jnp.mean(mse)
            
        return loss