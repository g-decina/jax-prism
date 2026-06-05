"""Quantile (Pinball) loss for quantile regression."""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array


class QuantileLoss:
    """Quantile loss (pinball loss) for quantile regression.

    For a quantile level q ∈ (0,1) and error e = pred - target:
        L(e, q) = max(q * e, (q - 1) * e)
                = q * e      if e >= 0  (underprediction penalized by q)
                = (q-1) * e  if e < 0   (overprediction penalized by 1-q)

    This asymmetric loss causes the model to predict the q-th quantile.

    Expects predictions to already be transformed quantile values (e.g., from
    QuantileHead.params_from_raw). Use QuantileHead to enforce monotonicity.

    Attributes:
        quantiles: Array of quantile levels to predict (must be sorted).
    """

    def __init__(
        self,
        quantiles: Array,
        calibration_weight: float | tuple[float, ...] = 1.0,
        calibration_sharpness: float = 10.0,
    ):
        """Initialize with target quantile levels.

        Args:
            quantiles: 1D array of quantile levels in (0, 1), sorted ascending.
                    e.g., jnp.array([0.1, 0.5, 0.9])
            calibration_weight: Loss term penalizing coverage deviation
                    from target for symmetric quantile pairs. Can be:
                    - float: Same weight for all pairs
                    - tuple: Per-pair weights, ordered from inner to outer
                        e.g., for [0.1, 0.25, 0.5, 0.75, 0.9]:
                        (weight_50_PI, weight_80_PI) for pairs (0.25,0.75), (0.1,0.9)
            calibration_sharpness: Sharpness of soft coverage sigmoid (k).
                    Higher = sharper approximation to hard coverage.
        """
        self.quantiles = jnp.asarray(quantiles)
        self.calibration_sharpness = calibration_sharpness

        # Precompute symmetric quantile pairs and their target coverages
        # e.g., (0.1, 0.9) -> 0.8 coverage, (0.25, 0.75) -> 0.5 coverage
        # Pairs are ordered inner to outer (closest to median first)
        self._calibration_pairs = []
        q_array = jnp.asarray(quantiles)
        n = len(q_array)
        for i in range(n // 2 - 1, -1, -1):  # Reverse: inner pairs first
            lower_q = float(q_array[i])
            upper_q = float(q_array[n - 1 - i])
            if abs((1 - upper_q) - lower_q) < 0.01:  # Symmetric pair
                target_coverage = upper_q - lower_q
                self._calibration_pairs.append((i, n - 1 - i, target_coverage))

        # Handle calibration_weight as scalar or per-pair tuple
        if isinstance(calibration_weight, (list, tuple)):
            if len(calibration_weight) != len(self._calibration_pairs):
                raise ValueError(
                    f"Calibration_weight tuple length ({len(calibration_weight)}) "
                    f"must match number of symmetric pairs ({len(self._calibration_pairs)}). "
                    f"Pairs (inner to outer): {[(p[2]) for p in self._calibration_pairs]}"
                )
            self._calibration_weights = tuple(calibration_weight)
        else:
            self._calibration_weights = tuple(
                calibration_weight for _ in self._calibration_pairs
            )

    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None,
    ) -> Array:
        """Compute mean quantile loss.

        Args:
            predictions: Quantile predictions, shape (..., num_quantiles).
                Should be pre-transformed (e.g., via QuantileHead.params_from_raw).
            targets: Ground truth values, shape (..., 1).
            mask: Optional mask, shape (..., 1). 1 = valid, 0 = ignore.

        Returns:
            Scalar mean quantile loss.
        """
        # Expand targets for broadcasting: (...) -> (..., 1)
        targets_expanded = targets[..., jnp.newaxis]

        # Compute errors: pred - target for each quantile
        errors = predictions - targets_expanded  # (..., Q)

        # Reshape quantiles for broadcasting: (Q,) -> (1, ..., Q)
        q = self.quantiles.reshape((1,) * (predictions.ndim - 1) + (-1,))

        # Pinball loss: max(q * e, (q - 1) * e)
        loss_a = q * errors
        loss_b = (q - 1.0) * errors
        pinball = jnp.maximum(loss_a, loss_b)  # (..., Q)

        # Average over quantiles
        loss_per_point = jnp.mean(pinball, axis=-1)  # (...)

        # Apply mask if provided
        if mask is not None:
            loss_per_point = loss_per_point * mask
            base_loss = jnp.sum(loss_per_point) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            base_loss = jnp.mean(loss_per_point)

        # Add calibration loss if enabled (any non-zero weight)
        if any(w > 0 for w in self._calibration_weights) and len(self._calibration_pairs) > 0:
            calibration_loss = self._compute_calibration_loss(
                predictions, targets_expanded.squeeze(-1), mask
            )
            # Weights already applied per-pair in _compute_calibration_loss
            return base_loss + calibration_loss

        return base_loss

    def _compute_calibration_loss(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None,
    ) -> Array:
        """Compute soft calibration loss for symmetric quantile pairs.

        Args:
            predictions: Transformed quantile predictions, shape (..., Q).
            targets: Ground truth values, shape (...).
            mask: Optional mask.

        Returns:
            Scalar calibration loss (weighted average across pairs).
        """
        k = self.calibration_sharpness
        total_loss = jnp.array(0.0)
        total_weight = 0.0

        for (lower_idx, upper_idx, target_coverage), weight in zip(
            self._calibration_pairs, self._calibration_weights
        ):
            if weight == 0.0:
                continue

            q_lower = predictions[..., lower_idx]
            q_upper = predictions[..., upper_idx]

            # Soft coverage: P(target in [q_lower, q_upper])
            in_lower = jax.nn.sigmoid(k * (targets - q_lower))
            in_upper = jax.nn.sigmoid(k * (q_upper - targets))
            soft_in_interval = in_lower * in_upper

            # Compute mean coverage (respecting mask)
            if mask is not None:
                soft_coverage = jnp.sum(soft_in_interval * mask) / jnp.maximum(
                    jnp.sum(mask), 1.0
                )
            else:
                soft_coverage = jnp.mean(soft_in_interval)

            # Penalize deviation from target coverage (weighted)
            total_loss = total_loss + weight * (target_coverage - soft_coverage) ** 2
            total_weight += weight

        return total_loss / max(total_weight, 1.0)

def interval_regularization(
    quantile_values: Array,
    taus: Array,
    min_scale: float = 1.0,
) -> Array:
    """Penalize intervals that are too narrow relative to probability mass.

    Encourages spread between quantiles to prevent collapse to the median.

    Args:
        quantile_values: Predicted quantile values, shape (..., num_quantiles).
        taus: Quantile levels, shape (num_quantiles,).
        min_scale: Minimum allowed width per unit probability mass.

    Returns:
        Scalar mean shortfall penalty.
    """
    widths = jnp.diff(quantile_values, axis=-1)
    delta_tau = jnp.diff(taus)
    min_widths = min_scale * delta_tau
    shortfall = jnp.maximum(0, min_widths - widths)
    return jnp.mean(shortfall)