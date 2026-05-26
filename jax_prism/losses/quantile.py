"""Quantile (Pinball) loss for quantile regression."""

import warnings

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

    When enforce_monotonicity=True, raw predictions are transformed to ensure
    quantile ordering (q_i <= q_{i+1}). The first output is the base quantile,
    and subsequent outputs are positive deltas added cumulatively.

    Attributes:
        quantiles: Array of quantile levels to predict (must be sorted).
        enforce_monotonicity: Whether to enforce q_i <= q_{i+1}.
    """

    def __init__(
        self,
        quantiles: Array,
        enforce_monotonicity: bool = True,
        calibration_weight: float | tuple[float, ...] = 1.0,
        calibration_sharpness: float = 10.0,
    ):
        """Initialize with target quantile levels.

        Args:
            quantiles: 1D array of quantile levels in (0, 1), sorted ascending.
                    e.g., jnp.array([0.1, 0.5, 0.9])
            enforce_monotonicity: If True, transform predictions to ensure
                    quantile ordering via cumulative softplus deltas.
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
        self.enforce_monotonicity = enforce_monotonicity
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

        # Warn if all calibration weights are zero with monotonicity enforcement
        if enforce_monotonicity and all(w == 0.0 for w in self._calibration_weights):
            warnings.warn(
                "All calibration_weights are 0.0 with enforce_monotonicity=True. "
                "This may cause prediction intervals to collapse. "
                "Consider calibration_weight >= 0.1.",
                UserWarning,
                stacklevel=2,
            )

    def transform_predictions(self, raw_predictions: Array) -> Array:
        """Transform raw predictions to centered monotonic quantiles.

        Uses a centered approach where median is independent and intervals
        expand outward via cumulative softplus deltas.

        For 5 quantiles [0.1, 0.25, 0.5, 0.75, 0.9], raw outputs are:
            raw[0] = median (q50, unconstrained)
            raw[1] = δ_inner_lower (q50 - q25 gap)
            raw[2] = δ_outer_lower (q25 - q10 gap)
            raw[3] = δ_inner_upper (q75 - q50 gap)
            raw[4] = δ_outer_upper (q90 - q75 gap)

        Args:
            raw_predictions: Raw model output, shape (..., num_quantiles).

        Returns:
            Transformed predictions with guaranteed q_i <= q_{i+1}.
        """
        if not self.enforce_monotonicity:
            return raw_predictions

        n_quantiles = raw_predictions.shape[-1]

        if n_quantiles % 2 == 0:
            raise ValueError("Number of quantiles must be odd (need a median).")

        n_pairs = (n_quantiles - 1) // 2

        # First output is the median (unconstrained)
        median = raw_predictions[..., 0:1]

        # Lower deltas: indices 1 to n_pairs (inner to outer from median)
        lower_deltas = jax.nn.softplus(raw_predictions[..., 1 : n_pairs + 1])

        # Upper deltas: indices n_pairs+1 to end (inner to outer from median)
        upper_deltas = jax.nn.softplus(raw_predictions[..., n_pairs + 1 :])

        # Cumulative sums to enforce monotonicity
        lower_cumsum = jnp.cumsum(lower_deltas, axis=-1)
        upper_cumsum = jnp.cumsum(upper_deltas, axis=-1)

        # Lower quantiles: median - cumsum, flip so outermost (q10) comes first
        lower_quantiles = median - jnp.flip(lower_cumsum, axis=-1)

        # Upper quantiles: median + cumsum
        upper_quantiles = median + upper_cumsum

        return jnp.concatenate([lower_quantiles, median, upper_quantiles], axis=-1)

    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None,
    ) -> Array:
        """Compute mean quantile loss.

        Args:
            predictions: Raw model output, shape (..., num_quantiles).
                If enforce_monotonicity=True, these are transformed first.
            targets: Ground truth values, shape (..., 1).
            mask: Optional mask, shape (..., 1). 1 = valid, 0 = ignore.

        Returns:
            Scalar mean quantile loss.
        """
        # Transform to monotonic quantiles if enabled
        predictions = self.transform_predictions(predictions)

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
