# Examples

Practical examples for using JAX-Prism.

## Table of Contents

- [Basic TFT Training](#basic-tft-training)
- [DP-SGD Training](#dp-sgd-training)
- [Quantile Forecasting](#quantile-forecasting)
- [Custom Distribution Head](#custom-distribution-head)
- [Working with Covariates](#working-with-covariates)
- [Evaluating Forecasts](#evaluating-forecasts)

---

## Basic TFT Training

Standard (non-private) training loop.

```python
import jax
import jax.numpy as jnp
import optax
import jax_prism as jp

# Configuration
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    encoder_length=168,
    decoder_length=24,
    num_known_features=3,
    num_output_params=2,  # Gaussian: (μ, σ)
)

# Initialize
key = jax.random.key(0)
model = jp.TemporalFusionTransformer(config)
dist_head = jp.GaussianHead()
loss_fn = jp.NLLLoss(dist_head)
optimizer = optax.adam(1e-3)

# Sample batch (replace with your data)
batch = jp.TimeSeriesBatch(
    past_targets=jax.random.normal(key, (32, 168, 1)),
    future_targets=jax.random.normal(key, (32, 24, 1)),
    past_known_covariates=jax.random.normal(key, (32, 168, 3)),
    future_known_covariates=jax.random.normal(key, (32, 24, 3)),
)

# Initialize parameters
params = model.init(key, batch)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, batch, key):
    def compute_loss(params):
        output = model.apply(
            params, batch, training=True,
            rngs={"dropout": key}
        )
        return loss_fn(output, batch.future_targets)

    loss, grads = jax.value_and_grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Training loop
for epoch in range(10):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, batch, subkey)
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

---

## DP-SGD Training

Training with differential privacy guarantees.

```python
import jax
import jax.numpy as jnp
import optax
import jax_prism as jp

# Model setup (same as above)
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    encoder_length=168,
    decoder_length=24,
    num_known_features=3,
    num_output_params=2,
)
model = jp.TemporalFusionTransformer(config)
dist_head = jp.GaussianHead()
optimizer = optax.adam(1e-3)

# Privacy parameters
clip_norm = 1.0
noise_multiplier = 1.1
delta = 1e-5
batch_size = 64
dataset_size = 10000
sample_rate = batch_size / dataset_size

# Initialize
key = jax.random.key(0)
batch = jp.TimeSeriesBatch(
    past_targets=jax.random.normal(key, (batch_size, 168, 1)),
    future_targets=jax.random.normal(key, (batch_size, 24, 1)),
    past_known_covariates=jax.random.normal(key, (batch_size, 168, 3)),
    future_known_covariates=jax.random.normal(key, (batch_size, 24, 3)),
)

params = model.init(key, batch)
opt_state = optimizer.init(params)
accountant = jp.RDPAccountant.create(noise_multiplier, sample_rate)


def loss_fn(params, batch):
    output = model.apply(params, batch, training=False)
    return jp.NLLLoss(dist_head)(output, batch.future_targets)


# DP Training loop
num_epochs = 10
target_epsilon = 5.0

for epoch in range(num_epochs):
    key, subkey = jax.random.split(key)

    # Compute DP gradients
    grads = jp.dp_gradients(
        loss_fn=loss_fn,
        params=params,
        batch=batch,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        key=subkey,
    )

    # Update model
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Update privacy accountant
    accountant = accountant.step(noise_multiplier, sample_rate)

    # Report privacy
    budget = accountant.get_privacy_spent(delta)
    print(f"Epoch {epoch}: ε={budget.epsilon:.2f}, δ={budget.delta:.0e}")

    # Early stopping if budget exceeded
    if budget.epsilon > target_epsilon:
        print(f"Stopping: privacy budget ({target_epsilon}) exceeded")
        break

print(f"\nFinal privacy: (ε={budget.epsilon:.2f}, δ={budget.delta:.0e})")
```

---

## Quantile Forecasting

Direct quantile regression without distributional assumptions.

```python
import jax
import jax.numpy as jnp
import optax
import jax_prism as jp

# Quantiles to predict
quantiles = jnp.array([0.1, 0.5, 0.9])

# Configure model for quantile output
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    encoder_length=168,
    decoder_length=24,
    num_known_features=3,
    num_output_params=len(quantiles),  # One output per quantile
)

model = jp.TemporalFusionTransformer(config)
dist_head = jp.QuantileHead(quantiles=quantiles)
loss_fn = jp.QuantileLoss(quantiles=quantiles)
optimizer = optax.adam(1e-3)

# Initialize
key = jax.random.key(0)
batch = jp.TimeSeriesBatch(
    past_targets=jax.random.normal(key, (32, 168, 1)),
    future_targets=jax.random.normal(key, (32, 24, 1)),
    past_known_covariates=jax.random.normal(key, (32, 168, 3)),
    future_known_covariates=jax.random.normal(key, (32, 24, 3)),
)

params = model.init(key, batch)
opt_state = optimizer.init(params)


@jax.jit
def train_step(params, opt_state, batch):
    def compute_loss(params):
        output = model.apply(params, batch, training=False)
        # Output shape: (B, T, Q) where Q = len(quantiles)
        # Expand targets: (B, T, 1) -> (B, T, 1, 1) for broadcast
        targets = batch.future_targets[..., jnp.newaxis]
        # Expand output: (B, T, Q) -> (B, T, 1, Q)
        output = output[:, :, jnp.newaxis, :]
        return jp.quantile_loss(targets, output, quantiles)

    loss, grads = jax.value_and_grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Training
for epoch in range(10):
    params, opt_state, loss = train_step(params, opt_state, batch)
    print(f"Epoch {epoch}: quantile_loss={loss:.4f}")

# Inference
output = model.apply(params, batch, training=False)
q10, q50, q90 = output[..., 0], output[..., 1], output[..., 2]

# Evaluate coverage
cov = jp.coverage(
    batch.future_targets,
    lower=q10[..., jnp.newaxis],
    upper=q90[..., jnp.newaxis],
)
print(f"80% interval coverage: {cov:.1%}")  # Should be ~80%
```

---

## Custom Distribution Head

Implementing a custom output distribution.

```python
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax_prism._typing import Array, PRNGKey, Shape


class StudentTHead:
    """Student-t distribution head for heavy-tailed forecasts."""

    @property
    def num_params(self) -> int:
        return 3  # (df, loc, scale)

    def params_from_raw(self, raw: Array) -> dict[str, Array]:
        """Transform raw output to distribution parameters."""
        df_raw, loc, scale_raw = jnp.split(raw, 3, axis=-1)

        # Ensure valid parameters
        df = jnp.exp(df_raw) + 2.0  # df > 2 for finite variance
        scale = jnp.exp(scale_raw) + 1e-6  # scale > 0

        return {
            "df": df.squeeze(-1),
            "loc": loc.squeeze(-1),
            "scale": scale.squeeze(-1),
        }

    def log_prob(self, params: dict[str, Array], targets: Array) -> Array:
        """Compute log probability under Student-t."""
        return stats.t.logpdf(
            targets,
            df=params["df"],
            loc=params["loc"],
            scale=params["scale"],
        )

    def sample(
        self,
        params: dict[str, Array],
        key: PRNGKey,
        sample_shape: Shape = (),
    ) -> Array:
        """Sample from Student-t distribution."""
        import jax.random as jr

        # Use normal/chi2 parameterization
        z = jr.normal(key, sample_shape + params["loc"].shape)
        key, subkey = jr.split(key)
        chi2 = jr.chisquare(subkey, params["df"], sample_shape + params["df"].shape)

        return params["loc"] + params["scale"] * z * jnp.sqrt(params["df"] / chi2)


# Usage
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    num_output_params=3,  # df, loc, scale
    # ...
)
model = jp.TemporalFusionTransformer(config)
dist_head = StudentTHead()

# Forward pass
output = model.apply(params, batch)
dist_params = dist_head.params_from_raw(output)
log_probs = dist_head.log_prob(dist_params, batch.future_targets)
```

---

## Working with Covariates

Using all covariate types in TFT.

```python
import jax
import jax.numpy as jnp
import jax_prism as jp

# Full covariate configuration
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    encoder_length=168,
    decoder_length=24,
    num_static_features=2,    # e.g., store_id, product_category
    num_known_features=5,     # e.g., day_of_week, month, holidays
    num_observed_features=3,  # e.g., weather, promotions
    num_output_params=2,
)

model = jp.TemporalFusionTransformer(config)

# Create batch with all covariate types
key = jax.random.key(0)
batch_size = 32

batch = jp.TimeSeriesBatch(
    # Target: sales quantity
    past_targets=jax.random.normal(key, (batch_size, 168, 1)),
    future_targets=jax.random.normal(key, (batch_size, 24, 1)),

    # Known covariates: calendar features (available in future)
    past_known_covariates=jax.random.normal(key, (batch_size, 168, 5)),
    future_known_covariates=jax.random.normal(key, (batch_size, 24, 5)),

    # Observed covariates: weather (only available in past)
    past_observed_covariates=jax.random.normal(key, (batch_size, 168, 3)),

    # Static covariates: store/product features
    static_covariates=jax.random.normal(key, (batch_size, 2)),
)

# Initialize and run
params = model.init(key, batch)
output = model.apply(params, batch)
print(f"Output shape: {output.shape}")  # (32, 24, 2)
```

---

## Evaluating Forecasts

Computing evaluation metrics.

```python
import jax.numpy as jnp
import jax_prism as jp

# Generate predictions
output = model.apply(params, test_batch, training=False)
dist_params = dist_head.params_from_raw(output)

# Point predictions (mean for Gaussian)
y_pred = dist_params["loc"]
y_true = test_batch.future_targets

# Point metrics
mae_score = jp.mae(y_true, y_pred[..., jnp.newaxis])
smape_score = jp.smape(y_true, y_pred[..., jnp.newaxis])

print(f"MAE: {mae_score:.4f}")
print(f"SMAPE: {smape_score:.2f}%")

# MASE (requires training data for scaling)
mase_score = jp.mase(
    y_true, y_pred[..., jnp.newaxis],
    y_train=train_batch.past_targets,
    seasonality=7,  # Weekly seasonality
)
print(f"MASE: {mase_score:.4f}")

# Probabilistic metrics with prediction intervals
# 80% interval: [q10, q90]
key = jax.random.key(123)
samples = dist_head.sample(dist_params, key, sample_shape=(1000,))
q10 = jnp.percentile(samples, 10, axis=0)
q90 = jnp.percentile(samples, 90, axis=0)

coverage_80 = jp.coverage(
    y_true,
    lower=q10[..., jnp.newaxis],
    upper=q90[..., jnp.newaxis],
)
print(f"80% coverage: {coverage_80:.1%}")

# Quantile loss
quantiles = jnp.array([0.1, 0.5, 0.9])
q_preds = jnp.stack([q10, dist_params["loc"], q90], axis=-1)
ql = jp.quantile_loss(
    y_true,
    q_preds[..., jnp.newaxis, :],
    quantiles,
)
print(f"Quantile loss: {ql:.4f}")
```

---

## Data Scaling

Normalizing time series before training.

```python
import jax.numpy as jnp
import jax_prism as jp

# Load your data
past_targets = jnp.array(...)  # (batch, time, features)

# Scale by last value (recommended for trends)
scaled_past, scale = jp.last_value_scale(past_targets)

# Or scale by median (robust to outliers)
scaled_past, scale = jp.median_scale(past_targets, k=10)

# Create batch with scaled data
batch = jp.TimeSeriesBatch(
    past_targets=scaled_past,
    future_targets=scaled_future,  # Scale future with same scale
    # ...
)

# After prediction, inverse scale
raw_output = model.apply(params, batch)
dist_params = dist_head.params_from_raw(raw_output)

# Inverse scale predictions
scaled_pred = dist_params["loc"]
original_pred = jp.inverse_scale(scaled_pred[..., jnp.newaxis], scale)
```
