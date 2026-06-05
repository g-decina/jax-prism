# JAX-Prism

Privacy-preserving probabilistic forecasting with JAX.

JAX-Prism combines **differentially private training** (DP-SGD) with **Bayesian uncertainty quantification** for time series forecasting. Built on JAX, Flax, and NumPyro.

## Features

- **Differential Privacy**: Train models with formal (ε, δ)-DP guarantees via DP-SGD
- **RDP Accounting**: Track privacy budget using Rényi Differential Privacy with subsampling amplification
- **Modern TFT**: Temporal Fusion Transformer with RoPE, RMSNorm, SwiGLU, and configurable attention (MHA/MQA/GQA)
- **Probabilistic Outputs**: Gaussian and quantile distribution heads with calibrated uncertainty
- **Evaluation Metrics**: MAE, SMAPE, MASE, quantile loss, coverage

## Installation

```bash
# Requires Python 3.11+
pip install jax-prism
```

Or install from source:

```bash
git clone https://github.com/g-decina/jax-prism
cd jax-prism
poetry install
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import jax_prism as jp

# Create a batch of time series data
batch = jp.TimeSeriesBatch(
    past_targets=jnp.ones((32, 168, 1)),           # (batch, encoder_steps, targets)
    future_targets=jnp.ones((32, 24, 1)),          # (batch, decoder_steps, targets)
    past_known_covariates=jnp.ones((32, 168, 3)),  # (batch, encoder_steps, features)
    future_known_covariates=jnp.ones((32, 24, 3)), # (batch, decoder_steps, features)
)

# Configure and initialize the model
config = jp.TFTConfig(
    hidden_size=64,
    num_heads=4,
    encoder_length=168,
    decoder_length=24,
    num_known_features=3,
    num_output_params=2,  # (μ, σ) for Gaussian
)
model = jp.TemporalFusionTransformer(config)

# Initialize parameters
key = jax.random.key(0)
params = model.init(key, batch)

# Forward pass
output = model.apply(params, batch)  # (32, 24, 2)

# Interpret as Gaussian distribution
dist_head = jp.GaussianHead()
dist_params = dist_head.params_from_raw(output)
log_probs = dist_head.log_prob(dist_params, batch.future_targets)
```

## Training with Differential Privacy

```python
import optax
import jax_prism as jp

# Privacy parameters
clip_norm = 1.0
noise_multiplier = 1.1
sample_rate = 64 / 10000  # batch_size / dataset_size

# Initialize privacy accountant
accountant = jp.RDPAccountant.create(
    noise_multiplier=noise_multiplier,
    sample_rate=sample_rate,
)

# Compute differentially private gradients
def loss_fn(params, batch):
    output = model.apply(params, batch)
    return jp.NLLLoss(dist_head)(output, batch.future_targets)

key = jax.random.key(42)
dp_grads = jp.dp_gradients(
    loss_fn=loss_fn,
    params=params,
    batch=batch,
    clip_norm=clip_norm,
    noise_multiplier=noise_multiplier,
    key=key,
)

# Update accountant after each step
accountant = accountant.step(noise_multiplier, sample_rate)

# Check privacy budget
budget = accountant.get_privacy_spent(delta=1e-5)
print(f"Privacy spent: ε={budget.epsilon:.2f}, δ={budget.delta:.0e}")
```

## Scaling / Normalization

JAX-Prism provides privacy-compatible normalization (zero DP cost for per-window scaling):

```python
import jax_prism as jp

# Scale by last value in each series
scaled_x, scale = jp.last_value_scale(x)

# Scale by median of last k values
scaled_x, scale = jp.median_scale(x, k=10)

# Inverse scaling for predictions
predictions = jp.inverse_scale(scaled_predictions, scale)
```

## Evaluation Metrics

```python
import jax_prism as jp

# Point metrics
mae = jp.mae(y_true, y_pred)
smape = jp.smape(y_true, y_pred)
mase = jp.mase(y_true, y_pred, y_train, seasonality=7)

# Probabilistic metrics
ql = jp.quantile_loss(y_true, quantile_preds, quantiles)
cov = jp.coverage(y_true, lower, upper)
```

## API Reference

### Data

| Export | Description |
|--------|-------------|
| `TimeSeriesBatch` | Immutable batch structure for time series data |
| `last_value_scale` | Scale by last value in each series |
| `median_scale` | Scale by median of last k values |
| `fixed_scale` | Scale by user-provided values |
| `inverse_scale` | Reverse scaling for predictions |

### Model

| Export | Description |
|--------|-------------|
| `TFTConfig` | Configuration for Temporal Fusion Transformer |
| `TemporalFusionTransformer` | TFT model with modern components |

### Distributions

| Export | Description |
|--------|-------------|
| `GaussianHead` | Gaussian distribution (μ, σ) output head |
| `QuantileHead` | Direct quantile regression output head |

### Losses

| Export | Description |
|--------|-------------|
| `NLLLoss` | Negative log-likelihood loss |
| `QuantileLoss` | Quantile/pinball loss |

### Privacy

| Export | Description |
|--------|-------------|
| `RDPAccountant` | Rényi DP accountant with subsampling amplification |
| `dp_gradients` | Compute differentially private gradients |
| `compute_per_sample_gradients` | Per-example gradient computation |
| `clip_gradients` | Global L2 gradient clipping |
| `add_noise` | Add calibrated Gaussian noise |

### Metrics

| Export | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `smape` | Symmetric Mean Absolute Percentage Error |
| `mase` | Mean Absolute Scaled Error |
| `quantile_loss` | Pinball/quantile loss |
| `coverage` | Prediction interval coverage |

## Architecture

JAX-Prism's TFT implementation uses modern components:

- **RoPE** (Rotary Position Embedding): Better extrapolation, zero learnable parameters
- **RMSNorm**: Faster than LayerNorm, fewer parameters
- **SwiGLU**: Improved gradient flow and expressiveness
- **GQA**: Configurable attention (MHA default, supports MQA/GQA via `num_kv_heads`)

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical decisions and roadmap.

## Requirements

- Python 3.11+
- JAX >= 0.7.0
- Flax >= 0.10.0
- Optax >= 0.2.0

## License

Apache 2.0

## Citation

```bibtex
@software{jax_prism,
  title = {JAX-Prism: Privacy-Preserving Probabilistic Forecasting},
  author = {Guillaume Decina},
  year = {2026},
  url = {https://github.com/g-decina/jax-prism}
}
```

## References

- Abadi et al. "Deep Learning with Differential Privacy." CCS 2016.
- Mironov. "Rényi Differential Privacy." CSF 2017.
- Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." IJF 2021.
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
