# API Reference

Complete API documentation for JAX-Prism.

## Table of Contents

- [Data](#data)
- [Model](#model)
- [Distributions](#distributions)
- [Losses](#losses)
- [Privacy](#privacy)
- [Metrics](#metrics)
- [Protocols](#protocols)

---

## Data

### TimeSeriesBatch

```python
from jax_prism import TimeSeriesBatch
```

Immutable pytree-compatible batch structure for time series data.

**Attributes:**

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `past_targets` | `Array` | `(B, T_enc, F)` | Historical target values |
| `future_targets` | `Array \| None` | `(B, T_dec, F)` | Future target values (for training) |
| `past_observed_covariates` | `Array \| None` | `(B, T_enc, N)` | Past-only features (e.g., weather) |
| `past_known_covariates` | `Array \| None` | `(B, T_enc, M)` | Known features in past |
| `future_known_covariates` | `Array \| None` | `(B, T_dec, M)` | Known features in future (e.g., calendar) |
| `static_covariates` | `Array \| None` | `(B, S)` | Time-invariant features |
| `mask` | `Array \| None` | `(B, T)` | Validity mask (1=valid, 0=missing) |

**Example:**

```python
batch = TimeSeriesBatch(
    past_targets=jnp.ones((32, 168, 1)),
    future_targets=jnp.ones((32, 24, 1)),
    past_known_covariates=jnp.ones((32, 168, 3)),
    future_known_covariates=jnp.ones((32, 24, 3)),
)

# Access encoder/decoder inputs
encoder_inputs = batch.get_encoder_inputs()
decoder_inputs = batch.get_decoder_inputs()
```

---

### Scaling Functions

```python
from jax_prism import last_value_scale, median_scale, fixed_scale, inverse_scale
```

Privacy-compatible normalization (zero DP cost for per-window scaling).

#### last_value_scale

```python
def last_value_scale(x: Array) -> Tuple[Array, Array]:
    """Scale by absolute value of last timestep.

    Args:
        x: Input array, shape (..., T, F).

    Returns:
        scaled_x: Normalized array, same shape as x.
        scale: Scale factors, shape (..., F).
    """
```

#### median_scale

```python
def median_scale(x: Array, k: int | None = None) -> Tuple[Array, Array]:
    """Scale by median of last k timesteps.

    Args:
        x: Input array, shape (..., T, F).
        k: Number of timesteps. If None, uses all.

    Returns:
        scaled_x: Normalized array.
        scale: Scale factors.
    """
```

#### fixed_scale

```python
def fixed_scale(x: Array, scale: Array) -> Array:
    """Apply user-provided scale.

    Args:
        x: Input array, shape (..., T, F).
        scale: Scale factors, shape (..., F).

    Returns:
        Scaled array.
    """
```

#### inverse_scale

```python
def inverse_scale(scaled_x: Array, scale: Array) -> Array:
    """Reverse scaling (for predictions).

    Args:
        scaled_x: Normalized array.
        scale: Scale factors from original scaling.

    Returns:
        Original-scale array.
    """
```

---

## Model

### TFTConfig

```python
from jax_prism import TFTConfig
```

Configuration dataclass for Temporal Fusion Transformer.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | `64` | Hidden dimension (must be divisible by `num_heads`) |
| `num_heads` | `int` | `4` | Number of attention heads |
| `num_kv_heads` | `int \| None` | `None` | KV heads for GQA/MQA (defaults to `num_heads`) |
| `num_lstm_layers` | `int` | `2` | LSTM layers in encoder/decoder |
| `dropout_rate` | `float` | `0.1` | Dropout probability |
| `attention_dropout_rate` | `float` | `0.0` | Attention dropout |
| `encoder_length` | `int` | `168` | Historical timesteps |
| `decoder_length` | `int` | `24` | Forecast horizon |
| `num_static_features` | `int` | `0` | Static covariate count |
| `num_known_features` | `int` | `0` | Known covariate count |
| `num_observed_features` | `int` | `0` | Observed covariate count |
| `num_output_params` | `int` | `2` | Output parameters per timestep |

**Properties:**

- `head_dim`: Dimension per attention head
- `total_sequence_length`: `encoder_length + decoder_length`
- `num_encoder_features`: Total encoder input features
- `num_decoder_features`: Total decoder input features

**Attention Modes:**

```python
# Multi-Head Attention (default)
config = TFTConfig(num_heads=8)

# Grouped Query Attention
config = TFTConfig(num_heads=8, num_kv_heads=2)

# Multi-Query Attention
config = TFTConfig(num_heads=8, num_kv_heads=1)
```

---

### TemporalFusionTransformer

```python
from jax_prism import TemporalFusionTransformer
```

Temporal Fusion Transformer with modern components.

**Components:**
- RoPE positional encoding
- RMSNorm normalization
- SwiGLU activations
- GQA attention (configurable)
- GRN gating
- VSN variable selection

**Example:**

```python
config = TFTConfig(
    hidden_size=128,
    num_heads=8,
    encoder_length=168,
    decoder_length=24,
    num_known_features=5,
    num_output_params=2,
)
model = TemporalFusionTransformer(config)

# Initialize
params = model.init(key, batch)

# Forward pass (inference)
output = model.apply(params, batch, training=False)

# Forward pass (training with dropout)
output = model.apply(
    params, batch, training=True,
    rngs={"dropout": dropout_key}
)
```

---

## Distributions

### GaussianHead

```python
from jax_prism import GaussianHead
```

Gaussian distribution output head.

**Properties:**
- `num_params = 2` (μ, σ)

**Methods:**

```python
head = GaussianHead()

# Transform raw output to distribution parameters
params = head.params_from_raw(raw_output)
# Returns: {"loc": μ, "scale": σ}

# Compute log probability
log_prob = head.log_prob(params, targets)

# Sample from distribution
samples = head.sample(params, key, sample_shape=(100,))
```

---

### QuantileHead

```python
from jax_prism import QuantileHead
```

Direct quantile regression output head.

**Constructor:**

```python
head = QuantileHead(quantiles=[0.1, 0.5, 0.9])
```

**Properties:**
- `num_params`: Number of quantiles

---

## Losses

### NLLLoss

```python
from jax_prism import NLLLoss
```

Negative log-likelihood loss for distribution heads.

**Example:**

```python
loss_fn = NLLLoss(distribution_head=GaussianHead())
loss = loss_fn(predictions, targets, mask=None)
```

---

### QuantileLoss

```python
from jax_prism import QuantileLoss
```

Quantile/pinball loss for direct quantile regression.

**Example:**

```python
loss_fn = QuantileLoss(quantiles=jnp.array([0.1, 0.5, 0.9]))
loss = loss_fn(predictions, targets, mask=None)
```

---

## Privacy

### RDPAccountant

```python
from jax_prism import RDPAccountant
```

Rényi Differential Privacy accountant with subsampling amplification.

**Class Methods:**

```python
# Create new accountant
accountant = RDPAccountant.create(
    noise_multiplier=1.1,
    sample_rate=0.01,
    orders=None,  # Uses default RDP orders
)
```

**Instance Methods:**

```python
# Record training step(s)
accountant = accountant.step(
    noise_multiplier=1.1,
    sample_rate=0.01,
    num_steps=1,
)

# Get privacy budget
budget = accountant.get_privacy_spent(delta=1e-5)
print(f"ε={budget.epsilon}, δ={budget.delta}")
```

**Properties:**
- `num_steps`: Total steps recorded
- `rdp`: Current RDP values at each order

---

### dp_gradients

```python
from jax_prism import dp_gradients
```

Compute differentially private gradients.

```python
def dp_gradients(
    loss_fn: Callable[[Params, Batch], Array],
    params: Params,
    batch: TimeSeriesBatch,
    clip_norm: float,
    noise_multiplier: float,
    key: PRNGKey,
) -> PyTree:
    """Compute DP gradients with clipping and noise.

    Args:
        loss_fn: Per-example loss function.
        params: Model parameters.
        batch: Input batch.
        clip_norm: L2 norm bound for clipping.
        noise_multiplier: σ/clip_norm ratio.
        key: PRNG key for noise.

    Returns:
        Noisy clipped gradients.
    """
```

---

### compute_per_sample_gradients

```python
from jax_prism import compute_per_sample_gradients
```

Compute gradients for each example in a batch.

```python
def compute_per_sample_gradients(
    loss_fn: Callable[[Params, Array], Array],
    params: Params,
    inputs: Array,
) -> PyTree:
    """Compute per-example gradients using vmap.

    Returns:
        Gradients with leading batch dimension.
    """
```

---

### clip_gradients

```python
from jax_prism import clip_gradients
```

Global L2 gradient clipping.

```python
def clip_gradients(
    gradients: PyTree,
    max_norm: float,
) -> PyTree:
    """Clip gradient global norm to max_norm."""
```

---

### add_noise

```python
from jax_prism import add_noise
```

Add calibrated Gaussian noise for differential privacy.

```python
def add_noise(
    gradients: PyTree,
    noise_multiplier: float,
    clip_norm: float,
    key: PRNGKey,
) -> PyTree:
    """Add Gaussian noise with std = noise_multiplier * clip_norm."""
```

---

## Metrics

### Point Metrics

```python
from jax_prism import mae, smape, mase
```

#### mae

```python
def mae(
    y_true: Array,
    y_pred: Array,
    mask: Array | None = None,
) -> Array:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth, shape (B, T, F).
        y_pred: Predictions, same shape.
        mask: Optional validity mask.

    Returns:
        Scalar MAE.
    """
```

#### smape

```python
def smape(
    y_true: Array,
    y_pred: Array,
    mask: Array | None = None,
) -> Array:
    """Symmetric Mean Absolute Percentage Error.

    Returns value in [0, 200].
    """
```

#### mase

```python
def mase(
    y_true: Array,
    y_pred: Array,
    y_train: Array,
    seasonality: int = 1,
    mask: Array | None = None,
) -> Array:
    """Mean Absolute Scaled Error.

    Args:
        y_true: Ground truth for evaluation.
        y_pred: Predictions.
        y_train: Training data for naive forecast error.
        seasonality: Seasonal period (1 = random walk).

    Returns:
        MASE. Values < 1 beat naive forecast.
    """
```

---

### Probabilistic Metrics

```python
from jax_prism import quantile_loss, coverage
```

#### quantile_loss

```python
def quantile_loss(
    y_true: Array,
    y_pred: Array,
    quantiles: Array,
    mask: Array | None = None,
) -> Array:
    """Pinball/quantile loss.

    Args:
        y_true: Ground truth, shape (B, T, F).
        y_pred: Quantile predictions, shape (B, T, F, Q).
        quantiles: Quantile levels, shape (Q,).

    Returns:
        Scalar loss.
    """
```

#### coverage

```python
def coverage(
    y_true: Array,
    lower: Array,
    upper: Array,
    mask: Array | None = None,
) -> Array:
    """Prediction interval coverage rate.

    Args:
        y_true: Ground truth.
        lower: Lower bound of interval.
        upper: Upper bound of interval.

    Returns:
        Coverage rate in [0, 1].
    """
```

---

## Protocols

JAX-Prism uses `typing.Protocol` for structural typing. Implement these interfaces without inheritance.

### DistributionHead

```python
from jax_prism._typing import DistributionHead

class CustomHead:
    @property
    def num_params(self) -> int: ...

    def params_from_raw(self, raw: Array) -> Dict[str, Array]: ...
    def log_prob(self, params: Dict[str, Array], targets: Array) -> Array: ...
    def sample(self, params: Dict[str, Array], key: PRNGKey, shape: tuple) -> Array: ...
```

### Loss

```python
from jax_prism._typing import Loss

class CustomLoss:
    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None,
    ) -> Array: ...
```

### PrivacyAccountant

```python
from jax_prism._typing import PrivacyAccountant

class CustomAccountant:
    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> "CustomAccountant": ...
    def get_privacy_spent(self, delta: float) -> PrivacyBudget: ...
```

### ForecastModel

```python
from jax_prism._typing import ForecastModel

class CustomModel:
    def __call__(self, inputs: Array, training: bool = False) -> Array: ...
```
