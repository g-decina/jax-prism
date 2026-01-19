# ARCHITECTURE.md — JAX-Prism

## Overview

JAX-Prism is a privacy-preserving probabilistic forecasting library built on JAX, Flax, and NumPyro. It combines differentially private training (DP-SGD) with Bayesian uncertainty quantification for time series forecasting.

**Positioning:** Technical depth that governance consultants lack, regulatory awareness that ML engineers lack. Flagship project for Lexicon Automata.

> JAX-Prism provides differentially private training with formal guarantees, Bayesian uncertainty quantification, and state-of-the-art neural architectures for time series forecasting.
>
> The package provides:
>
> - A **timeseries dataset class** handling variable transformations, missing values, subsampling, and multiple history lengths—with privacy-compatible normalization
> - A **functional training API** with optional OOP wrappers, supporting standard SGD, DP-SGD, and Bayesian inference (SGLD, SVI)
> - **Differential privacy primitives**: per-sample gradients, clipping, noise calibration, and composable privacy accountants (RDP, GDP, zCDP)
> - **Neural network architectures** (TFT, DeepAR, N-BEATS) modernized with current best practices and built-in interpretability
> - **Calibrated uncertainty quantification** combining aleatoric and epistemic uncertainty with diagnostic tools
> - **Multi-horizon probabilistic metrics**: CRPS, quantile loss, coverage, calibration diagrams
> - **Privacy-aware hyperparameter tuning** via differentially private selection mechanisms
> - **Auditing tools** for empirical privacy verification and governance framework integration

---

## Stack

- **JAX**: Autodiff and transformations
- **Flax**: Neural network modules
- **NumPyro**: Bayesian inference (v0.2.0+)
- **Optax**: Optimizers

---

## Core Design Principles

### Functional Core, OOP Shell

Pure functions for all computation; thin class wrappers for discoverability.

```python
# Functional core (power users)
grads, accountant, metrics = dp_gradients(loss_fn, params, batch, ...)

# OOP shell (convenience)
trainer = Trainer(model, dp_config=...)
trainer.fit(data)
```

### Protocols Over ABCs

Structural typing via `typing.Protocol`. Users can provide implementations without inheritance.

### Explicit State

No hidden mutation. Accountants, RNG keys, and optimizer state are threaded explicitly through all functions.

---

## Technical Decisions

### TFT Architecture (Modernized)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Position encoding | **RoPE** | Extrapolation to longer horizons, zero learnable params (DP efficient) |
| Normalization | **RMSNorm** | Faster, fewer params than LayerNorm |
| Activation | **SwiGLU** | Better gradient flow, more expressive |
| Attention | **MHA default, configurable** | `num_kv_heads` param enables MQA/GQA |

### DP Integration

| Approach | API | Use Case |
|----------|-----|----------|
| Option B | `dp_gradients()` | Power users, custom loops |
| Option C | `make_dp_train_step()` | Standard usage, returns jit-compiled step |
| OOP wrapper | `Trainer(dp_config=...)` | Convenience API |

### Privacy Accounting

- **v0.1.0**: RDP accountant with subsampling amplification
- **v0.5.0**: GDP, zCDP, numerical composition (PRV)

### Output Layer

- **Pluggable distributions**: Model outputs raw parameters; `DistributionHead` interprets them
- **Generic loss interface**: Loss functions take distribution as argument

### Normalization Strategy

- **Default**: Per-window normalization (last value or median of last k)
- **DP cost**: Zero (no learned statistics from training data)
- **Fallback**: User-provided fixed scale
- **v0.2.0**: Hierarchical scale inference via NumPyro
- **v0.3.0**: Optional DP quantile estimation for per-series normalization

### Missing Values

- **Masking**: Ignore missing timesteps in loss computation
- **No imputation in v0.1.0** (adds complexity, potential DP cost)

### Uncertainty Quantification

Two sources, combined in v0.2.0:

| Source | Type | Mechanism |
|--------|------|-----------|
| Output distribution | Aleatoric | Model outputs (μ, σ) or quantiles |
| Bayesian posterior | Epistemic | NumPyro inference over weights |

Combined via Monte Carlo:
```
p(y|x, data) = ∫ p(y|x, θ) p(θ|data) dθ
```

---

## Directory Structure

```
jax_prism/
├── __init__.py
├── _typing.py                  # Core type definitions
├── data/
│   ├── __init__.py
│   ├── batch.py                # TimeSeriesBatch
│   ├── scaling.py              # Normalizers (LastValue, Median, Fixed)
│   └── utils.py
├── distributions/
│   ├── __init__.py
│   ├── base.py                 # DistributionHead protocol
│   ├── gaussian.py
│   ├── quantile.py
│   ├── negative_binomial.py
│   └── student_t.py
├── losses/
│   ├── __init__.py
│   ├── base.py                 # Loss protocol
│   ├── nll.py
│   ├── quantile.py
│   └── point.py
├── metrics/
│   ├── __init__.py
│   ├── point.py
│   ├── probabilistic.py
│   └── calibration.py
├── models/
│   ├── __init__.py
│   ├── base.py                 # ForecastModel protocol
│   ├── tft/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── components.py       # GRN, VSN
│   │   ├── embeddings.py
│   │   └── config.py
│   └── components/
│       ├── __init__.py
│       ├── attention.py        # GQA (covers MHA, MQA)
│       ├── normalization.py    # RMSNorm
│       └── positional.py       # RoPE
├── privacy/
│   ├── __init__.py
│   ├── accountants/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── rdp.py
│   │   └── utils.py
│   ├── clipping.py
│   ├── noise.py
│   ├── gradients.py            # dp_gradients
│   └── training.py             # make_dp_train_step
├── inference/
│   ├── __init__.py
│   ├── sgd.py
│   ├── dp_sgd.py
│   └── optim.py
├── nn/
│   ├── __init__.py
│   ├── forecast_model.py       # OOP wrapper
│   └── trainer.py              # Trainer class
└── utils/
    ├── __init__.py
    ├── pytree.py
    ├── rng.py
    └── jax_utils.py

tests/
├── conftest.py
├── test_batch.py
├── test_distributions.py
├── test_rdp_accountant.py
├── test_clipping.py
├── test_attention.py
├── test_tft.py
└── ...
```

---

## Versioned Roadmap

### v0.1.0 — Foundation + TFT + DP-SGD ✓

**Goal:** Train a TFT with differential privacy on a standard dataset.

- [X] `_typing.py` — core types
- [X] `data/batch.py` — TimeSeriesBatch
- [X] `data/scaling.py` — last_value_scale, median_scale, fixed_scale, inverse_scale
- [X] `models/components/normalization.py` — RMSNorm
- [X] `models/components/positional.py` — RoPE
- [X] `models/components/attention.py` — GQA (configurable MHA/MQA)
- [X] `models/tft/components.py` — GRN, VSN, SwiGLU
- [X] `models/tft/model.py` — TFT
- [X] `models/tft/config.py` — TFTConfig
- [X] `distributions/gaussian.py` — GaussianHead
- [X] `distributions/quantile.py` — QuantileHead
- [X] `losses/nll.py` — NLLLoss
- [X] `losses/quantile.py` — QuantileLoss
- [X] `privacy/accountants/base.py` — Accountant protocol
- [X] `privacy/accountants/rdp.py` — RDP accountant
- [X] `privacy/clipping.py` — gradient clipping
- [X] `privacy/noise.py` — Gaussian mechanism
- [X] `privacy/gradients.py` — dp_gradients
- [X] `privacy/training.py` — make_dp_train_step, make_train_step
- [X] `metrics/point.py` — MAE, SMAPE, MASE
- [X] `metrics/probabilistic.py` — quantile loss, coverage
- [X] Integration test: TFT + DP-SGD on synthetic data
- [X] End-to-end demo: CTA ridership forecasting notebook

**Out of scope:** TimeSeriesDataset.from_dataframe, imputation, Bayesian inference, logging, tuning, other architectures.

### v0.1.1 — API Polish (Patch)

**Goal:** Fix friction points discovered in v0.1.0 demo notebook.

- [ ] `privacy/training.py` — thread dropout RNG key through make_train_step
- [ ] `distributions/gaussian.py` — ensure output shapes match target shapes (avoid squeeze dance)
- [ ] `data/scaling.py` — inverse_scale should handle 2D arrays gracefully
- [ ] `TFTConfig` — ensure __post_init__ runs correctly (num_kv_heads default)
- [ ] Documentation — docstrings for all public API functions

### v0.2.0 — Uncertainty Quantification

- Per-parameter output heads (`ParamHeadConfig`) for aleatoric uncertainty calibration
- SGLD training loop (non-DP)
- DP-SGLD (unified noise)
- Posterior predictive sampling
- Calibration metrics (reliability diagrams, PIT)
- CRPS metric
- Hierarchical scale inference

### v0.3.0 — Data Pipeline

- `TimeSeriesDataset.from_dataframe()`
- Automatic sliding window creation with configurable stride
- Automatic covariate detection (known vs observed)
- Train/val/test temporal splitting with proper leakage prevention
- Gap detection and warning for missing dates in time series
- DP-compatible normalization options
- `create_sliding_windows()` utility function

### v0.4.0 — Model Zoo

- DeepAR
- N-BEATS
- Simple baselines (naive, seasonal naive)

### v0.5.0 — Advanced Privacy

- NumPyro SVI integration
- GDP accountant
- zCDP accountant
- Adaptive clipping
- Private hyperparameter selection (report-noisy-max)

### v0.6.0 — Diagnostics & Auditing

- Membership inference attack suite
- Gradient norm tracking
- Privacy consumption curves
- spectrum-governance integration hooks

### v0.7.0 — Production Hardening

- Checkpointing (model + accountant state)
- Multi-device (pmap)
- Export for serving

### v0.8.0 — API Refinement

- `model.fit()` convenience API
- Callbacks (early stopping, model checkpoint, custom)
- Learning rate schedule integration (warmup, cosine decay, etc.)
- Progress logging utilities (loss curves, metrics tracking)
- Sensible defaults for common use cases

### v0.9.0 — Docs & Testing

- Sphinx API docs
- Tutorials
- >90% coverage

### v1.0.0 — Stable Release

---

## Key Interfaces

### TimeSeriesBatch

```python
@struct.dataclass
class TimeSeriesBatch:
    target: Array                          # (batch, time, targets)
    known_future: Array | None = None      # (batch, time, features)
    observed_past: Array | None = None     # (batch, encoder_time, features)
    static_categorical: Array | None = None
    static_real: Array | None = None
    mask: Array | None = None              # (batch, time)
```

### DistributionHead Protocol

```python
class DistributionHead(Protocol):
    num_params: int
    
    def __call__(self, params: Array) -> Distribution: ...
    def sample(self, params: Array, key: PRNGKey, shape: tuple) -> Array: ...
    def log_prob(self, params: Array, targets: Array) -> Array: ...
    def quantile(self, params: Array, q: Array) -> Array: ...
```

### Loss Protocol

```python
class Loss(Protocol):
    def __call__(
        self, 
        predictions: Array,
        targets: Array, 
        distribution: DistributionHead,
        mask: Array | None = None
    ) -> Array: ...
```

### dp_gradients (Option B)

```python
def dp_gradients(
    loss_fn: Callable,
    params: PyTree,
    batch: TimeSeriesBatch,
    clip_norm: float,
    noise_multiplier: float,
    accountant: RDPAccountant,
    key: PRNGKey,
    *,
    clipping_strategy: str = "global",
) -> tuple[PyTree, RDPAccountant, dict]: ...
```

### make_dp_train_step (Option C)

```python
def make_dp_train_step(
    model_apply: Callable,
    loss_fn: Loss,
    optimizer: optax.GradientTransformation,
    distribution: DistributionHead,
    clip_norm: float,
    noise_multiplier: float,
) -> Callable[[Params, OptState, Accountant, Batch, Key], 
              tuple[Params, OptState, Accountant, Metrics]]: ...
```

---

## User-Facing API (Target)

```python
import jax_prism as jp

# Data (v0.3.0)
dataset = jp.TimeSeriesDataset.from_dataframe(df, target="sales", ...)
train_loader = dataset.to_dataloader(batch_size=64)

# Model
model = jp.TFT(
    hidden_size=128,
    num_heads=4,
    output_distribution=jp.Gaussian(),
)

# Training with DP
trainer = jp.Trainer(
    model,
    optimizer="adam",
    lr=1e-3,
    dp_config=jp.DPConfig(
        clip_norm=1.0,
        noise_multiplier=1.1,
        target_epsilon=3.0,
        target_delta=1e-5,
    ),
)
result = trainer.fit(train_loader, epochs=50)
print(f"Final (ε, δ): ({result.epsilon:.2f}, {result.delta})")

# Bayesian (v0.2.0)
trainer = jp.BayesianTrainer(model, inference="sgld", dp_config=...)
posterior = trainer.fit(train_loader, num_samples=1000)

# Prediction & evaluation
predictions = model.predict(test_loader)
metrics = jp.evaluate(predictions, actuals, metrics=["mase", "crps"])
```

---

## References

- **DP-SGD**: Abadi et al. "Deep Learning with Differential Privacy." CCS 2016.
- **RDP**: Mironov. "Rényi Differential Privacy." CSF 2017.
- **Subsampling**: Wang, Balle, Kasiviswanathan. "Subsampled Rényi Differential Privacy." AISTATS 2019.
- **TFT**: Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." IJF 2021.
- **RoPE**: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
- **SwiGLU**: Shazeer. "GLU Variants Improve Transformer." 2020.
- **GDP**: Dong, Roth, Su. "Gaussian Differential Privacy." JMLR 2022.

---

## Decision Log

Record non-obvious choices here as development proceeds.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01 | RoPE over learned positional | Extrapolation + DP efficiency |
| 2025-01 | RMSNorm over LayerNorm | Faster, fewer params |
| 2025-01 | SwiGLU over ELU | Better expressiveness in modern transformers |
| 2025-01 | Per-window normalization default | Zero DP cost |
| 2025-01 | Masking for missing values | Simpler than imputation, no DP cost |
| 2025-01 | MHA default, configurable | Preserve interpretability, allow MQA for DP efficiency |
| 2025-01 | Functional scaling API | Functions (last_value_scale) over classes (LastValueScaler) for JAX idioms |
| 2025-01 | v0.1.0 validated on CTA ridership | End-to-end demo notebook confirms pipeline works |

## Lessons from v0.1.0 Demo

The CTA ridership forecasting notebook (`testkit/cta_ridership_demo.ipynb`) revealed:

1. **Data pipeline is biggest friction** — manual sliding windows, train/test split, scaling. Priority for v0.3.0.
2. **Dropout RNG threading** — `make_train_step` uses fixed RNG key; needs proper threading. Fix in v0.1.1.
3. **Shape inconsistencies** — distribution outputs (B, T) vs targets (B, T, 1) require manual squeeze. Fix in v0.1.1.
4. **TFTConfig post_init** — `num_kv_heads=None` default doesn't propagate correctly. Fix in v0.1.1.
5. **Calibration takes time** — 3 epochs gave 100% coverage (too wide); 20+ epochs with LR schedule gave 89% (better).
6. **Metrics work well** — MAE, SMAPE, coverage all functional and useful.