# Differential Privacy Guide

This guide explains how to train models with formal differential privacy guarantees using JAX-Prism.

## Table of Contents

- [What is Differential Privacy?](#what-is-differential-privacy)
- [DP-SGD Overview](#dp-sgd-overview)
- [Privacy Parameters](#privacy-parameters)
- [Using the RDP Accountant](#using-the-rdp-accountant)
- [Computing DP Gradients](#computing-dp-gradients)
- [Privacy Budget Management](#privacy-budget-management)
- [Best Practices](#best-practices)

---

## What is Differential Privacy?

Differential Privacy (DP) provides a mathematical guarantee that the output of a computation doesn't reveal too much about any individual in the input dataset. A mechanism M is (ε, δ)-differentially private if:

```
P[M(D) ∈ S] ≤ e^ε · P[M(D') ∈ S] + δ
```

for all datasets D and D' differing in one record, and all possible outputs S.

**Interpretation:**
- **ε (epsilon)**: Privacy loss. Lower = more private. Typical values: 1-10.
- **δ (delta)**: Probability of catastrophic failure. Should be < 1/n where n is dataset size. Typical: 10⁻⁵ to 10⁻⁷.

---

## DP-SGD Overview

DP-SGD (Differentially Private Stochastic Gradient Descent) makes training private by:

1. **Per-sample gradients**: Compute gradients for each example independently
2. **Gradient clipping**: Bound each example's gradient to limit influence
3. **Noise addition**: Add calibrated Gaussian noise to the aggregate
4. **Privacy accounting**: Track cumulative privacy expenditure

```
┌─────────────────────────────────────────────────────────────┐
│                       DP-SGD Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Batch          Per-Sample       Clipping        Noise      │
│   ─────  ───►   Gradients   ───►  (L2 norm)  ───►  (+)  ───► │
│                                                    ↑         │
│                                           N(0, σ²C²I)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Privacy Parameters

### Clip Norm (C)

The maximum L2 norm for each example's gradient.

```python
clip_norm = 1.0
```

**Tradeoffs:**
- **Too small**: Clips most gradients, slows learning
- **Too large**: Outliers dominate, need more noise

**Guidelines:**
- Start with median gradient norm from a few non-private steps
- Typical range: 0.1 to 10.0

### Noise Multiplier (σ)

Ratio of noise standard deviation to clip norm.

```python
noise_multiplier = 1.1  # σ = 1.1 * clip_norm
```

**Tradeoffs:**
- **Higher σ**: More privacy (lower ε), noisier gradients
- **Lower σ**: Less privacy (higher ε), cleaner gradients

**Guidelines:**
- Start with 1.0-1.5 for moderate privacy
- Use accountant to find σ for target ε

### Sample Rate (q)

Probability that each example is included in a batch.

```python
sample_rate = batch_size / dataset_size
```

**Privacy Amplification:**
Subsampling provides "free" privacy improvement. A mechanism with privacy loss ε becomes approximately O(q·ε) when applied to a random q-fraction.

---

## Using the RDP Accountant

JAX-Prism uses Rényi Differential Privacy (RDP) for tight composition.

### Creating an Accountant

```python
from jax_prism import RDPAccountant

accountant = RDPAccountant.create(
    noise_multiplier=1.1,
    sample_rate=0.01,
)
```

### Recording Steps

The accountant is **immutable**. Each `.step()` returns a new instance:

```python
# After each training step
accountant = accountant.step(
    noise_multiplier=1.1,
    sample_rate=0.01,
    num_steps=1,
)

# Or record multiple identical steps at once
accountant = accountant.step(
    noise_multiplier=1.1,
    sample_rate=0.01,
    num_steps=100,
)
```

### Checking Privacy Budget

```python
budget = accountant.get_privacy_spent(delta=1e-5)
print(f"After {accountant.num_steps} steps:")
print(f"  ε = {budget.epsilon:.2f}")
print(f"  δ = {budget.delta:.0e}")
```

### Example: Finding Noise for Target ε

```python
import jax_prism as jp

def find_noise_multiplier(target_epsilon, delta, sample_rate, num_steps):
    """Binary search for noise_multiplier achieving target epsilon."""
    low, high = 0.1, 100.0

    for _ in range(50):
        mid = (low + high) / 2
        accountant = jp.RDPAccountant.create(mid, sample_rate)
        accountant = accountant.step(mid, sample_rate, num_steps)
        budget = accountant.get_privacy_spent(delta)

        if budget.epsilon < target_epsilon:
            high = mid
        else:
            low = mid

    return mid

# Example: ε=3, δ=1e-5, 10K steps, 1% sample rate
sigma = find_noise_multiplier(
    target_epsilon=3.0,
    delta=1e-5,
    sample_rate=0.01,
    num_steps=10000,
)
print(f"Required noise multiplier: {sigma:.2f}")
```

---

## Computing DP Gradients

### Low-Level API

```python
import jax
import jax_prism as jp

def loss_fn(params, batch):
    output = model.apply(params, batch)
    return jp.NLLLoss(dist_head)(output, batch.future_targets)

# 1. Compute per-sample gradients
per_sample_grads = jp.compute_per_sample_gradients(
    loss_fn, params, batch
)

# 2. Clip each gradient
clipped_grads = jax.tree.map(
    lambda g: jp.clip_gradients(g, clip_norm),
    per_sample_grads,
)

# 3. Average
avg_grads = jax.tree.map(
    lambda g: g.mean(axis=0),
    clipped_grads,
)

# 4. Add noise
key = jax.random.key(42)
noisy_grads = jp.add_noise(
    avg_grads,
    noise_multiplier=noise_multiplier,
    clip_norm=clip_norm,
    key=key,
)
```

### High-Level API

```python
import jax_prism as jp

dp_grads = jp.dp_gradients(
    loss_fn=loss_fn,
    params=params,
    batch=batch,
    clip_norm=1.0,
    noise_multiplier=1.1,
    key=jax.random.key(42),
)
```

---

## Privacy Budget Management

### Total Budget

Privacy composes across training steps. After T steps:

```
ε_total ≈ O(√T · ε_step)  # with RDP composition
```

### Spending Strategy

For a fixed total budget ε_target:

1. **More epochs, more noise**: Train longer with higher σ
2. **Fewer epochs, less noise**: Train shorter with lower σ

Both achieve the same ε but have different utility tradeoffs.

### Example: Full Training Loop

```python
import jax
import optax
import jax_prism as jp

# Setup
model = jp.TemporalFusionTransformer(config)
optimizer = optax.adam(1e-3)
dist_head = jp.GaussianHead()

# Privacy config
clip_norm = 1.0
noise_multiplier = 1.1
delta = 1e-5
batch_size = 64
dataset_size = 10000
sample_rate = batch_size / dataset_size

# Initialize
key = jax.random.key(0)
params = model.init(key, sample_batch)
opt_state = optimizer.init(params)
accountant = jp.RDPAccountant.create(noise_multiplier, sample_rate)

def loss_fn(params, batch):
    output = model.apply(params, batch)
    return jp.NLLLoss(dist_head)(output, batch.future_targets)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        key, subkey = jax.random.split(key)

        # Compute DP gradients
        grads = jp.dp_gradients(
            loss_fn, params, batch,
            clip_norm, noise_multiplier, subkey
        )

        # Update model
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Update accountant
        accountant = accountant.step(noise_multiplier, sample_rate)

    # Check budget
    budget = accountant.get_privacy_spent(delta)
    print(f"Epoch {epoch}: ε={budget.epsilon:.2f}")

    # Optional: early stop if budget exceeded
    if budget.epsilon > target_epsilon:
        print("Privacy budget exhausted!")
        break
```

---

## Best Practices

### 1. Choose δ Appropriately

```python
delta = 1.0 / (10 * dataset_size)  # Standard choice
```

### 2. Tune Clip Norm Empirically

```python
# Run a few steps without noise to calibrate
for batch in sample_batches:
    grads = compute_gradients(params, batch)
    norm = compute_global_norm(grads)
    norms.append(norm)

clip_norm = jnp.median(jnp.array(norms))
```

### 3. Use Large Batch Sizes

Larger batches → lower sample rate → better privacy amplification:

```python
# Prefer batch_size=256 over batch_size=32
sample_rate_256 = 256 / 10000  # 0.0256
sample_rate_32 = 32 / 10000    # 0.0032
# The 32-batch needs ~8x more noise for same ε
```

### 4. Consider Virtual Batching

If memory-limited, accumulate gradients across microbatches:

```python
accumulated_grads = None
for microbatch in split_batch(batch, num_microbatches):
    micro_grads = compute_clipped_grads(params, microbatch)
    if accumulated_grads is None:
        accumulated_grads = micro_grads
    else:
        accumulated_grads = jax.tree.map(
            lambda a, b: a + b,
            accumulated_grads, micro_grads
        )

# Add noise once to the total
noisy_grads = add_noise(accumulated_grads, ...)
```

### 5. Pre-train on Public Data

If public data is available:

1. Pre-train on public data (no DP cost)
2. Fine-tune on private data with DP

This reduces the number of private steps needed.

### 6. Use Privacy-Efficient Architectures

JAX-Prism's TFT uses DP-efficient components:

- **RoPE**: Zero learnable parameters for positions
- **RMSNorm**: Fewer parameters than LayerNorm
- **Per-window normalization**: Zero DP cost (no learned statistics)

---

## Further Reading

- Abadi et al. "Deep Learning with Differential Privacy." CCS 2016.
- Mironov. "Rényi Differential Privacy." CSF 2017.
- Wang, Balle, Kasiviswanathan. "Subsampled Rényi Differential Privacy." AISTATS 2019.
- Google DP Library: https://github.com/google/differential-privacy
