# torch-endid

[![docs](https://img.shields.io/badge/docs-site-blue.svg)](https://xiangao.github.io/torch-endid/)

GPU-accelerated distributional difference-in-differences via engression.

Combines [lwdid](https://github.com/xwshen51/lwdid) panel transformations (Lee & Wooldridge, 2025) with [torch-engression](../torch-engression/) GPU-accelerated distributional regression to produce ATT, quantile treatment effects (QTE), and counterfactual distributions from panel data.

## Installation

```bash
pip install torch-endid
```

Or from source:

```bash
git clone https://github.com/xiangao/torch-endid
cd torch-endid
pip install -e .
```

Requires PyTorch >= 2.0, torch-engression >= 0.1.0, and lwdid >= 0.2.3.

## Quick Start

```python
import pandas as pd
from torch_endid import endid

# Load panel data
data = pd.read_csv("panel_data.csv")

# Common-timing DiD with distributional effects
result = endid(
    data, y="outcome", ivar="unit", tvar="time",
    post="post", dvar="treated",
    rolling="demean",
    num_epochs=500, nboot=200,
)

# Point estimate
print(f"ATT = {result.att:.4f} (SE = {result.se:.4f})")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

# Quantile treatment effects
print(result.qte)

# Visualization
result.plot_qte()
result.plot_density()

# Summary
result.summary()

# Save / load
result.save("result.pt")
from torch_endid import EndidResult
result = EndidResult.load("result.pt")
```

## Staggered Adoption

```python
# Staggered DiD with treatment cohorts
result = endid(
    data, y="outcome", ivar="unit", tvar="time",
    gvar="first_treatment_year",    # NA/0/Inf = never-treated
    rolling="demean",
    control_group="never_treated",  # or "not_yet_treated"
    num_epochs=500, nboot=200,
)

# Per-cohort results
for name, cr in result.cohort_results.items():
    print(f"Cohort {name}: ATT={cr.att:.4f} (SE={cr.se:.4f})")
```

## GPU Acceleration

Auto-detects GPU. GPU speedup scales with cross-section size:

| N units | CPU (s) | GPU (s) | Speedup |
|---------|---------|---------|---------|
| 100     | 7.1     | 28.3    | 0.3x    |
| 1,000   | 11.6    | 28.1    | 0.4x    |
| 5,000   | 22.7    | 20.2    | 1.1x    |
| 10,000  | 43.5    | 28.5    | **1.5x** |

(nboot=5-10, num_epochs=200, GTX 1080 Ti)

GPU overhead dominates for small cross-sections (<1K units) typical of DiD. For large panels (5K+ units), GPU becomes beneficial. CPU is recommended for most empirical applications.

```python
# Auto-detect GPU (default)
result = endid(data, y="y", ivar="id", tvar="t", post="post", dvar="D")

# Explicit device
result = endid(..., device="cuda")
result = endid(..., device="cpu")

# Control bootstrap batching
result = endid(..., batch_bootstrap=True, max_concurrent=4)
```

## Documentation & examples

Full documentation: **<https://xiangao.github.io/torch-endid/>**

| Page | Description |
|------|-------------|
| [Benchmark notebook](https://github.com/xiangao/torch-endid/blob/main/nb/benchmark.ipynb) | End-to-end runtime benchmark |
| [Examples page](https://xiangao.github.io/torch-endid/examples/) | Notebook and generated benchmark figure links |

## How It Works

1. **Panel → cross-section**: lwdid's `apply_rolling_transform()` residualizes panel outcomes using unit-specific pre-treatment observations (demean, detrend, demeanq, or detrendq)
2. **Fit engression**: `torch_engression.engression()` learns the conditional distribution P(ydot | D, controls) on GPU
3. **ATT**: For each treated unit, compare E[Y|D=1, controls_i] vs E[Y|D=0, controls_i], then average
4. **QTE**: Pool Monte Carlo samples under D=1 and D=0, compute quantile differences
5. **Bootstrap**: Resample units, refit engression, compute ATT+QTE per replicate → SEs and CIs

## API Reference

### `endid(data, y, ivar, tvar, **kwargs)`

Main function. Dispatches to common-timing or staggered path.

**Key parameters:**
- `gvar` (str): First-treatment-year column → staggered design
- `post` (str): Binary post indicator → common-timing design
- `dvar` (str): Treatment group indicator
- `rolling` (str): "demean" (default), "detrend", "demeanq", "detrendq"
- `controls` (list[str]): Control variable column names
- `quantiles` (list[float]): Quantiles for QTE (default: [0.1, ..., 0.9])
- `nsample` (int): MC samples for predictions (default: 500)
- `nboot` (int): Bootstrap replicates (default: 200)
- `noise_dim` (int): Engression noise dimension (default: 5)
- `hidden_dim` (int): Hidden layer width (default: 100)
- `num_layer` (int): Number of layers (default: 3)
- `num_epochs` (int): Training epochs (default: 1000)
- `lr` (float): Learning rate (default: 1e-3)
- `device` (str): "cpu", "cuda", or None (auto)
- `batch_bootstrap` (bool): Batch bootstrap on GPU (default: True)
- `max_concurrent` (int): Max concurrent bootstrap models (default: 4)

### `EndidResult`

- `.att`, `.se`, `.ci_lower`, `.ci_upper` — ATT with inference
- `.qte` — DataFrame with quantile, effect, se, ci_lower, ci_upper
- `.att_boot` — Bootstrap ATT distribution
- `.cohort_results` — Per-cohort results (staggered only)
- `.summary()` — Print results
- `.plot_qte()` — QTE with CI band
- `.plot_density()` — Counterfactual density comparison
- `.save(path)` / `EndidResult.load(path)` — Serialization

## References

- Lee, Y. & Wooldridge, J. M. (2025). A simple panel data approach to difference-in-differences under general treatment patterns.
- Shen, X. & Meinshausen, N. (2024). Engression: Extrapolation through the Lens of Distributional Regression. *JMLR*.
- Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods. *JoE*.
