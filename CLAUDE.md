# torch-endid

GPU-accelerated distributional difference-in-differences via engression.

## Project Structure

```
torch-endid/
├── src/torch_endid/
│   ├── __init__.py          # Package exports: endid(), EndidResult
│   ├── endid.py             # Main function, common-timing path, input validation
│   ├── staggered.py         # Staggered adoption path, cohort aggregation
│   ├── fitting.py           # fit_engression_cs(): engression on cross-section, ATT/QTE
│   ├── bootstrap.py         # Sequential and batched GPU bootstrap inference
│   ├── transforms.py        # Thin wrapper around lwdid.transformations
│   ├── results.py           # EndidResult, CohortResult dataclasses
│   └── data/
│       └── castle.py        # Castle Doctrine dataset loader
├── data/castle.csv          # Bundled test data (50 states, 2000-2010)
├── tests/                   # pytest suite (40 tests)
├── nb/benchmark.ipynb       # CPU vs GPU benchmark
└── pyproject.toml           # hatchling build
```

## Key Design Decisions

- **Depends on torch-engression**: Uses `torch_engression.engression()` for GPU-accelerated neural network training
- **Depends on lwdid**: Uses `lwdid.transformations.apply_rolling_transform()` for panel preprocessing
- **Batched GPU bootstrap**: Trains multiple bootstrap models concurrently on GPU (max_concurrent=4)
- **Per-unit ATT**: Evaluates E[Y|D=1,X_i] - E[Y|D=0,X_i] per treated unit, then averages (Jensen's inequality)
- **No D*X interactions**: Engression learns nonlinear relationships directly
- **Parameter names match R endid**: rolling, control_group, aggregate, gvar, post, etc.

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Development

```bash
pip install -e ".[dev]"
```

## Origins

- Algorithm from: R `endid` package at `~/projects/software/endid/`
- GPU patterns from: `torch-engression` at `~/projects/claude/torch-engression/`
- Panel transforms from: `lwdid` Python package
- This is a standalone package (not modifying the originals)
