"""Shared fixtures for torch-endid tests."""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def simple_panel(seed):
    """Simple common-timing panel: 20 units, 10 periods, constant ATT=2.0.

    Units 0-9 are treated (post>=6), units 10-19 are never-treated.
    """
    n_units = 20
    n_periods = 10
    n_treated = 10
    tpost1 = 6

    rows = []
    for i in range(n_units):
        unit_fe = np.random.normal(0, 1)  # unit fixed effect
        for t in range(1, n_periods + 1):
            is_treated = i < n_treated
            is_post = t >= tpost1
            y = unit_fe + 0.5 * np.random.normal()
            if is_treated and is_post:
                y += 2.0  # constant treatment effect
            rows.append({
                "unit": i,
                "time": t,
                "y": y,
                "post": int(is_post),
                "D": int(is_treated),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def heterogeneous_panel(seed):
    """Panel with heterogeneous treatment effects for QTE testing.

    Effect is proportional to unit's baseline level:
    high-Y units get larger treatment effect.
    """
    n_units = 40
    n_periods = 10
    n_treated = 20
    tpost1 = 6

    rows = []
    for i in range(n_units):
        unit_fe = np.random.normal(0, 2)
        for t in range(1, n_periods + 1):
            is_treated = i < n_treated
            is_post = t >= tpost1
            y = unit_fe + 0.3 * np.random.normal()
            if is_treated and is_post:
                # Effect proportional to baseline (via unit FE)
                y += 1.0 + 0.5 * unit_fe
            rows.append({
                "unit": i,
                "time": t,
                "y": y,
                "post": int(is_post),
                "D": int(is_treated),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def staggered_panel(seed):
    """Staggered adoption panel: 30 units, 3 cohorts + never-treated.

    Cohort 2005: units 0-4 (treated from t=5)
    Cohort 2007: units 5-9 (treated from t=7)
    Cohort 2009: units 10-14 (treated from t=9)
    Never-treated: units 15-29
    """
    rows = []
    cohort_map = {}
    for i in range(30):
        if i < 5:
            cohort_map[i] = 5
        elif i < 10:
            cohort_map[i] = 7
        elif i < 15:
            cohort_map[i] = 9
        else:
            cohort_map[i] = float("inf")  # never-treated

    for i in range(30):
        unit_fe = np.random.normal(0, 1)
        g = cohort_map[i]
        for t in range(1, 11):
            is_post = t >= g if g != float("inf") else False
            y = unit_fe + 0.3 * np.random.normal()
            if is_post:
                y += 2.0
            rows.append({
                "unit": i,
                "time": t,
                "y": y,
                "gvar": g,
                "post": int(is_post),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def castle_data():
    """Load castle doctrine dataset."""
    from torch_endid.data.castle import load_castle
    return load_castle()
