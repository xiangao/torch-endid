"""Tests for bootstrap inference."""

import numpy as np
import pytest
import torch

from torch_endid.bootstrap import bootstrap_endid
from torch_endid.transforms import transform_panel_to_cross_section


class TestBootstrap:

    def _get_cross_section(self, simple_panel):
        return transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )

    def test_sequential_bootstrap_returns_se(self, simple_panel):
        cs = self._get_cross_section(simple_panel)
        result = bootstrap_endid(
            Y=cs["Y"], D=cs["D"],
            nboot=10, num_epochs=50, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result["se"] > 0
        assert result["ci_lower"] < result["ci_upper"]
        assert len(result["att_boot"]) > 0

    def test_batched_bootstrap_returns_se(self, simple_panel):
        cs = self._get_cross_section(simple_panel)
        result = bootstrap_endid(
            Y=cs["Y"], D=cs["D"],
            nboot=8, num_epochs=50, nsample=50,
            batch_bootstrap=True, max_concurrent=4,
            verbose=False, seed=42,
        )
        assert result["se"] > 0
        assert len(result["att_boot"]) > 0

    def test_qte_bootstrap(self, simple_panel):
        cs = self._get_cross_section(simple_panel)
        quantiles = [0.25, 0.5, 0.75]
        result = bootstrap_endid(
            Y=cs["Y"], D=cs["D"],
            nboot=10, quantiles=quantiles,
            num_epochs=50, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert len(result["qte_mean"]) == 3
        assert len(result["qte_se"]) == 3
        assert len(result["qte_ci_lower"]) == 3

    def test_att_in_reasonable_range(self, simple_panel):
        """ATT should be positive (true ATT=2) for most bootstrap reps."""
        cs = self._get_cross_section(simple_panel)
        result = bootstrap_endid(
            Y=cs["Y"], D=cs["D"],
            nboot=10, num_epochs=100, nsample=100,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        # Most bootstrap ATTs should be positive
        frac_positive = (result["att_boot"] > 0).mean()
        assert frac_positive > 0.5, f"Only {frac_positive:.0%} positive ATTs"

    def test_cpu_fallback(self, simple_panel):
        """batch_bootstrap=True with device='cpu' falls back to sequential."""
        cs = self._get_cross_section(simple_panel)
        result = bootstrap_endid(
            Y=cs["Y"], D=cs["D"],
            nboot=5, num_epochs=50, nsample=50,
            batch_bootstrap=True, device="cpu",
            verbose=False, seed=42,
        )
        assert result["se"] > 0
