"""Tests for cross-section engression fitting."""

import numpy as np
import pytest
import torch

from torch_endid.fitting import fit_engression_cs
from torch_endid.transforms import transform_panel_to_cross_section


class TestFitEngressionCS:

    def test_constant_effect_att(self, simple_panel):
        """On constant ATT=2 DGP, fitted ATT should be close to 2."""
        result = transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"], controls=result["controls"],
            num_epochs=300, noise_dim=5, hidden_dim=50,
            nsample=200, verbose=False, seed=42,
        )
        # ATT should be roughly 2.0 (within tolerance for small sample + NN)
        assert abs(fit["att"] - 2.0) < 1.5, f"ATT={fit['att']:.2f}, expected ~2.0"

    def test_returns_model(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"],
            num_epochs=50, verbose=False, seed=42,
        )
        from torch_engression import Engressor
        assert isinstance(fit["model"], Engressor)

    def test_qte_has_correct_quantiles(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )
        quantiles = [0.25, 0.5, 0.75]
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"],
            quantiles=quantiles,
            num_epochs=50, verbose=False, seed=42,
        )
        assert fit["qte"]["quantile"] == quantiles
        assert len(fit["qte"]["effect"]) == 3

    def test_samples_not_empty(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"],
            num_epochs=50, nsample=100, verbose=False, seed=42,
        )
        assert len(fit["samples_treated"]) > 0
        assert len(fit["samples_control"]) > 0

    def test_with_controls(self, simple_panel):
        df = simple_panel.copy()
        df["x1"] = df["unit"] * 0.1
        result = transform_panel_to_cross_section(
            data=df, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6,
            dvar="D", controls=["x1"],
        )
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"], controls=result["controls"],
            num_epochs=50, verbose=False, seed=42,
        )
        assert "att" in fit
        assert fit["model"] is not None

    @pytest.mark.gpu
    def test_gpu_device(self, simple_panel):
        """Test fitting on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU")
        result = transform_panel_to_cross_section(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", rolling="demean", tpost1=6, dvar="D",
        )
        fit = fit_engression_cs(
            Y=result["Y"], D=result["D"],
            num_epochs=50, device="cuda", verbose=False, seed=42,
        )
        assert fit["att"] is not None
