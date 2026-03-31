"""End-to-end tests for endid() common-timing path."""

import numpy as np
import pytest
import torch

from torch_endid import endid, EndidResult


class TestEndidCommonTiming:

    def test_basic_run(self, simple_panel):
        """endid() runs and returns EndidResult."""
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D", rolling="demean",
            num_epochs=100, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert isinstance(result, EndidResult)
        assert result.design == "common_timing"

    def test_att_positive(self, simple_panel):
        """ATT should be positive on constant-effect DGP (true ATT=2)."""
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D",
            num_epochs=200, nboot=10, nsample=100,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.att > 0, f"ATT={result.att:.2f}, expected positive"

    def test_se_positive(self, simple_panel):
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.se > 0

    def test_ci_contains_att(self, simple_panel):
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D",
            num_epochs=50, nboot=10, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        # CI should contain the point estimate
        assert result.ci_lower <= result.att <= result.ci_upper

    def test_qte_dataframe(self, simple_panel):
        quantiles = [0.25, 0.5, 0.75]
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D", quantiles=quantiles,
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert list(result.qte.columns) == ["quantile", "effect", "se", "ci_lower", "ci_upper"]
        assert len(result.qte) == 3

    def test_summary_runs(self, simple_panel, capsys):
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        result.summary()
        captured = capsys.readouterr()
        assert "ATT:" in captured.out
        assert "common timing" in captured.out

    def test_save_load_roundtrip(self, simple_panel, tmp_path):
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        path = str(tmp_path / "result.pt")
        result.save(path)
        loaded = EndidResult.load(path)
        assert abs(loaded.att - result.att) < 1e-6
        assert loaded.design == result.design
        assert len(loaded.qte) == len(result.qte)

    def test_detrend_method(self, simple_panel):
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D", rolling="detrend",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.design == "common_timing"

    def test_with_controls(self, simple_panel):
        df = simple_panel.copy()
        df["x1"] = df["unit"] * 0.1
        result = endid(
            data=df, y="y", ivar="unit", tvar="time",
            post="post", dvar="D", controls=["x1"],
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.controls == ["x1"]

    def test_invalid_rolling(self, simple_panel):
        with pytest.raises(ValueError, match="rolling must be"):
            endid(
                data=simple_panel, y="y", ivar="unit", tvar="time",
                post="post", rolling="invalid", verbose=False,
            )

    def test_missing_column(self, simple_panel):
        with pytest.raises(ValueError, match="not found"):
            endid(
                data=simple_panel, y="nonexistent", ivar="unit", tvar="time",
                post="post", verbose=False,
            )

    def test_no_post_or_gvar(self, simple_panel):
        with pytest.raises(ValueError, match="supply 'post'"):
            endid(
                data=simple_panel, y="y", ivar="unit", tvar="time",
                verbose=False,
            )

    @pytest.mark.gpu
    def test_gpu_endid(self, simple_panel):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU")
        result = endid(
            data=simple_panel, y="y", ivar="unit", tvar="time",
            post="post", dvar="D", device="cuda",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=True, max_concurrent=2,
            verbose=False, seed=42,
        )
        assert result.att is not None
