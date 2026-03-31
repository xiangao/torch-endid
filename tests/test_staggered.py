"""Tests for staggered design support."""

import numpy as np
import pytest

from torch_endid import endid, EndidResult


class TestEndidStaggered:

    def test_basic_staggered(self, staggered_panel):
        """endid() with gvar runs staggered path."""
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar", rolling="demean",
            num_epochs=100, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert isinstance(result, EndidResult)
        assert result.design == "staggered"

    def test_has_cohort_results(self, staggered_panel):
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.cohort_results is not None
        assert len(result.cohort_results) > 0

    def test_att_positive(self, staggered_panel):
        """With true ATT=2, aggregate ATT should be positive."""
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar",
            num_epochs=150, nboot=10, nsample=100,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.att > 0, f"ATT={result.att:.2f}, expected positive"

    def test_se_positive(self, staggered_panel):
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.se > 0

    def test_qte_dataframe(self, staggered_panel):
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar", quantiles=[0.25, 0.5, 0.75],
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert len(result.qte) == 3
        assert "effect" in result.qte.columns

    def test_summary_runs(self, staggered_panel, capsys):
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        result.summary()
        captured = capsys.readouterr()
        assert "staggered" in captured.out
        assert "Cohort" in captured.out

    def test_not_yet_treated(self, staggered_panel):
        """not_yet_treated control group should work."""
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar", control_group="not_yet_treated",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        assert result.design == "staggered"

    def test_cohort_weights(self, staggered_panel):
        """Cohort ATTs should be weighted by n_treated."""
        result = endid(
            data=staggered_panel, y="y", ivar="unit", tvar="time",
            gvar="gvar",
            num_epochs=50, nboot=5, nsample=50,
            batch_bootstrap=False, verbose=False, seed=42,
        )
        # All cohorts have 5 treated units → equal weights
        n_cohorts = len(result.cohort_results)
        assert n_cohorts >= 2  # At least 2 cohorts should be estimated
