"""Tests for panel transforms wrapper."""

import torch
from torch_endid.transforms import transform_panel_to_cross_section


class TestTransformPanelToCrossSection:

    def test_basic_demean(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
        )
        assert "Y" in result
        assert "D" in result
        assert "cross_section" in result
        assert result["Y"].shape[0] == 20  # 20 units
        assert result["D"].shape[0] == 20
        assert result["D"].sum() == 10  # 10 treated
        assert result["controls"] is None

    def test_cross_section_has_correct_columns(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
        )
        cs = result["cross_section"]
        assert "ydot" in cs.columns
        assert "ydot_postavg" in cs.columns
        assert "firstpost" in cs.columns

    def test_demean_removes_unit_mean(self, simple_panel):
        """Demeaning should make pre-period ydot average ≈ 0 for each unit."""
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
        )
        df_trans = result["transformed"]
        pre = df_trans[df_trans["post"] == 0]
        pre_means = pre.groupby("unit")["ydot"].mean()
        # Each unit's pre-period ydot should be ~0 (demeaned)
        assert pre_means.abs().max() < 1e-10

    def test_treated_ydot_postavg_positive(self, simple_panel):
        """With ATT=2, treated units' ydot_postavg should be positive."""
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
        )
        Y = result["Y"]
        D = result["D"]
        treated_mean = Y[D == 1].mean().item()
        control_mean = Y[D == 0].mean().item()
        # Treated should have higher ydot_postavg (ATT=2)
        assert treated_mean - control_mean > 1.0

    def test_detrend(self, simple_panel):
        """Detrending should also work."""
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="detrend",
            tpost1=6,
            dvar="D",
        )
        assert result["Y"].shape[0] == 20

    def test_with_controls(self, simple_panel):
        """Controls should be extracted as a tensor."""
        df = simple_panel.copy()
        df["x1"] = df["unit"] * 0.1  # time-invariant control
        result = transform_panel_to_cross_section(
            data=df,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
            controls=["x1"],
        )
        assert result["controls"] is not None
        assert result["controls"].shape == (20, 1)

    def test_tensors_are_float32(self, simple_panel):
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
            dvar="D",
        )
        assert result["Y"].dtype == torch.float32
        assert result["D"].dtype == torch.float32

    def test_infer_treated_from_post(self, simple_panel):
        """When no dvar/treated_units provided, infer from post column."""
        # This panel has D column but let's not pass it
        result = transform_panel_to_cross_section(
            data=simple_panel,
            y="y",
            ivar="unit",
            tvar="time",
            post="post",
            rolling="demean",
            tpost1=6,
        )
        # All units have post==1 in some periods (it's a calendar indicator),
        # so all 20 units should be "treated" under this inference
        # This tests the inference path runs without error
        assert result["Y"].shape[0] == 20
