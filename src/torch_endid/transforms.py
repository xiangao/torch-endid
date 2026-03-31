"""Panel data transforms — thin wrapper around lwdid.transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lwdid.transformations import apply_rolling_transform


def transform_panel_to_cross_section(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    post: str,
    rolling: str,
    tpost1: int,
    treated_units: list | set | None = None,
    dvar: str | None = None,
    controls: list[str] | None = None,
    season_var: str | None = None,
    Q: int = 4,
) -> dict:
    """Transform panel data to cross-section and extract tensors.

    Calls lwdid's apply_rolling_transform, then extracts the firstpost
    cross-section with outcome, treatment indicator, and controls as tensors.

    Args:
        data: Long-format panel DataFrame.
        y: Outcome column name.
        ivar: Unit identifier column name.
        tvar: Time column name.
        post: Binary post-treatment indicator column name.
        rolling: Transformation method (demean, detrend, demeanq, detrendq).
        tpost1: First post-treatment period.
        treated_units: Set of treated unit IDs. If None, inferred from dvar or post.
        dvar: Treatment group indicator column name.
        controls: List of control variable column names.
        season_var: Seasonal indicator column for demeanq/detrendq.
        Q: Number of seasonal periods (default 4).

    Returns:
        dict with keys:
            Y: torch.Tensor (n,) — residualized outcomes
            D: torch.Tensor (n,) — treatment indicators (0/1)
            controls: torch.Tensor (n, p) or None — control variables
            cross_section: pd.DataFrame — the firstpost cross-section
            transformed: pd.DataFrame — full transformed data
    """
    df = data.copy()

    # Apply lwdid transformation
    df_trans = apply_rolling_transform(
        data=df,
        y=y,
        ivar=ivar,
        tindex=tvar,
        post=post,
        rolling=rolling,
        tpost1=tpost1,
        season_var=season_var,
        Q=Q,
    )

    # Determine treatment indicator
    if treated_units is not None:
        df_trans["_d_"] = df_trans[ivar].isin(treated_units).astype(int)
    elif dvar is not None:
        df_trans["_d_"] = df_trans[dvar].astype(int)
    else:
        # Infer from post column: treated = units ever having post==1
        df_trans["_d_"] = df_trans[ivar].isin(
            df_trans.loc[df_trans[post] == 1, ivar].unique()
        ).astype(int)

    # Extract firstpost cross-section
    cs = df_trans[df_trans["firstpost"] == True].copy()  # noqa: E712

    # Drop units with missing outcome or controls
    keep_cols = ["ydot_postavg"]
    if controls:
        keep_cols.extend(controls)
    cs = cs.dropna(subset=keep_cols)

    if len(cs) == 0:
        raise ValueError("No valid observations in the firstpost cross-section.")

    # Extract tensors
    Y = torch.tensor(cs["ydot_postavg"].values, dtype=torch.float32)
    D = torch.tensor(cs["_d_"].values, dtype=torch.float32)

    ctrl_tensor = None
    if controls:
        ctrl_tensor = torch.tensor(
            cs[controls].values, dtype=torch.float32
        )

    return {
        "Y": Y,
        "D": D,
        "controls": ctrl_tensor,
        "cross_section": cs,
        "transformed": df_trans,
    }
