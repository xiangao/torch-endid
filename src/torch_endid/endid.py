"""Main endid() function — common-timing and dispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import transform_panel_to_cross_section
from .fitting import fit_engression_cs
from .bootstrap import bootstrap_endid
from .results import EndidResult


def endid(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    post: str | None = None,
    dvar: str | None = None,
    rolling: str = "demean",
    control_group: str = "never_treated",
    aggregate: str = "overall",
    controls: list[str] | None = None,
    season_var: str | None = None,
    quantiles: list[float] | None = None,
    nsample: int = 500,
    nboot: int = 200,
    noise_dim: int = 5,
    hidden_dim: int = 100,
    num_layer: int = 3,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    device: str | None = None,
    batch_bootstrap: bool = True,
    max_concurrent: int = 4,
    seed: int | None = None,
    verbose: bool = True,
) -> EndidResult:
    """GPU-accelerated distributional difference-in-differences.

    Combines lwdid panel transformations with torch-engression distributional
    regression to produce ATT, quantile treatment effects (QTE), and
    counterfactual distributions.

    Args:
        data: Long-format panel DataFrame.
        y: Outcome column name.
        ivar: Unit identifier column name.
        tvar: Time column name (numeric).
        gvar: First-treatment-year column for staggered designs.
            Units with NA, 0, or Inf are never-treated.
        post: Binary post-treatment indicator column (common-timing).
        dvar: Treatment group indicator column.
        rolling: Transformation method (demean, detrend, demeanq, detrendq).
        control_group: Control group for staggered designs
            ("never_treated" or "not_yet_treated").
        aggregate: Aggregation for staggered ("overall", "cohort", "none").
        controls: List of control variable column names.
        season_var: Seasonal indicator column for demeanq/detrendq.
        quantiles: Quantiles for QTE. Default [0.1, ..., 0.9].
        nsample: MC samples for engression predictions.
        nboot: Number of bootstrap replicates.
        noise_dim: Engression noise dimension.
        hidden_dim: Engression hidden layer width.
        num_layer: Number of layers.
        num_epochs: Training epochs.
        lr: Learning rate.
        device: Device (None=auto, "cuda", "cpu").
        batch_bootstrap: Batch multiple bootstrap fits on GPU.
        max_concurrent: Max concurrent models for batched bootstrap.
        seed: Random seed.
        verbose: Show progress.

    Returns:
        EndidResult with ATT, QTE, CIs, fitted model, and samples.
    """
    if quantiles is None:
        quantiles = [round(q, 1) for q in np.arange(0.1, 1.0, 0.1).tolist()]

    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    for col in [y, ivar, tvar]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if rolling not in ("demean", "detrend", "demeanq", "detrendq"):
        raise ValueError(f"rolling must be demean/detrend/demeanq/detrendq, got '{rolling}'")

    # Dispatch to staggered if gvar provided
    if gvar is not None:
        from .staggered import endid_staggered
        return endid_staggered(
            data=data, y=y, ivar=ivar, tvar=tvar, gvar=gvar,
            rolling=rolling, control_group=control_group,
            aggregate=aggregate, controls=controls,
            season_var=season_var, quantiles=quantiles,
            nsample=nsample, nboot=nboot,
            noise_dim=noise_dim, hidden_dim=hidden_dim,
            num_layer=num_layer, num_epochs=num_epochs,
            lr=lr, device=device, batch_bootstrap=batch_bootstrap,
            max_concurrent=max_concurrent, seed=seed, verbose=verbose,
        )

    # --- Common-timing path ---
    if post is None:
        raise ValueError("For common-timing design, supply 'post' column name.")
    if post not in data.columns:
        raise ValueError(f"Column '{post}' not found in data.")

    # Determine first post period
    post_periods = sorted(data.loc[data[post] == 1, tvar].unique())
    if len(post_periods) == 0:
        raise ValueError("No post-treatment periods found (no rows with post==1).")
    tpost1 = int(post_periods[0])

    # Determine treated units
    treated_units = None
    if dvar is not None:
        if dvar not in data.columns:
            raise ValueError(f"Column '{dvar}' not found in data.")
        treated_units = set(data.loc[data[dvar] == 1, ivar].unique())

    # Transform panel to cross-section
    cs_result = transform_panel_to_cross_section(
        data=data, y=y, ivar=ivar, tvar=tvar,
        post=post, rolling=rolling, tpost1=tpost1,
        treated_units=treated_units, dvar=dvar,
        controls=controls, season_var=season_var,
    )

    Y = cs_result["Y"]
    D = cs_result["D"]
    ctrl = cs_result["controls"]

    n_treated = int((D == 1).sum())
    n_control = int((D == 0).sum())
    if verbose:
        print(f"Cross-section: {len(D)} units ({n_treated} treated, {n_control} control)")

    # Fit engression on cross-section
    if verbose:
        print("Fitting engression model...")
    fit = fit_engression_cs(
        Y=Y, D=D, controls=ctrl,
        quantiles=quantiles, nsample=nsample,
        noise_dim=noise_dim, hidden_dim=hidden_dim,
        num_layer=num_layer, num_epochs=num_epochs,
        lr=lr, device=device, verbose=verbose, seed=seed,
    )

    # Bootstrap inference
    if verbose:
        print(f"Running {nboot} bootstrap replicates...")
    boot = bootstrap_endid(
        Y=Y, D=D, controls=ctrl,
        nboot=nboot, quantiles=quantiles, nsample=nsample,
        noise_dim=noise_dim, hidden_dim=hidden_dim,
        num_layer=num_layer, num_epochs=num_epochs,
        lr=lr, device=device, batch_bootstrap=batch_bootstrap,
        max_concurrent=max_concurrent, seed=seed, verbose=verbose,
    )

    # Build QTE DataFrame
    qte_df = pd.DataFrame({
        "quantile": quantiles,
        "effect": boot["qte_mean"],
        "se": boot["qte_se"],
        "ci_lower": boot["qte_ci_lower"],
        "ci_upper": boot["qte_ci_upper"],
    })

    return EndidResult(
        design="common_timing",
        att=boot["att_mean"],
        se=boot["se"],
        ci_lower=boot["ci_lower"],
        ci_upper=boot["ci_upper"],
        nboot=nboot,
        qte=qte_df,
        att_boot=boot["att_boot"],
        engression_model=fit["model"],
        cross_section=cs_result["cross_section"],
        samples_treated=fit["samples_treated"],
        samples_control=fit["samples_control"],
        rolling=rolling,
        controls=controls,
    )
