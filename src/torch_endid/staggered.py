"""Staggered adoption DiD via engression."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .transforms import transform_panel_to_cross_section
from .fitting import fit_engression_cs
from .bootstrap import bootstrap_endid
from .results import EndidResult, CohortResult


def endid_staggered(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
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
    """Staggered adoption DiD via engression.

    For each treatment cohort, constructs a cross-section from cohort-treated
    units + control units, fits engression, and runs bootstrap inference.
    Aggregates across cohorts using n_treated weights.
    """
    if quantiles is None:
        quantiles = [round(q, 1) for q in np.arange(0.1, 1.0, 0.1).tolist()]

    if gvar not in data.columns:
        raise ValueError(f"Column '{gvar}' not found in data.")

    # Identify cohorts and never-treated
    gvals = data[gvar].copy()
    never_treated_mask = gvals.isna() | (gvals == 0) | np.isinf(gvals)
    never_treated_units = set(data.loc[never_treated_mask, ivar].unique())
    cohorts = sorted(data.loc[~never_treated_mask, gvar].unique())

    if len(cohorts) == 0:
        raise ValueError("No treatment cohorts found in gvar.")
    if len(never_treated_units) == 0 and control_group == "never_treated":
        raise ValueError("No never-treated units found. Use control_group='not_yet_treated'.")

    if verbose:
        print(f"Staggered design: {len(cohorts)} cohorts, {len(never_treated_units)} never-treated units")

    cohort_results = {}

    for g in cohorts:
        g_int = int(g)
        treated_units = set(data.loc[data[gvar] == g, ivar].unique())

        # Determine control units
        if control_group == "never_treated":
            control_units = never_treated_units
        elif control_group == "not_yet_treated":
            nyt_mask = (~never_treated_mask & (data[gvar] > g)) | never_treated_mask
            control_units = set(data.loc[nyt_mask, ivar].unique())
        else:
            raise ValueError("control_group must be 'never_treated' or 'not_yet_treated'.")

        if len(treated_units) < 2 or len(control_units) < 2:
            warnings.warn(
                f"Cohort {g_int}: skipping (n_treated={len(treated_units)}, "
                f"n_control={len(control_units)})."
            )
            continue

        # Subset data
        keep_units = treated_units | control_units
        df_g = data[data[ivar].isin(keep_units)].copy()

        # For not-yet-treated controls, remove their post-treatment observations
        if control_group == "not_yet_treated":
            nyt_units = control_units - never_treated_units
            if nyt_units:
                nyt_gvar_map = (
                    data.loc[data[ivar].isin(nyt_units), [ivar, gvar]]
                    .drop_duplicates()
                    .set_index(ivar)[gvar]
                    .to_dict()
                )
                drop_mask = df_g[ivar].isin(nyt_units) & df_g.apply(
                    lambda row: row[tvar] >= nyt_gvar_map.get(row[ivar], float("inf")),
                    axis=1,
                )
                df_g = df_g[~drop_mask]

        # Create post indicator for this cohort
        df_g["_post_g_"] = (df_g[tvar] >= g).astype(int)

        # Create treatment indicator
        df_g["_d_g_"] = df_g[ivar].isin(treated_units).astype(int)

        if verbose:
            print(f"\nCohort {g_int}: {len(treated_units)} treated, {len(control_units)} control")

        try:
            cs_result = transform_panel_to_cross_section(
                data=df_g, y=y, ivar=ivar, tvar=tvar,
                post="_post_g_", rolling=rolling, tpost1=g_int,
                treated_units=treated_units, controls=controls,
                season_var=season_var,
            )
        except Exception as e:
            warnings.warn(f"Cohort {g_int}: transform failed ({e}).")
            continue

        Y_cs = cs_result["Y"]
        D_cs = cs_result["D"]
        ctrl = cs_result["controls"]

        if (D_cs == 1).sum() < 2 or (D_cs == 0).sum() < 2:
            warnings.warn(f"Cohort {g_int}: degenerate cross-section.")
            continue

        # Fit engression
        try:
            fit_g = fit_engression_cs(
                Y=Y_cs, D=D_cs, controls=ctrl,
                quantiles=quantiles, nsample=nsample,
                noise_dim=noise_dim, hidden_dim=hidden_dim,
                num_layer=num_layer, num_epochs=num_epochs,
                lr=lr, device=device, verbose=verbose, seed=seed,
            )
        except Exception as e:
            warnings.warn(f"Cohort {g_int}: engression failed ({e}).")
            continue

        # Bootstrap
        try:
            boot_g = bootstrap_endid(
                Y=Y_cs, D=D_cs, controls=ctrl,
                nboot=nboot, quantiles=quantiles, nsample=nsample,
                noise_dim=noise_dim, hidden_dim=hidden_dim,
                num_layer=num_layer, num_epochs=num_epochs,
                lr=lr, device=device, batch_bootstrap=batch_bootstrap,
                max_concurrent=max_concurrent, verbose=verbose,
            )
        except Exception as e:
            warnings.warn(f"Cohort {g_int}: bootstrap failed ({e}).")
            continue

        qte_g = pd.DataFrame({
            "quantile": quantiles,
            "effect": boot_g["qte_mean"],
            "se": boot_g["qte_se"],
            "ci_lower": boot_g["qte_ci_lower"],
            "ci_upper": boot_g["qte_ci_upper"],
        })

        cohort_results[str(g_int)] = CohortResult(
            cohort=g_int,
            n_treated=len(treated_units),
            n_control=len(control_units),
            att=boot_g["att_mean"],
            se=boot_g["se"],
            ci_lower=boot_g["ci_lower"],
            ci_upper=boot_g["ci_upper"],
            att_boot=boot_g["att_boot"],
            qte=qte_g,
            qte_boot_mat=boot_g["qte_boot_mat"],
        )

        if verbose:
            print(f"  ATT = {boot_g['att_mean']:.4f} (SE = {boot_g['se']:.4f})")

    if len(cohort_results) == 0:
        raise RuntimeError("No cohorts could be estimated. Check data and parameters.")

    # Aggregate across cohorts
    agg = _aggregate_cohorts(cohort_results, quantiles, nboot)

    return EndidResult(
        design="staggered",
        att=agg["att"],
        se=agg["se"],
        ci_lower=agg["ci_lower"],
        ci_upper=agg["ci_upper"],
        nboot=nboot,
        qte=agg["qte"],
        att_boot=agg["att_boot"],
        cohort_results=cohort_results,
        rolling=rolling,
        controls=controls,
    )


def _aggregate_cohorts(
    cohort_results: dict[str, CohortResult],
    quantiles: list[float],
    nboot: int,
) -> dict:
    """Aggregate cohort-level results with n_treated weights."""
    n_treated = np.array([cr.n_treated for cr in cohort_results.values()])
    weights = n_treated / n_treated.sum()

    # Weighted ATT
    atts = np.array([cr.att for cr in cohort_results.values()])
    att_overall = float(np.sum(weights * atts))

    # Pool bootstrap draws
    boot_lens = [len(cr.att_boot) for cr in cohort_results.values()]
    B = min(boot_lens)

    att_boot_overall = np.zeros(B)
    qte_boot_overall = np.zeros((B, len(quantiles)))

    for i, cr in enumerate(cohort_results.values()):
        att_boot_overall += weights[i] * cr.att_boot[:B]
        if cr.qte_boot_mat is not None and len(cr.qte_boot_mat) >= B:
            qte_boot_overall += weights[i] * cr.qte_boot_mat[:B]

    se_overall = float(np.std(att_boot_overall, ddof=1))
    ci_lower = float(np.quantile(att_boot_overall, 0.025))
    ci_upper = float(np.quantile(att_boot_overall, 0.975))

    qte_mean = np.mean(qte_boot_overall, axis=0).tolist()
    qte_se = np.std(qte_boot_overall, axis=0, ddof=1).tolist()
    qte_ci_lower = np.quantile(qte_boot_overall, 0.025, axis=0).tolist()
    qte_ci_upper = np.quantile(qte_boot_overall, 0.975, axis=0).tolist()

    qte_df = pd.DataFrame({
        "quantile": quantiles,
        "effect": qte_mean,
        "se": qte_se,
        "ci_lower": qte_ci_lower,
        "ci_upper": qte_ci_upper,
    })

    return {
        "att": att_overall,
        "se": se_overall,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "att_boot": att_boot_overall,
        "qte": qte_df,
    }
