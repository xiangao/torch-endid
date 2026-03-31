"""Cross-section engression fitting: ATT and QTE computation."""

from __future__ import annotations

import numpy as np
import torch
from torch_engression import engression, Engressor


def fit_engression_cs(
    Y: torch.Tensor,
    D: torch.Tensor,
    controls: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    nsample: int = 500,
    noise_dim: int = 5,
    hidden_dim: int = 100,
    num_layer: int = 3,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
    seed: int | None = None,
) -> dict:
    """Fit engression on a DiD cross-section and compute treatment effects.

    Trains engression on Y ~ (D, controls), then computes ATT by comparing
    predictions at D=1 vs D=0 for each treated unit individually.

    Args:
        Y: Residualized outcomes (n,).
        D: Treatment indicator (n,), values 0/1.
        controls: Control variables (n, p) or None.
        quantiles: Quantiles for QTE. Default [0.1, ..., 0.9].
        nsample: MC samples for engression predictions.
        noise_dim: Engression noise dimension.
        hidden_dim: Engression hidden layer width.
        num_layer: Number of layers.
        num_epochs: Training epochs.
        lr: Learning rate.
        device: Device (None=auto, "cuda", "cpu").
        verbose: Show training progress.
        seed: Random seed.

    Returns:
        dict with: model, att, qte (DataFrame-like dict), samples_treated,
        samples_control.
    """
    if quantiles is None:
        quantiles = [round(q, 1) for q in np.arange(0.1, 1.0, 0.1).tolist()]

    # Build predictor matrix X = [D, controls]
    D_col = D.unsqueeze(1) if D.dim() == 1 else D
    if controls is not None:
        X = torch.cat([D_col, controls], dim=1)
    else:
        X = D_col

    Y_col = Y.unsqueeze(1) if Y.dim() == 1 else Y

    # Fit engression
    model = engression(
        X, Y_col,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        verbose=verbose,
        seed=seed,
    )

    # Compute ATT and QTE
    att, qte, samples_treated, samples_control = _compute_effects(
        model, D, controls, quantiles, nsample
    )

    return {
        "model": model,
        "att": att,
        "qte": qte,
        "samples_treated": samples_treated,
        "samples_control": samples_control,
    }


def _compute_effects(
    model: Engressor,
    D: torch.Tensor,
    controls: torch.Tensor | None,
    quantiles: list[float],
    nsample: int,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Compute ATT and QTE from a fitted engression model.

    For each treated unit, compares predictions at (D=1, controls_i) vs
    (D=0, controls_i), then averages. This respects Jensen's inequality
    for nonlinear models.

    Returns:
        (att, qte_dict, samples_treated, samples_control)
    """
    idx_treated = (D == 1).nonzero(as_tuple=True)[0]
    if len(idx_treated) == 0:
        raise ValueError("No treated units found.")

    # Build counterfactual X matrices for treated units
    if controls is not None:
        ctrl_treated = controls[idx_treated]
        X1 = torch.cat([torch.ones(len(idx_treated), 1), ctrl_treated], dim=1)
        X0 = torch.cat([torch.zeros(len(idx_treated), 1), ctrl_treated], dim=1)
    else:
        X1 = torch.ones(len(idx_treated), 1)
        X0 = torch.zeros(len(idx_treated), 1)

    # ATT: average difference in conditional means
    yhat1 = model.predict(X1, target="mean", sample_size=nsample)
    yhat0 = model.predict(X0, target="mean", sample_size=nsample)
    att = (yhat1 - yhat0).mean().item()

    # QTE: quantiles of pooled counterfactual samples
    s1 = model.sample(X1, sample_size=nsample)  # (n_treated, 1, nsample)
    s0 = model.sample(X0, sample_size=nsample)

    s1_pool = s1.cpu().numpy().flatten()
    s0_pool = s0.cpu().numpy().flatten()

    qte_effects = []
    for q in quantiles:
        q1 = np.quantile(s1_pool, q)
        q0 = np.quantile(s0_pool, q)
        qte_effects.append(q1 - q0)

    qte = {
        "quantile": quantiles,
        "effect": qte_effects,
    }

    return att, qte, s1_pool, s0_pool
