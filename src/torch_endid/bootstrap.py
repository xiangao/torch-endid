"""Bootstrap inference for endid — sequential and batched GPU."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from torch_engression import Engressor
from torch_engression.loss import energy_loss_two_sample
from .fitting import fit_engression_cs, _compute_effects


def bootstrap_endid(
    Y: torch.Tensor,
    D: torch.Tensor,
    controls: torch.Tensor | None = None,
    nboot: int = 200,
    quantiles: list[float] | None = None,
    nsample: int = 500,
    alpha: float = 0.05,
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
) -> dict:
    """Bootstrap inference for ATT and QTE.

    Resamples units with replacement, refits engression on each replicate,
    computes ATT and QTE. Returns SEs and percentile CIs.

    Args:
        Y: Residualized outcomes (n,).
        D: Treatment indicator (n,).
        controls: Control variables (n, p) or None.
        nboot: Number of bootstrap replicates.
        quantiles: Quantiles for QTE.
        nsample: MC samples for predictions.
        alpha: Significance level for CIs.
        noise_dim, hidden_dim, num_layer, num_epochs, lr: Engression params.
        device: Device for training.
        batch_bootstrap: If True, train multiple models concurrently on GPU.
        max_concurrent: Max concurrent models for batched bootstrap.
        seed: Random seed.
        verbose: Show progress bar.

    Returns:
        dict with: att_mean, se, ci_lower, ci_upper, att_boot,
        qte_mean, qte_se, qte_ci_lower, qte_ci_upper, qte_boot_mat.
    """
    if quantiles is None:
        quantiles = [round(q, 1) for q in np.arange(0.1, 1.0, 0.1).tolist()]

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    n = len(Y)
    nq = len(quantiles)

    if batch_bootstrap and device != "cpu" and torch.cuda.is_available():
        att_boot, qte_boot_mat = _batched_bootstrap(
            Y, D, controls, nboot, quantiles, nsample,
            noise_dim, hidden_dim, num_layer, num_epochs, lr,
            device, max_concurrent, verbose,
        )
    else:
        att_boot, qte_boot_mat = _sequential_bootstrap(
            Y, D, controls, nboot, quantiles, nsample,
            noise_dim, hidden_dim, num_layer, num_epochs, lr,
            device, verbose,
        )

    # Remove failed replicates
    valid = ~np.isnan(att_boot)
    att_valid = att_boot[valid]
    qte_valid = qte_boot_mat[valid]

    if len(att_valid) == 0:
        raise RuntimeError("All bootstrap replicates failed.")

    se = float(np.std(att_valid, ddof=1))
    ci_lower = float(np.quantile(att_valid, alpha / 2))
    ci_upper = float(np.quantile(att_valid, 1 - alpha / 2))
    att_mean = float(np.mean(att_valid))

    qte_mean = np.mean(qte_valid, axis=0).tolist()
    qte_se = np.std(qte_valid, axis=0, ddof=1).tolist()
    qte_ci_lower = np.quantile(qte_valid, alpha / 2, axis=0).tolist()
    qte_ci_upper = np.quantile(qte_valid, 1 - alpha / 2, axis=0).tolist()

    return {
        "att_mean": att_mean,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "att_boot": att_valid,
        "qte_mean": qte_mean,
        "qte_se": qte_se,
        "qte_ci_lower": qte_ci_lower,
        "qte_ci_upper": qte_ci_upper,
        "qte_boot_mat": qte_valid,
    }


def _one_replicate(
    Y, D, controls, quantiles, nsample,
    noise_dim, hidden_dim, num_layer, num_epochs, lr, device,
) -> tuple[float, list[float]]:
    """Run a single bootstrap replicate."""
    n = len(Y)
    idx = torch.randint(0, n, (n,))

    Y_b = Y[idx]
    D_b = D[idx]
    ctrl_b = controls[idx] if controls is not None else None

    # Need at least 2 treated and 2 control
    if (D_b == 1).sum() < 2 or (D_b == 0).sum() < 2:
        return float("nan"), [float("nan")] * len(quantiles)

    try:
        fit = fit_engression_cs(
            Y=Y_b, D=D_b, controls=ctrl_b,
            quantiles=quantiles, nsample=nsample,
            noise_dim=noise_dim, hidden_dim=hidden_dim,
            num_layer=num_layer, num_epochs=num_epochs,
            lr=lr, device=device, verbose=False,
        )
        return fit["att"], fit["qte"]["effect"]
    except Exception:
        return float("nan"), [float("nan")] * len(quantiles)


def _sequential_bootstrap(
    Y, D, controls, nboot, quantiles, nsample,
    noise_dim, hidden_dim, num_layer, num_epochs, lr,
    device, verbose,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential bootstrap — one model at a time."""
    att_boot = np.full(nboot, np.nan)
    qte_boot = np.full((nboot, len(quantiles)), np.nan)

    pbar = tqdm(range(nboot), desc="Bootstrap", disable=not verbose)
    for b in pbar:
        att_b, qte_b = _one_replicate(
            Y, D, controls, quantiles, nsample,
            noise_dim, hidden_dim, num_layer, num_epochs, lr, device,
        )
        att_boot[b] = att_b
        qte_boot[b] = qte_b
        if not np.isnan(att_b):
            pbar.set_postfix({"ATT_b": f"{att_b:.3f}"})

    return att_boot, qte_boot


def _batched_bootstrap(
    Y: torch.Tensor,
    D: torch.Tensor,
    controls: torch.Tensor | None,
    nboot: int,
    quantiles: list[float],
    nsample: int,
    noise_dim: int,
    hidden_dim: int,
    num_layer: int,
    num_epochs: int,
    lr: float,
    device: str | None,
    max_concurrent: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched GPU bootstrap — train multiple models concurrently.

    Partitions nboot into chunks of max_concurrent. Within each chunk,
    creates separate Engressor instances and trains them with interleaved
    epochs to maximize GPU utilization.
    """
    from torch_engression.utils import auto_device

    dev = auto_device(device)
    n = len(Y)
    nq = len(quantiles)

    att_boot = np.full(nboot, np.nan)
    qte_boot = np.full((nboot, nq), np.nan)

    n_chunks = (nboot + max_concurrent - 1) // max_concurrent
    pbar = tqdm(range(n_chunks), desc="Bootstrap (batched)", disable=not verbose)

    for chunk_idx in pbar:
        start = chunk_idx * max_concurrent
        end = min(start + max_concurrent, nboot)
        chunk_size = end - start

        # Create resampled datasets and models for this chunk
        models = []
        train_data = []  # stores (X_s, Y_s)
        orig_resampled_data = []  # stores (D_b, ctrl_b) for effect calculation
        optimizers = []
        valid_mask = [True] * chunk_size

        for j in range(chunk_size):
            idx = torch.randint(0, n, (n,))
            Y_b = Y[idx]
            D_b = D[idx]
            ctrl_b = controls[idx] if controls is not None else None

            if (D_b == 1).sum() < 2 or (D_b == 0).sum() < 2:
                valid_mask[j] = False
                models.append(None)
                train_data.append(None)
                orig_resampled_data.append(None)
                optimizers.append(None)
                continue

            # Build X
            D_col = D_b.unsqueeze(1)
            X_b = torch.cat([D_col, ctrl_b], dim=1) if ctrl_b is not None else D_col
            Y_col = Y_b.unsqueeze(1)

            in_dim = X_b.shape[1]
            out_dim = Y_col.shape[1]

            eng = Engressor(
                in_dim=in_dim, out_dim=out_dim,
                noise_dim=noise_dim, hidden_dim=hidden_dim,
                num_layer=num_layer, num_epochs=num_epochs,
                lr=lr, device=device, verbose=False,
            )

            # Standardize and move to device
            X_s, Y_s = eng._standardize_data_and_record_stats(X_b, Y_col)
            X_s = X_s.to(eng.device)
            Y_s = Y_s.to(eng.device)

            models.append(eng)
            train_data.append((X_s, Y_s))
            orig_resampled_data.append((D_b, ctrl_b))
            optimizers.append(eng.optimizer)

        # Interleaved training: all models share epochs
        use_amp = dev.type == "cuda"
        scalers = [
            torch.amp.GradScaler("cuda", enabled=use_amp) if valid_mask[j] else None
            for j in range(chunk_size)
        ]

        for epoch in range(num_epochs):
            for j in range(chunk_size):
                if not valid_mask[j]:
                    continue
                eng = models[j]
                X_s, Y_s = train_data[j]
                opt = optimizers[j]
                scaler = scalers[j]

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    y1 = eng.model(X_s)
                    y2 = eng.model(X_s)
                    loss, _, _ = energy_loss_two_sample(
                        Y_s, y1, y2, verbose=True
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        # Compute effects for each model
        for j in range(chunk_size):
            if not valid_mask[j]:
                continue
            eng = models[j]
            eng.model.eval()

            idx_global = start + j
            try:
                D_b, ctrl_b = orig_resampled_data[j]
                att_b, qte_dict, _, _ = _compute_effects(
                    eng, D_b, ctrl_b, quantiles, nsample
                )
                att_boot[idx_global] = att_b
                qte_boot[idx_global] = qte_dict["effect"]
            except Exception:
                pass

        # Free GPU memory
        del models, train_data, orig_resampled_data, optimizers, scalers
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        valid_so_far = ~np.isnan(att_boot[:end])
        if valid_so_far.any():
            pbar.set_postfix({
                "ATT_mean": f"{np.nanmean(att_boot[:end]):.3f}",
                "done": f"{end}/{nboot}",
            })

    return att_boot, qte_boot
