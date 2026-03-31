"""EndidResult — result container with summary, plotting, and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch


@dataclass
class CohortResult:
    """Per-cohort results for staggered designs."""
    cohort: int | float
    n_treated: int
    n_control: int
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    att_boot: np.ndarray
    qte: pd.DataFrame
    qte_boot_mat: np.ndarray


@dataclass
class EndidResult:
    """Result container for endid estimation.

    Contains ATT, QTE, bootstrap distributions, fitted model,
    and methods for summary, plotting, and serialization.
    """
    design: str  # "common_timing" or "staggered"
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    nboot: int

    qte: pd.DataFrame  # columns: quantile, effect, se, ci_lower, ci_upper
    att_boot: np.ndarray  # bootstrap ATT distribution

    # Staggered-specific
    cohort_results: dict[str, CohortResult] | None = None

    # Fitted model + data
    engression_model: object = None  # Engressor
    cross_section: pd.DataFrame = field(default_factory=pd.DataFrame)
    samples_treated: np.ndarray = field(default_factory=lambda: np.array([]))
    samples_control: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    rolling: str = "demean"
    controls: list[str] | None = None

    def summary(self) -> None:
        """Print summary of results."""
        print(f"endid: {self.design.replace('_', ' ')} design")
        print(f"  Rolling: {self.rolling}")
        print(f"  ATT: {self.att:.4f} (SE = {self.se:.4f})")
        print(f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        print(f"  Bootstrap: {self.nboot} replicates")

        if self.controls:
            print(f"  Controls: {', '.join(self.controls)}")

        n_treated = int((self.cross_section.get("_d_", pd.Series()) == 1).sum())
        n_control = int((self.cross_section.get("_d_", pd.Series()) == 0).sum())
        if n_treated > 0:
            print(f"  N treated: {n_treated}, N control: {n_control}")

        print(f"\nQuantile Treatment Effects:")
        print(self.qte.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if self.cohort_results:
            print(f"\nCohort-level results:")
            for name, cr in self.cohort_results.items():
                print(f"  Cohort {name}: ATT={cr.att:.4f} (SE={cr.se:.4f}), "
                      f"n_treated={cr.n_treated}, n_control={cr.n_control}")

    def plot_qte(self, ax=None) -> None:
        """Plot QTE with confidence intervals."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        q = self.qte["quantile"]
        eff = self.qte["effect"]

        ax.plot(q, eff, "o-", color="dodgerblue", label="QTE")
        ax.fill_between(
            q, self.qte["ci_lower"], self.qte["ci_upper"],
            alpha=0.2, color="dodgerblue", label="95% CI",
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=self.att, color="red", linestyle=":", alpha=0.7,
                   label=f"ATT = {self.att:.3f}")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Treatment Effect")
        ax.set_title("Quantile Treatment Effects")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_density(self, ax=None) -> None:
        """Plot counterfactual density comparison."""
        import matplotlib.pyplot as plt

        if len(self.samples_treated) == 0 or len(self.samples_control) == 0:
            print("No samples available for density plot.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(self.samples_treated, bins=50, density=True, alpha=0.5,
                color="dodgerblue", label="Treated (D=1)")
        ax.hist(self.samples_control, bins=50, density=True, alpha=0.5,
                color="coral", label="Control (D=0)")
        ax.set_xlabel("ydot_postavg")
        ax.set_ylabel("Density")
        ax.set_title("Counterfactual Distributions")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def save(self, path: str) -> None:
        """Save results to disk."""
        state = {
            "design": self.design,
            "att": self.att,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "nboot": self.nboot,
            "qte": self.qte.to_dict(),
            "att_boot": self.att_boot,
            "rolling": self.rolling,
            "controls": self.controls,
            "samples_treated": self.samples_treated,
            "samples_control": self.samples_control,
            "cross_section": self.cross_section.to_dict(),
        }
        # Save engression model separately if present
        if self.engression_model is not None:
            state["has_model"] = True
            model_path = path.replace(".pt", "_model.pt")
            self.engression_model.save(model_path)
        else:
            state["has_model"] = False

        # Save cohort results
        if self.cohort_results:
            state["cohort_results"] = {
                k: {
                    "cohort": cr.cohort,
                    "n_treated": cr.n_treated,
                    "n_control": cr.n_control,
                    "att": cr.att,
                    "se": cr.se,
                    "ci_lower": cr.ci_lower,
                    "ci_upper": cr.ci_upper,
                    "att_boot": cr.att_boot,
                    "qte": cr.qte.to_dict(),
                    "qte_boot_mat": cr.qte_boot_mat,
                }
                for k, cr in self.cohort_results.items()
            }

        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device=None) -> "EndidResult":
        """Load results from disk."""
        from torch_engression import Engressor

        state = torch.load(path, map_location="cpu", weights_only=False)

        model = None
        if state.get("has_model", False):
            model_path = path.replace(".pt", "_model.pt")
            model = Engressor.load(model_path, device=device)

        cohort_results = None
        if "cohort_results" in state and state["cohort_results"]:
            cohort_results = {
                k: CohortResult(
                    cohort=v["cohort"],
                    n_treated=v["n_treated"],
                    n_control=v["n_control"],
                    att=v["att"],
                    se=v["se"],
                    ci_lower=v["ci_lower"],
                    ci_upper=v["ci_upper"],
                    att_boot=v["att_boot"],
                    qte=pd.DataFrame(v["qte"]),
                    qte_boot_mat=v["qte_boot_mat"],
                )
                for k, v in state["cohort_results"].items()
            }

        return cls(
            design=state["design"],
            att=state["att"],
            se=state["se"],
            ci_lower=state["ci_lower"],
            ci_upper=state["ci_upper"],
            nboot=state["nboot"],
            qte=pd.DataFrame(state["qte"]),
            att_boot=state["att_boot"],
            engression_model=model,
            cross_section=pd.DataFrame(state["cross_section"]),
            samples_treated=state["samples_treated"],
            samples_control=state["samples_control"],
            rolling=state["rolling"],
            controls=state["controls"],
            cohort_results=cohort_results,
        )
