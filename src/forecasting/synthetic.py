"""
Synthetic Phi trajectory generator — 8 welfare scenarios.

Generates time-series DataFrames where each row has 8 construct values
(c, kappa, j, p, eps, lam_L, lam_P, xi) plus a computed Phi column.
Used for training and evaluating Phi forecasting models.
"""

import numpy as np
import pandas as pd

from src.inference.welfare_scoring import ALL_CONSTRUCTS, compute_phi


class PhiScenarioGenerator:
    """Generate synthetic Phi trajectories for welfare forecasting experiments.

    Each scenario encodes a narrative about how welfare constructs evolve
    over time, grounded in the Phi(humanity) framework.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Uses ``np.random.default_rng``.
    """

    SCENARIOS = (
        "stable_community",
        "capitalism_suppresses_love",
        "surveillance_state",
        "willful_ignorance",
        "recovery_arc",
        "sudden_crisis",
        "slow_decay",
        "random_walk",
    )

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, scenario: str, length: int = 200) -> pd.DataFrame:
        """Generate a single scenario trajectory.

        Parameters
        ----------
        scenario : str
            One of :attr:`SCENARIOS`.
        length : int
            Number of time steps (rows).

        Returns
        -------
        pd.DataFrame
            Columns: all 8 constructs + ``"phi"``.

        Raises
        ------
        ValueError
            If *scenario* is not in :attr:`SCENARIOS`.
        """
        dispatch = {
            "stable_community": self._stable_community,
            "capitalism_suppresses_love": self._capitalism_suppresses_love,
            "surveillance_state": self._surveillance_state,
            "willful_ignorance": self._willful_ignorance,
            "recovery_arc": self._recovery_arc,
            "sudden_crisis": self._sudden_crisis,
            "slow_decay": self._slow_decay,
            "random_walk": self._random_walk,
        }
        if scenario not in dispatch:
            raise ValueError(
                f"Unknown scenario: {scenario!r}. "
                f"Choose from: {', '.join(self.SCENARIOS)}"
            )
        return dispatch[scenario](length)

    def generate_dataset(
        self,
        scenarios_per_type: int = 5,
        length: int = 200,
    ) -> list[pd.DataFrame]:
        """Generate multiple trajectories for every scenario type.

        Parameters
        ----------
        scenarios_per_type : int
            How many trajectories to generate per scenario.
        length : int
            Number of time steps per trajectory.

        Returns
        -------
        list[pd.DataFrame]
            ``len(SCENARIOS) * scenarios_per_type`` DataFrames.
        """
        dataset: list[pd.DataFrame] = []
        for scenario in self.SCENARIOS:
            for _ in range(scenarios_per_type):
                dataset.append(self.generate(scenario, length=length))
        return dataset

    # ── Private: scenario builders ────────────────────────────────────────

    def _stable_community(self, length: int) -> pd.DataFrame:
        """All constructs hover around 0.5 with low noise."""
        data = {}
        for c in ALL_CONSTRUCTS:
            trend = np.full(length, 0.5)
            noise = self._rng.normal(0, 0.01, length)
            data[c] = trend + noise
        return self._finalize(data, length)

    def _capitalism_suppresses_love(self, length: int) -> pd.DataFrame:
        """lam_L declines 0.6 -> 0.1; purpose also erodes."""
        data = {}
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731

        data["lam_L"] = np.linspace(0.6, 0.1, length) + noise()
        data["p"] = np.linspace(0.55, 0.2, length) + noise()
        # Other constructs stay moderate
        data["c"] = np.full(length, 0.5) + noise()
        data["kappa"] = np.full(length, 0.45) + noise()
        data["j"] = np.linspace(0.5, 0.35, length) + noise()
        data["eps"] = np.full(length, 0.4) + noise()
        data["lam_P"] = np.full(length, 0.45) + noise()
        data["xi"] = np.full(length, 0.5) + noise()
        return self._finalize(data, length)

    def _surveillance_state(self, length: int) -> pd.DataFrame:
        """xi rises 0.3 -> 0.9; lam_L drops 0.6 -> 0.1; empathy erodes."""
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731

        data = {}
        data["xi"] = np.linspace(0.3, 0.9, length) + noise()
        data["lam_L"] = np.linspace(0.6, 0.1, length) + noise()
        data["eps"] = np.linspace(0.5, 0.2, length) + noise()
        data["c"] = np.full(length, 0.45) + noise()
        data["kappa"] = np.full(length, 0.4) + noise()
        data["j"] = np.linspace(0.45, 0.3, length) + noise()
        data["p"] = np.full(length, 0.4) + noise()
        data["lam_P"] = np.linspace(0.5, 0.7, length) + noise()
        return self._finalize(data, length)

    def _willful_ignorance(self, length: int) -> pd.DataFrame:
        """lam_L rises 0.3 -> 0.8; xi drops 0.7 -> 0.15."""
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731

        data = {}
        data["lam_L"] = np.linspace(0.3, 0.8, length) + noise()
        data["xi"] = np.linspace(0.7, 0.15, length) + noise()
        data["c"] = np.full(length, 0.5) + noise()
        data["kappa"] = np.full(length, 0.45) + noise()
        data["j"] = np.full(length, 0.5) + noise()
        data["p"] = np.full(length, 0.45) + noise()
        data["eps"] = np.full(length, 0.4) + noise()
        data["lam_P"] = np.full(length, 0.45) + noise()
        return self._finalize(data, length)

    def _recovery_arc(self, length: int) -> pd.DataFrame:
        """All constructs drop then lam_L recovers first (community leads)."""
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731

        # V-shaped curve: drops to nadir at ~40% then recovers
        def v_curve(start: float, nadir: float, end: float) -> np.ndarray:
            nadir_idx = int(length * 0.4)
            down = np.linspace(start, nadir, nadir_idx)
            up = np.linspace(nadir, end, length - nadir_idx)
            return np.concatenate([down, up])

        data = {}
        # lam_L recovers first and strongest (community leads recovery)
        data["lam_L"] = v_curve(0.6, 0.15, 0.65) + noise()
        # Other constructs follow with slower or weaker recovery
        data["c"] = v_curve(0.55, 0.2, 0.45) + noise()
        data["kappa"] = v_curve(0.5, 0.2, 0.4) + noise()
        data["j"] = v_curve(0.5, 0.15, 0.4) + noise()
        data["p"] = v_curve(0.5, 0.15, 0.4) + noise()
        data["eps"] = v_curve(0.45, 0.15, 0.35) + noise()
        data["lam_P"] = v_curve(0.5, 0.2, 0.4) + noise()
        data["xi"] = v_curve(0.55, 0.2, 0.45) + noise()
        return self._finalize(data, length)

    def _sudden_crisis(self, length: int) -> pd.DataFrame:
        """kappa and lam_P crash in the middle third."""
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731
        third = length // 3

        def crisis_curve(normal: float, crash: float) -> np.ndarray:
            before = np.full(third, normal)
            during = np.linspace(normal, crash, third)
            after = np.linspace(crash, normal * 0.8, length - 2 * third)
            return np.concatenate([before, during, after])

        data = {}
        data["kappa"] = crisis_curve(0.55, 0.1) + noise()
        data["lam_P"] = crisis_curve(0.5, 0.1) + noise()
        data["c"] = np.full(length, 0.5) + noise()
        data["lam_L"] = np.full(length, 0.5) + noise()
        data["j"] = crisis_curve(0.5, 0.25) + noise()
        data["p"] = np.full(length, 0.45) + noise()
        data["eps"] = np.full(length, 0.45) + noise()
        data["xi"] = np.full(length, 0.5) + noise()
        return self._finalize(data, length)

    def _slow_decay(self, length: int) -> pd.DataFrame:
        """All 8 constructs decline at different rates from 0.7."""
        noise = lambda: self._rng.normal(0, 0.01, length)  # noqa: E731

        # Different decay endpoints for each construct
        endpoints = {
            "c": 0.35,
            "kappa": 0.30,
            "j": 0.25,
            "p": 0.20,
            "eps": 0.25,
            "lam_L": 0.15,
            "lam_P": 0.30,
            "xi": 0.20,
        }
        data = {}
        for c in ALL_CONSTRUCTS:
            data[c] = np.linspace(0.7, endpoints[c], length) + noise()
        return self._finalize(data, length)

    def _random_walk(self, length: int) -> pd.DataFrame:
        """Correlated random walks with mean-reversion toward 0.5."""
        data = {}
        mean_reversion = 0.05  # pull toward 0.5 per step
        step_std = 0.02

        for c in ALL_CONSTRUCTS:
            values = np.empty(length)
            values[0] = 0.5 + self._rng.normal(0, 0.1)
            for i in range(1, length):
                reversion = mean_reversion * (0.5 - values[i - 1])
                step = self._rng.normal(0, step_std)
                values[i] = values[i - 1] + reversion + step
            data[c] = values
        return self._finalize(data, length)

    # ── Private: utilities ────────────────────────────────────────────────

    def _finalize(self, data: dict, length: int) -> pd.DataFrame:
        """Clamp values to [0, 1] and compute Phi for each row."""
        df = pd.DataFrame(data)
        # Clamp all constructs to [0, 1]
        for c in ALL_CONSTRUCTS:
            df[c] = df[c].clip(0.0, 1.0)
        # Compute Phi for each row
        df["phi"] = df.apply(
            lambda row: compute_phi({c: float(row[c]) for c in ALL_CONSTRUCTS}),
            axis=1,
        )
        return df
