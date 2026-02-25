# Phi Forecasting Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate Dignity-Model's CNN+LSTM+Attention backbone into wave-experiment for Phi trajectory forecasting, then train on HF Jobs GPU infrastructure.

**Architecture:** Layered forecast stack — synthetic data generator → signal processing → pipeline → PhiForecaster model (shared backbone, dual heads) → symbolic composition layers for urgency/gap prediction. Both repos edited in parallel.

**Tech Stack:** PyTorch 2.x, Dignity-Model backbone (CNN1D+LSTM+Attention), wave-experiment welfare scoring, HF Jobs for GPU training, Trackio for experiment tracking.

---

### Task 0: Set up Dignity-Model as git submodule + create src/forecasting/ package

**Files:**
- Create: `src/forecasting/__init__.py`
- Create: `tests/forecasting/__init__.py`
- Modify: `.gitmodules` (created by git submodule add)

**Step 1: Register Dignity-Model as git submodule**

The repo is already cloned at `Dignity-Model/`. Convert it to a proper submodule:

```bash
# Remove the existing clone (it's not tracked as a submodule yet)
rm -rf Dignity-Model/
git submodule add git@github.com:crichalchemist/Dignity-Model.git Dignity-Model
```

If SSH fails (no key), use the directory as-is and skip the submodule formality — the code is already present.

**Step 2: Create the forecasting package**

```bash
mkdir -p src/forecasting tests/forecasting
```

```python
# src/forecasting/__init__.py
"""Phi trajectory forecasting — predicting welfare evolution over time."""
```

```python
# tests/forecasting/__init__.py
```

**Step 3: Add Dignity-Model to Python path**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
pythonpath = [".", "Dignity-Model"]
```

This lets `from core.signals import SignalProcessor` resolve from `Dignity-Model/core/signals.py`.

**Step 4: Verify import path works**

```bash
python -c "import sys; sys.path.insert(0, 'Dignity-Model'); from core.signals import SignalProcessor; print('OK:', SignalProcessor)"
```

**Step 5: Commit**

```bash
git add src/forecasting/ tests/forecasting/ pyproject.toml
git commit -m "feat(forecasting): scaffold package + add Dignity-Model to Python path"
```

---

### Task 1: Synthetic Phi trajectory generator

**Files:**
- Create: `src/forecasting/synthetic.py`
- Create: `tests/forecasting/test_synthetic.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_synthetic.py
"""Tests for synthetic Phi trajectory generator."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.synthetic import PhiScenarioGenerator
from src.inference.welfare_scoring import ALL_CONSTRUCTS, compute_phi


class TestPhiScenarioGenerator:
    """Test that the generator produces valid 8-construct time-series."""

    def test_stable_community_shape(self):
        gen = PhiScenarioGenerator(seed=42)
        df = gen.generate("stable_community", length=200)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        for c in ALL_CONSTRUCTS:
            assert c in df.columns

    def test_all_constructs_in_bounds(self):
        gen = PhiScenarioGenerator(seed=42)
        for scenario in gen.SCENARIOS:
            df = gen.generate(scenario, length=100)
            for c in ALL_CONSTRUCTS:
                assert df[c].min() >= 0.0, f"{scenario}.{c} went below 0"
                assert df[c].max() <= 1.0, f"{scenario}.{c} went above 1"

    def test_phi_column_computed(self):
        gen = PhiScenarioGenerator(seed=42)
        df = gen.generate("stable_community", length=100)
        assert "phi" in df.columns
        # Verify phi matches compute_phi on first row
        row = {c: df[c].iloc[0] for c in ALL_CONSTRUCTS}
        expected = compute_phi(row)
        assert abs(df["phi"].iloc[0] - expected) < 1e-6

    def test_capitalism_suppresses_love(self):
        gen = PhiScenarioGenerator(seed=42)
        df = gen.generate("capitalism_suppresses_love", length=200)
        # lam_L should decline from start to end
        assert df["lam_L"].iloc[0] > df["lam_L"].iloc[-1]
        assert df["lam_L"].iloc[-1] < 0.2  # drops near floor

    def test_surveillance_state_divergence(self):
        gen = PhiScenarioGenerator(seed=42)
        df = gen.generate("surveillance_state", length=200)
        # xi rises while lam_L falls — truth without love
        assert df["xi"].iloc[-1] > df["xi"].iloc[0]
        assert df["lam_L"].iloc[-1] < df["lam_L"].iloc[0]

    def test_recovery_arc(self):
        gen = PhiScenarioGenerator(seed=42)
        df = gen.generate("recovery_arc", length=300)
        # Phi should dip then recover
        mid = len(df) // 2
        assert df["phi"].iloc[mid] < df["phi"].iloc[0]
        assert df["phi"].iloc[-1] > df["phi"].iloc[mid]

    def test_generate_dataset(self):
        gen = PhiScenarioGenerator(seed=42)
        dataset = gen.generate_dataset(scenarios_per_type=2, length=100)
        # 8 scenarios x 2 each = 16 sequences, each 100 rows
        assert len(dataset) == 16
        assert all(len(df) == 100 for df in dataset)

    def test_unknown_scenario_raises(self):
        gen = PhiScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown scenario"):
            gen.generate("nonexistent_scenario")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/forecasting/test_synthetic.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.forecasting.synthetic'`

**Step 3: Implement the synthetic generator**

```python
# src/forecasting/synthetic.py
"""Synthetic Phi trajectory generator — thought experiments as time-series."""
import numpy as np
import pandas as pd

from src.inference.welfare_scoring import ALL_CONSTRUCTS, compute_phi


class PhiScenarioGenerator:
    """Generate community welfare scenarios as 8-construct time-series.

    Each scenario encodes a theory of how welfare evolves:
    - capitalism_suppresses_love: hooks' insight that survival economics starves love
    - surveillance_state: Fricker's epistemic injustice — truth serving power
    - recovery_arc: can community solidarity alone pull constructs back?
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
        self.rng = np.random.default_rng(seed)

    def generate(self, scenario: str, length: int = 200) -> pd.DataFrame:
        """Generate a single scenario time-series."""
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {self.SCENARIOS}")

        method = getattr(self, f"_gen_{scenario}")
        constructs = method(length)

        df = pd.DataFrame(constructs)
        # Clamp to [0, 1]
        for c in ALL_CONSTRUCTS:
            df[c] = df[c].clip(0.0, 1.0)

        # Compute ground-truth Phi at each timestep
        df["phi"] = df.apply(
            lambda row: compute_phi({c: row[c] for c in ALL_CONSTRUCTS}), axis=1
        )
        return df

    def generate_dataset(
        self, scenarios_per_type: int = 5, length: int = 200
    ) -> list[pd.DataFrame]:
        """Generate multiple instances of each scenario."""
        dataset = []
        for scenario in self.SCENARIOS:
            for i in range(scenarios_per_type):
                gen = PhiScenarioGenerator(seed=self.rng.integers(0, 2**31) + i)
                dataset.append(gen.generate(scenario, length))
        return dataset

    # --- Scenario generators ---

    def _gen_stable_community(self, length: int) -> dict:
        noise = 0.02
        return {
            c: 0.5 + self.rng.normal(0, noise, length).cumsum().clip(-0.1, 0.1)
            for c in ALL_CONSTRUCTS
        }

    def _gen_capitalism_suppresses_love(self, length: int) -> dict:
        base = {c: 0.5 + self.rng.normal(0, 0.01, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}
        # Love decays linearly from 0.6 to 0.1
        base["lam_L"] = np.linspace(0.6, 0.1, length) + self.rng.normal(0, 0.01, length)
        # Purpose also erodes (capitalism narrows purpose to profit)
        base["p"] = np.linspace(0.5, 0.25, length) + self.rng.normal(0, 0.01, length)
        return base

    def _gen_surveillance_state(self, length: int) -> dict:
        base = {c: 0.5 + self.rng.normal(0, 0.01, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}
        # Truth rises (surveillance = institutional transparency serving control)
        base["xi"] = np.linspace(0.3, 0.9, length) + self.rng.normal(0, 0.01, length)
        # Love drops (no care behind the monitoring)
        base["lam_L"] = np.linspace(0.6, 0.1, length) + self.rng.normal(0, 0.01, length)
        # Empathy also drops (perspective-taking replaced by profiling)
        base["eps"] = np.linspace(0.5, 0.15, length) + self.rng.normal(0, 0.01, length)
        return base

    def _gen_willful_ignorance(self, length: int) -> dict:
        base = {c: 0.5 + self.rng.normal(0, 0.01, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}
        # Love rises (community solidarity strong)
        base["lam_L"] = np.linspace(0.3, 0.8, length) + self.rng.normal(0, 0.01, length)
        # Truth drops (community refuses uncomfortable facts)
        base["xi"] = np.linspace(0.7, 0.15, length) + self.rng.normal(0, 0.01, length)
        return base

    def _gen_recovery_arc(self, length: int) -> dict:
        third = length // 3
        base = {}
        for c in ALL_CONSTRUCTS:
            decline = np.linspace(0.5, 0.15, third)
            hold = np.full(third, 0.15)
            recover = np.linspace(0.15, 0.45, length - 2 * third)
            base[c] = np.concatenate([decline, hold, recover]) + self.rng.normal(0, 0.01, length)
        # lam_L recovers faster (community leads recovery)
        recover_lam = np.linspace(0.15, 0.7, length - 2 * third)
        base["lam_L"] = np.concatenate([
            np.linspace(0.5, 0.15, third),
            np.full(third, 0.15),
            recover_lam,
        ]) + self.rng.normal(0, 0.01, length)
        return base

    def _gen_sudden_crisis(self, length: int) -> dict:
        base = {c: 0.5 + self.rng.normal(0, 0.01, length).cumsum().clip(-0.1, 0.1)
                for c in ALL_CONSTRUCTS}
        crisis_start, crisis_end = length // 3, 2 * length // 3
        # Compassion and protection crash during crisis
        for c in ("kappa", "lam_P"):
            base[c][crisis_start:crisis_end] = np.linspace(
                0.5, 0.1, crisis_end - crisis_start
            ) + self.rng.normal(0, 0.01, crisis_end - crisis_start)
        return base

    def _gen_slow_decay(self, length: int) -> dict:
        rates = {"c": 0.002, "kappa": 0.001, "j": 0.003, "p": 0.001,
                 "eps": 0.002, "lam_L": 0.003, "lam_P": 0.001, "xi": 0.002}
        base = {}
        for c in ALL_CONSTRUCTS:
            base[c] = 0.7 - rates[c] * np.arange(length) + self.rng.normal(0, 0.01, length)
        return base

    def _gen_random_walk(self, length: int) -> dict:
        base = {}
        for c in ALL_CONSTRUCTS:
            walk = np.zeros(length)
            walk[0] = 0.5
            for i in range(1, length):
                walk[i] = walk[i - 1] + self.rng.normal(0, 0.02)
                # Mean-reversion toward 0.5
                walk[i] += 0.01 * (0.5 - walk[i])
            base[c] = walk
        return base
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/forecasting/test_synthetic.py -v
```

Expected: all 8 tests PASS

**Step 5: Commit**

```bash
git add src/forecasting/synthetic.py tests/forecasting/test_synthetic.py
git commit -m "feat(forecasting): synthetic Phi trajectory generator — 8 welfare scenarios"
```

---

### Task 2: Signal processing — extend Dignity's SignalProcessor + create adapter

**Files:**
- Modify: `Dignity-Model/core/signals.py` (add 3 methods after line 177)
- Create: `src/forecasting/signals.py`
- Create: `tests/forecasting/test_signals.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_signals.py
"""Tests for Phi construct signal processing."""
import numpy as np
import pytest

from src.forecasting.signals import PhiSignalProcessor


class TestPhiSignalProcessor:
    """Test construct-level signal computation."""

    def test_construct_volatility(self):
        values = np.array([0.5, 0.52, 0.48, 0.51, 0.49] * 10)
        vol = PhiSignalProcessor.volatility(values, window=5)
        assert len(vol) == len(values)
        assert vol[10] > 0  # non-zero after window fills

    def test_construct_momentum(self):
        values = np.linspace(0.3, 0.7, 50)  # steady uptrend
        mom = PhiSignalProcessor.price_momentum(values, window=5)
        assert mom[10] > 0  # positive momentum in uptrend

    def test_synergy_signal(self):
        a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        b = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        syn = PhiSignalProcessor.synergy_signal(a, b, window=3)
        assert len(syn) == 5
        assert all(s >= 0 for s in syn)

    def test_synergy_penalizes_imbalance(self):
        a = np.full(20, 0.9)
        b = np.full(20, 0.1)
        syn_imbalanced = PhiSignalProcessor.synergy_signal(a, b, window=5)
        a2 = np.full(20, 0.5)
        b2 = np.full(20, 0.5)
        syn_balanced = PhiSignalProcessor.synergy_signal(a2, b2, window=5)
        # Balanced should have higher synergy at same-ish sum
        assert syn_balanced[-1] > syn_imbalanced[-1]

    def test_divergence_signal(self):
        a = np.full(20, 0.9)
        b = np.full(20, 0.1)
        div = PhiSignalProcessor.divergence_signal(a, b, window=5)
        assert len(div) == 20
        assert div[-1] > 0.5  # large divergence

    def test_phi_derivative(self):
        phi = np.linspace(0.3, 0.7, 50)  # steadily increasing
        dphi = PhiSignalProcessor.phi_derivative(phi)
        assert len(dphi) == 50
        assert np.mean(dphi[1:]) > 0  # positive derivative

    def test_compute_all_signals(self):
        import pandas as pd
        from src.inference.welfare_scoring import ALL_CONSTRUCTS
        data = {c: np.random.default_rng(42).uniform(0.3, 0.7, 100) for c in ALL_CONSTRUCTS}
        df = pd.DataFrame(data)
        result = PhiSignalProcessor.compute_all_signals(df)
        # Should have: 8 raw + 8 vol + 8 mom + 5 synergy + 5 divergence + 1 phi = 35
        assert result.shape[1] >= 34
        assert result.shape[0] == 100
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/forecasting/test_signals.py -v
```

**Step 3: Add Phi-specific methods to Dignity's SignalProcessor**

Add after line 177 in `Dignity-Model/core/signals.py`:

```python
    @staticmethod
    def synergy_signal(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling geometric mean synergy between two construct time-series."""
        geo = np.sqrt(np.maximum(0, a) * np.maximum(0, b))
        if len(geo) < window:
            return geo
        result = np.zeros_like(geo)
        for i in range(window, len(geo)):
            result[i] = np.mean(geo[i - window : i])
        result[:window] = result[window] if window < len(geo) else 0
        return result

    @staticmethod
    def divergence_signal(a: np.ndarray, b: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling squared divergence between two construct time-series."""
        sq_diff = (a - b) ** 2
        if len(sq_diff) < window:
            return sq_diff
        result = np.zeros_like(sq_diff)
        for i in range(window, len(sq_diff)):
            result[i] = np.mean(sq_diff[i - window : i])
        result[:window] = result[window] if window < len(sq_diff) else 0
        return result

    @staticmethod
    def phi_derivative(phi: np.ndarray) -> np.ndarray:
        """Rate of change of Phi over time."""
        return np.gradient(phi)
```

**Step 4: Create the adapter in wave-experiment**

```python
# src/forecasting/signals.py
"""Phi construct signal processing — adapter over Dignity's SignalProcessor."""
import sys
import numpy as np
import pandas as pd

# Add Dignity-Model to path if not already
sys.path.insert(0, "Dignity-Model") if "Dignity-Model" not in sys.path else None

from core.signals import SignalProcessor as _DignitySignals

from src.inference.welfare_scoring import (
    ALL_CONSTRUCTS,
    CONSTRUCT_PAIRS,
    CURIOSITY_CROSS_PAIR,
    PENALTY_PAIRS,
    compute_phi,
)


class PhiSignalProcessor(_DignitySignals):
    """Extends Dignity's SignalProcessor with Phi-specific signal computation.

    Inherits: volatility, entropy, price_momentum, directional_change,
              regime_detection, synergy_signal, divergence_signal, phi_derivative
    """

    @classmethod
    def compute_all_signals(cls, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Compute all 34+ features from raw construct time-series.

        Input: DataFrame with columns matching ALL_CONSTRUCTS (8 cols).
        Output: DataFrame with 34+ columns: raw + volatility + momentum
                + synergy + divergence + phi.
        """
        result = df.copy()

        # 8 volatilities
        for c in ALL_CONSTRUCTS:
            result[f"{c}_vol"] = cls.volatility(df[c].values, window=window)

        # 8 momentums
        for c in ALL_CONSTRUCTS:
            result[f"{c}_mom"] = cls.price_momentum(df[c].values, window=window)

        # 5 synergy signals (4 primary + 1 curiosity)
        all_pairs = list(CONSTRUCT_PAIRS) + [CURIOSITY_CROSS_PAIR]
        for a, b in all_pairs:
            result[f"syn_{a}_{b}"] = cls.synergy_signal(
                df[a].values, df[b].values, window=window
            )

        # 5 divergence signals
        for a, b in PENALTY_PAIRS:
            result[f"div_{a}_{b}"] = cls.divergence_signal(
                df[a].values, df[b].values, window=window
            )

        # Phi
        result["phi"] = df.apply(
            lambda row: compute_phi({c: row[c] for c in ALL_CONSTRUCTS}), axis=1
        )

        return result
```

**Step 5: Run tests**

```bash
pytest tests/forecasting/test_signals.py -v
```

**Step 6: Commit both repos**

```bash
# Wave-experiment
git add src/forecasting/signals.py tests/forecasting/test_signals.py
# Dignity-Model changes
cd Dignity-Model && git add core/signals.py && git commit -m "feat(signals): add synergy_signal, divergence_signal, phi_derivative for welfare forecasting" && cd ..
git add Dignity-Model src/forecasting/signals.py tests/forecasting/test_signals.py
git commit -m "feat(forecasting): signal processing adapter — 34-feature construct signals"
```

---

### Task 3: PhiPipeline — scaling + windowing

**Files:**
- Create: `src/forecasting/pipeline.py`
- Create: `tests/forecasting/test_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_pipeline.py
"""Tests for PhiPipeline preprocessing."""
import numpy as np
import pytest

from src.forecasting.pipeline import PhiPipeline
from src.forecasting.synthetic import PhiScenarioGenerator
from src.inference.welfare_scoring import ALL_CONSTRUCTS


class TestPhiPipeline:
    def setup_method(self):
        self.gen = PhiScenarioGenerator(seed=42)
        self.df = self.gen.generate("stable_community", length=200)

    def test_fit_transform_shape(self):
        pipe = PhiPipeline(seq_len=50)
        X = pipe.fit_transform(self.df)
        assert X.ndim == 2
        assert X.shape[0] == 200
        assert X.shape[1] >= 34

    def test_create_sequences_shape(self):
        pipe = PhiPipeline(seq_len=50)
        X = pipe.fit_transform(self.df)
        X_seq, y_seq = pipe.create_sequences(X, y=self.df["phi"].values)
        assert X_seq.shape == (151, 50, X.shape[1])  # (200-50+1, 50, features)
        assert y_seq.shape == (151,)

    def test_scaling_reversible(self):
        pipe = PhiPipeline(seq_len=50)
        X = pipe.fit_transform(self.df)
        # Scaled values should have roughly zero median
        assert abs(np.median(X[:, 0])) < 1.0

    def test_pipeline_process(self):
        pipe = PhiPipeline(seq_len=50)
        X_seq, y_seq = pipe.process(self.df, labels=self.df["phi"].values)
        assert X_seq.ndim == 3
        assert y_seq is not None
```

**Step 2: Run to verify failure, then implement**

```python
# src/forecasting/pipeline.py
"""PhiPipeline — preprocessing for Phi construct time-series."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.forecasting.signals import PhiSignalProcessor
from src.inference.welfare_scoring import ALL_CONSTRUCTS


class PhiPipeline:
    """Transform raw construct time-series into scaled, windowed sequences.

    Pipeline: raw constructs → compute signals (34 features) → scale → window.
    """

    def __init__(self, seq_len: int = 100, window: int = 20):
        self.seq_len = seq_len
        self.window = window
        self.scaler = RobustScaler()
        self.fitted = False
        self.feature_names: list[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 34+ signal features from raw constructs."""
        return PhiSignalProcessor.compute_all_signals(df, window=self.window)

    def fit(self, df: pd.DataFrame) -> "PhiPipeline":
        """Fit the scaler on training data."""
        features = self.compute_features(df)
        self.feature_names = list(features.columns)
        self.scaler.fit(features.values)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to scaled feature array [n_samples, n_features]."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        features = self.compute_features(df)
        return self.scaler.transform(features[self.feature_names].values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Sliding window → [n_seq, seq_len, features]."""
        n_samples, n_features = X.shape
        n_seq = (n_samples - self.seq_len) // stride + 1
        if n_seq <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for seq_len={self.seq_len}")

        X_seq = np.zeros((n_seq, self.seq_len, n_features))
        for i in range(n_seq):
            start = i * stride
            X_seq[i] = X[start : start + self.seq_len]

        y_seq = None
        if y is not None:
            y_seq = np.array([y[i * stride + self.seq_len - 1] for i in range(n_seq)])

        return X_seq, y_seq

    def process(
        self, df: pd.DataFrame, labels: np.ndarray | None = None,
        fit: bool = True, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Full pipeline: features → scale → window."""
        X = self.fit_transform(df) if fit else self.transform(df)
        return self.create_sequences(X, labels, stride)
```

**Step 3: Run tests, commit**

```bash
pytest tests/forecasting/test_pipeline.py -v
git add src/forecasting/pipeline.py tests/forecasting/test_pipeline.py
git commit -m "feat(forecasting): PhiPipeline — scaling + windowing for 34-feature construct signals"
```

---

### Task 4: PhiForecaster model — backbone + dual heads

**Files:**
- Create: `src/forecasting/model.py`
- Create: `tests/forecasting/test_model.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_model.py
"""Tests for PhiForecaster model."""
import torch
import pytest

from src.forecasting.model import PhiForecaster


class TestPhiForecaster:
    def test_forward_shapes(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        x = torch.randn(4, 50, 34)  # [batch, seq_len, features]
        phi_pred, construct_pred, attn = model(x)
        assert phi_pred.shape == (4, 5, 1)       # [batch, pred_len, 1]
        assert construct_pred.shape == (4, 5, 8)  # [batch, pred_len, 8]
        assert attn.shape == (4, 50)               # [batch, seq_len]

    def test_predict_phi(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        x = torch.randn(2, 50, 34)
        phi = model.predict_phi(x)
        assert phi.shape == (2, 5, 1)

    def test_predict_constructs(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        x = torch.randn(2, 50, 34)
        constructs = model.predict_constructs(x)
        assert constructs.shape == (2, 5, 8)

    def test_gradient_flows(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        x = torch.randn(4, 50, 34, requires_grad=False)
        phi_pred, construct_pred, _ = model(x)
        loss = phi_pred.mean() + construct_pred.mean()
        loss.backward()
        # Check backbone gradients exist
        for p in model.backbone.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_multi_task_loss(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        loss_fn = model.compute_loss(
            phi_pred=torch.randn(4, 5, 1),
            phi_target=torch.randn(4, 5, 1),
            construct_pred=torch.randn(4, 5, 8),
            construct_target=torch.randn(4, 5, 8),
        )
        assert loss_fn.item() > 0

    def test_num_parameters(self):
        model = PhiForecaster(input_size=34, hidden_size=64, n_layers=1, pred_len=5)
        assert model.num_parameters > 0
```

**Step 2: Implement the model**

```python
# src/forecasting/model.py
"""PhiForecaster — Dignity backbone + dual forecast heads for Phi prediction."""
import sys

sys.path.insert(0, "Dignity-Model") if "Dignity-Model" not in sys.path else None

import torch
import torch.nn as nn

from models.backbone.hybrid import DignityBackbone
from models.head.forecast import ForecastHead


class PhiForecaster(nn.Module):
    """Forecast Phi trajectories and per-construct evolution.

    Shared backbone (CNN1D + LSTM + Attention) with two heads:
    - phi_head: scalar Phi forecast [batch, pred_len, 1]
    - construct_head: 8-channel construct forecast [batch, pred_len, 8]

    Multi-task loss: L = L_phi + alpha * L_construct
    """

    def __init__(
        self,
        input_size: int = 34,
        hidden_size: int = 256,
        n_layers: int = 2,
        pred_len: int = 10,
        dropout: float = 0.1,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha

        self.backbone = DignityBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.phi_head = ForecastHead(
            input_size=hidden_size,
            pred_len=pred_len,
            num_features=1,
            hidden_size=hidden_size // 2,
            dropout=dropout,
        )

        self.construct_head = ForecastHead(
            input_size=hidden_size,
            pred_len=pred_len,
            num_features=8,
            hidden_size=hidden_size // 2,
            dropout=dropout,
        )

        self._loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (phi_pred, construct_pred, attention_weights)."""
        context, attn = self.backbone(x)
        phi_pred = self.phi_head(context)
        construct_pred = self.construct_head(context)
        return phi_pred, construct_pred, attn

    def predict_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: predict Phi trajectory only."""
        self.eval()
        with torch.no_grad():
            phi, _, _ = self.forward(x)
        return phi

    def predict_constructs(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: predict construct trajectories only."""
        self.eval()
        with torch.no_grad():
            _, constructs, _ = self.forward(x)
        return constructs

    def compute_loss(
        self,
        phi_pred: torch.Tensor,
        phi_target: torch.Tensor,
        construct_pred: torch.Tensor,
        construct_target: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-task loss: L = L_phi + alpha * L_construct."""
        l_phi = self._loss_fn(phi_pred, phi_target)
        l_construct = self._loss_fn(construct_pred, construct_target)
        return l_phi + self.alpha * l_construct

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Step 3: Run tests, commit**

```bash
pytest tests/forecasting/test_model.py -v
git add src/forecasting/model.py tests/forecasting/test_model.py
git commit -m "feat(forecasting): PhiForecaster — dual-head model for Phi + construct prediction"
```

---

### Task 5: Training engine wrapper

**Files:**
- Create: `src/forecasting/engine.py`
- Create: `tests/forecasting/test_engine.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_engine.py
"""Tests for Phi forecasting training engine."""
import torch
import pytest

from src.forecasting.engine import train_phi_epoch
from src.forecasting.model import PhiForecaster


class TestTrainPhiEpoch:
    def test_loss_decreases_over_batch(self):
        model = PhiForecaster(input_size=34, hidden_size=32, n_layers=1, pred_len=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Fake data: 8 sequences, each [50, 34] features
        X = torch.randn(8, 50, 34)
        # Targets: [8, 5, 1] for phi, [8, 5, 8] for constructs
        y_phi = torch.randn(8, 5, 1)
        y_construct = torch.randn(8, 5, 8)

        dataset = torch.utils.data.TensorDataset(X, y_phi, y_construct)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        metrics = train_phi_epoch(
            model, loader, optimizer, device=torch.device("cpu"), use_amp=False
        )
        assert "loss" in metrics
        assert metrics["loss"] > 0
```

**Step 2: Implement**

```python
# src/forecasting/engine.py
"""Training engine for PhiForecaster — wraps Dignity's engine patterns."""
import torch
import torch.nn as nn
from tqdm import tqdm


def train_phi_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train PhiForecaster for one epoch.

    Expects dataloader to yield (X, y_phi, y_construct) batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y_phi, y_construct = [b.to(device) for b in batch]

        optimizer.zero_grad()
        phi_pred, construct_pred, _ = model(x)
        loss = model.compute_loss(phi_pred, y_phi, construct_pred, y_construct)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return {"loss": total_loss / max(num_batches, 1)}


def validate_phi_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate PhiForecaster for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            x, y_phi, y_construct = [b.to(device) for b in batch]
            phi_pred, construct_pred, _ = model(x)
            loss = model.compute_loss(phi_pred, y_phi, construct_pred, y_construct)
            total_loss += loss.item()

    return {"loss": total_loss / max(num_batches, 1)}
```

**Step 3: Run tests, commit**

```bash
pytest tests/forecasting/test_engine.py -v
git add src/forecasting/engine.py tests/forecasting/test_engine.py
git commit -m "feat(forecasting): training engine — multi-task train/validate loops"
```

---

### Task 6: Layer 1 — Phi trajectory forecasting

**Files:**
- Create: `src/forecasting/phi_trajectory.py`
- Create: `tests/forecasting/test_phi_trajectory.py`

**Step 1: Write the failing tests**

```python
# tests/forecasting/test_phi_trajectory.py
"""Tests for Layer 1: Phi trajectory forecasting."""
import pytest
import torch

from src.forecasting.phi_trajectory import PhiTrajectoryForecaster


class TestPhiTrajectoryForecaster:
    def test_forecast_returns_dict(self):
        forecaster = PhiTrajectoryForecaster(hidden_size=32, n_layers=1, pred_len=5)
        result = forecaster.forecast_from_scenario("stable_community", history_len=100)
        assert "phi_predicted" in result
        assert "phi_actual" in result
        assert len(result["phi_predicted"]) == 5

    def test_forecast_bounded(self):
        forecaster = PhiTrajectoryForecaster(hidden_size=32, n_layers=1, pred_len=5)
        result = forecaster.forecast_from_scenario("capitalism_suppresses_love", history_len=100)
        # Predictions should be finite
        assert all(abs(v) < 10 for v in result["phi_predicted"])
```

**Step 2: Implement**

```python
# src/forecasting/phi_trajectory.py
"""Layer 1: Phi trajectory forecasting — predict scalar Phi(t+1...t+k)."""
import numpy as np
import torch

from src.forecasting.model import PhiForecaster
from src.forecasting.pipeline import PhiPipeline
from src.forecasting.synthetic import PhiScenarioGenerator
from src.inference.welfare_scoring import ALL_CONSTRUCTS


class PhiTrajectoryForecaster:
    """End-to-end Phi trajectory prediction from scenarios or raw data."""

    def __init__(self, hidden_size: int = 256, n_layers: int = 2, pred_len: int = 10):
        self.pred_len = pred_len
        self.pipeline = PhiPipeline(seq_len=50)
        self.model = PhiForecaster(
            input_size=34, hidden_size=hidden_size, n_layers=n_layers, pred_len=pred_len
        )

    def forecast_from_scenario(
        self, scenario: str, history_len: int = 200, seed: int = 42
    ) -> dict:
        """Generate a scenario and predict future Phi from its history."""
        gen = PhiScenarioGenerator(seed=seed)
        df = gen.generate(scenario, length=history_len + self.pred_len)

        # Split into history (input) and future (target)
        history = df.iloc[:history_len]
        future = df.iloc[history_len:]

        # Run pipeline
        X = self.pipeline.fit_transform(history)
        X_seq = X[np.newaxis, -self.pipeline.seq_len:]  # last window
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        # Predict
        phi_pred = self.model.predict_phi(X_tensor)  # [1, pred_len, 1]
        phi_predicted = phi_pred[0, :, 0].numpy().tolist()
        phi_actual = future["phi"].values[: self.pred_len].tolist()

        return {
            "phi_predicted": phi_predicted,
            "phi_actual": phi_actual,
            "scenario": scenario,
        }
```

**Step 3: Run tests, commit**

```bash
pytest tests/forecasting/test_phi_trajectory.py -v
git add src/forecasting/phi_trajectory.py tests/forecasting/test_phi_trajectory.py
git commit -m "feat(forecasting): Layer 1 — Phi trajectory forecasting from scenarios"
```

---

### Task 7: Layers 3-4 — symbolic composition (urgency + gap prediction)

**Files:**
- Create: `src/forecasting/construct_forecast.py`
- Create: `src/forecasting/urgency_forecast.py`
- Create: `src/forecasting/gap_prediction.py`
- Create: `tests/forecasting/test_urgency_forecast.py`
- Create: `tests/forecasting/test_gap_prediction.py`

**Step 1: Write failing tests**

```python
# tests/forecasting/test_urgency_forecast.py
"""Tests for Layer 3: hypothesis urgency forecasting."""
import pytest

from src.forecasting.urgency_forecast import forecast_hypothesis_urgency
from src.detective.hypothesis import Hypothesis
from src.inference.welfare_scoring import ALL_CONSTRUCTS


class TestUrgencyForecast:
    def test_declining_constructs_increase_urgency(self):
        h = Hypothesis.create("Resource gap in records 2013-2017", confidence=0.8)

        current = {c: 0.5 for c in ALL_CONSTRUCTS}
        future = {c: 0.3 for c in ALL_CONSTRUCTS}  # everything declining

        current_urgency = forecast_hypothesis_urgency(h, current)
        future_urgency = forecast_hypothesis_urgency(h, future)

        # Future urgency should be higher when constructs are lower
        assert future_urgency >= current_urgency

    def test_stable_constructs_stable_urgency(self):
        h = Hypothesis.create("Test hypothesis", confidence=0.5)
        metrics = {c: 0.5 for c in ALL_CONSTRUCTS}
        urgency = forecast_hypothesis_urgency(h, metrics)
        assert 0 <= urgency <= 1
```

```python
# tests/forecasting/test_gap_prediction.py
"""Tests for Layer 4: document gap prediction."""
import pytest

from src.forecasting.gap_prediction import predict_gap_emergence
from src.inference.welfare_scoring import ALL_CONSTRUCTS, CONSTRUCT_FLOORS


class TestGapPrediction:
    def test_below_floor_flagged(self):
        future = {c: 0.5 for c in ALL_CONSTRUCTS}
        future["xi"] = 0.1  # below floor of 0.3
        at_risk = predict_gap_emergence(future)
        assert "xi" in at_risk

    def test_above_floor_not_flagged(self):
        future = {c: 0.5 for c in ALL_CONSTRUCTS}
        at_risk = predict_gap_emergence(future)
        assert len(at_risk) == 0

    def test_multiple_at_risk(self):
        future = {c: 0.05 for c in ALL_CONSTRUCTS}
        at_risk = predict_gap_emergence(future)
        assert len(at_risk) == 8  # all below their floors
```

**Step 2: Implement**

```python
# src/forecasting/construct_forecast.py
"""Layer 2: per-construct trajectory forecasting (wrapper around model.predict_constructs)."""
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def forecasted_metrics_to_dict(construct_array) -> dict[str, float]:
    """Convert a [8] array of construct predictions to a named dict."""
    return {c: float(construct_array[i]) for i, c in enumerate(ALL_CONSTRUCTS)}
```

```python
# src/forecasting/urgency_forecast.py
"""Layer 3: hypothesis urgency forecasting via symbolic composition."""
from typing import Dict

from src.detective.hypothesis import Hypothesis
from src.inference.welfare_scoring import score_hypothesis_welfare


def forecast_hypothesis_urgency(
    hypothesis: Hypothesis,
    future_metrics: Dict[str, float],
) -> float:
    """Predict future urgency by scoring a hypothesis against forecasted construct levels.

    This is symbolic composition: no new model, just welfare scoring
    applied to numerically forecasted metrics.
    """
    return score_hypothesis_welfare(hypothesis, future_metrics)
```

```python
# src/forecasting/gap_prediction.py
"""Layer 4: document gap emergence prediction via symbolic composition."""
from typing import Dict, List

from src.inference.welfare_scoring import ALL_CONSTRUCTS, CONSTRUCT_FLOORS


def predict_gap_emergence(
    future_metrics: Dict[str, float],
) -> List[str]:
    """Predict which constructs will produce information gaps.

    A construct approaching or below its hard floor is likely to produce
    information gaps — the system needs more data about what's failing.
    """
    return [
        c for c in ALL_CONSTRUCTS
        if future_metrics.get(c, 0.5) < CONSTRUCT_FLOORS.get(c, 0.1)
    ]
```

**Step 3: Run tests, commit**

```bash
pytest tests/forecasting/test_urgency_forecast.py tests/forecasting/test_gap_prediction.py -v
git add src/forecasting/construct_forecast.py src/forecasting/urgency_forecast.py src/forecasting/gap_prediction.py tests/forecasting/test_urgency_forecast.py tests/forecasting/test_gap_prediction.py
git commit -m "feat(forecasting): Layers 2-4 — construct forecast, urgency prediction, gap emergence"
```

---

### Task 8: Full regression test + integration validation

**Files:**
- None created — run existing + new tests

**Step 1: Run all forecasting tests**

```bash
pytest tests/forecasting/ -v
```

Expected: all new tests pass.

**Step 2: Run full wave-experiment test suite**

```bash
pytest tests/ -v --timeout=120
```

Expected: 421+ existing tests pass, 0 regressions. Note the 7 pre-existing failures in `test_document_ingestion` and `test_parallel_evolution` are expected.

**Step 3: Commit if any fixes needed**

---

### Task 9: Local training smoke test

**Files:**
- Create: `scripts/train_phi_forecaster.py`

**Step 1: Create training script**

```python
# scripts/train_phi_forecaster.py
"""Train PhiForecaster on synthetic data — local smoke test."""
import torch
import numpy as np

from src.forecasting.synthetic import PhiScenarioGenerator
from src.forecasting.pipeline import PhiPipeline
from src.forecasting.model import PhiForecaster
from src.forecasting.engine import train_phi_epoch, validate_phi_epoch
from src.inference.welfare_scoring import ALL_CONSTRUCTS


def main():
    seed = 42
    pred_len = 10
    seq_len = 50
    hidden_size = 64
    epochs = 20
    batch_size = 16
    lr = 1e-3

    print("Generating synthetic data...")
    gen = PhiScenarioGenerator(seed=seed)
    dataset = gen.generate_dataset(scenarios_per_type=10, length=200)

    print(f"Generated {len(dataset)} scenario sequences")

    pipe = PhiPipeline(seq_len=seq_len)

    all_X, all_y_phi, all_y_construct = [], [], []

    for df in dataset:
        X = pipe.fit_transform(df) if not pipe.fitted else pipe.transform(df)
        n_features = X.shape[1]

        for i in range(len(X) - seq_len - pred_len):
            all_X.append(X[i : i + seq_len])

            phi_target = df["phi"].values[i + seq_len : i + seq_len + pred_len]
            all_y_phi.append(phi_target.reshape(-1, 1))

            construct_target = np.stack([
                df[c].values[i + seq_len : i + seq_len + pred_len]
                for c in ALL_CONSTRUCTS
            ], axis=-1)
            all_y_construct.append(construct_target)

    X_all = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_phi = torch.tensor(np.array(all_y_phi), dtype=torch.float32)
    y_construct = torch.tensor(np.array(all_y_construct), dtype=torch.float32)

    print(f"Training data: {X_all.shape[0]} sequences, {n_features} features")

    # Train/val split
    n = len(X_all)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = torch.utils.data.TensorDataset(X_all[train_idx], y_phi[train_idx], y_construct[train_idx])
    val_ds = torch.utils.data.TensorDataset(X_all[val_idx], y_phi[val_idx], y_construct[val_idx])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cpu")
    model = PhiForecaster(
        input_size=n_features, hidden_size=hidden_size, n_layers=1, pred_len=pred_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model: {model.num_parameters:,} parameters")
    print(f"Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_metrics = train_phi_epoch(model, train_loader, optimizer, device, use_amp=False)
        val_metrics = validate_phi_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} — train_loss: {train_metrics['loss']:.4f}, val_loss: {val_metrics['loss']:.4f}")

    print("Done. Saving checkpoint...")
    torch.save(model.state_dict(), "checkpoints/phi_forecaster_smoke.pt")
    print("Saved to checkpoints/phi_forecaster_smoke.pt")


if __name__ == "__main__":
    main()
```

**Step 2: Run the smoke test**

```bash
mkdir -p checkpoints
python scripts/train_phi_forecaster.py
```

Expected: loss decreases over 20 epochs. This validates the full pipeline end-to-end on CPU.

**Step 3: Commit**

```bash
git add scripts/train_phi_forecaster.py
git commit -m "feat(forecasting): local training smoke test script"
```

---

### Task 10: GPU training on HF Jobs

**Files:**
- Create: `scripts/train_phi_hf_job.py` (UV script with PEP 723 metadata)

**Invoke:** `huggingface-skills:hugging-face-jobs` skill for HF Jobs infrastructure.

This task creates a UV script that:
1. Generates synthetic data (same as Task 9)
2. Trains PhiForecaster on GPU (A10G or L4)
3. Logs metrics via Trackio
4. Saves checkpoint to HF Hub

**Step 1: Create the HF Jobs training script**

Use the `huggingface-skills:hugging-face-jobs` skill to create a UV script with PEP 723 metadata that trains the PhiForecaster on HF GPU infrastructure.

**Step 2: Submit the job**

```bash
hf jobs run scripts/train_phi_hf_job.py --hardware a10g-small --timeout 3600
```

**Step 3: Monitor with Trackio**

Use the `huggingface-skills:hugging-face-trackio` skill to track training metrics.

**Step 4: Commit**

```bash
git add scripts/train_phi_hf_job.py
git commit -m "feat(forecasting): HF Jobs GPU training script with Trackio monitoring"
```

---

### Task 11: Final verification against success criteria

**Step 1: Run verification tests**

```bash
pytest tests/forecasting/ -v
pytest tests/ -v --timeout=120
```

**Step 2: Verify success criteria**

1. ✅ Synthetic generator produces all 8 scenarios — verified by Task 1 tests
2. ✅ PhiPipeline produces 34+ features — verified by Task 3 tests
3. ✅ PhiForecaster trains and converges — verified by Task 9 smoke test
4. ✅ Layer 1 Phi forecast works — verified by Task 6 tests
5. ✅ Layer 2 construct forecast works — verified by Task 7 tests
6. ✅ Layer 3 urgency forecast works — verified by Task 7 tests
7. ✅ Layer 4 gap prediction works — verified by Task 7 tests
8. ✅ All existing tests pass — verified by Task 8 regression test
