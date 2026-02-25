"""Layer 1: Phi trajectory forecasting — predict scalar Phi(t+1...t+k).

End-to-end forecaster that:
  1. Generates (or accepts) welfare construct time-series
  2. Computes 36-column signal features via PhiPipeline
  3. Feeds the feature window through PhiForecaster
  4. Returns predicted vs actual Phi trajectories
"""

import numpy as np
import torch

from src.forecasting.model import PhiForecaster
from src.forecasting.pipeline import PhiPipeline
from src.forecasting.synthetic import PhiScenarioGenerator


class PhiTrajectoryForecaster:
    """End-to-end Phi trajectory prediction from scenarios or raw data.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension for PhiForecaster backbone.
    n_layers : int
        Number of LSTM layers.
    pred_len : int
        Number of future Phi values to predict.
    seq_len : int
        Length of the input sequence window fed to the model.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        n_layers: int = 2,
        pred_len: int = 10,
        seq_len: int = 50,
    ):
        self.pred_len = pred_len
        self.pipeline = PhiPipeline(seq_len=seq_len)
        self.model = PhiForecaster(
            input_size=36,
            hidden_size=hidden_size,
            n_layers=n_layers,
            pred_len=pred_len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast_from_scenario(
        self,
        scenario: str,
        history_len: int = 200,
        seed: int = 42,
    ) -> dict:
        """Generate a scenario trajectory and forecast future Phi values.

        Parameters
        ----------
        scenario : str
            One of PhiScenarioGenerator.SCENARIOS.
        history_len : int
            Number of time-steps used as model input history.
        seed : int
            Random seed for the scenario generator.

        Returns
        -------
        dict
            Keys: ``phi_predicted`` (list[float]), ``phi_actual`` (list[float]),
            ``scenario`` (str).
        """
        gen = PhiScenarioGenerator(seed=seed)
        df = gen.generate(scenario, length=history_len + self.pred_len)

        history = df.iloc[:history_len]
        future = df.iloc[history_len:]

        # Feature engineering: 36-column signal matrix via pipeline
        X = self.pipeline.fit_transform(history)

        # Take the last seq_len rows as the model input window
        X_seq = X[np.newaxis, -self.pipeline.seq_len :]  # [1, seq_len, 36]
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        phi_pred = self.model.predict_phi(X_tensor)
        phi_predicted = phi_pred[0, :, 0].numpy().tolist()
        phi_actual = future["phi"].values[: self.pred_len].tolist()

        return {
            "phi_predicted": phi_predicted,
            "phi_actual": phi_actual,
            "scenario": scenario,
        }
