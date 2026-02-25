"""PhiPipeline — preprocessing for Phi construct time-series."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.forecasting.signals import PhiSignalProcessor
from src.inference.welfare_scoring import ALL_CONSTRUCTS


class PhiPipeline:
    """Transform raw construct time-series into scaled, windowed sequences."""

    def __init__(self, seq_len: int = 100, window: int = 20):
        self.seq_len = seq_len
        self.window = window
        self.scaler = RobustScaler()
        self.fitted = False
        self.feature_names: list[str] = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return PhiSignalProcessor.compute_all_signals(df, window=self.window)

    def fit(self, df: pd.DataFrame) -> "PhiPipeline":
        features = self.compute_features(df)
        self.feature_names = list(features.columns)
        self.scaler.fit(features.values)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        features = self.compute_features(df)
        return self.scaler.transform(features[self.feature_names].values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def create_sequences(self, X, y=None, stride=1):
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

    def process(self, df, labels=None, fit=True, stride=1):
        X = self.fit_transform(df) if fit else self.transform(df)
        return self.create_sequences(X, labels, stride)
