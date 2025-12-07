import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMRegimeEncoder:
    """
    Simple and robust Gaussian HMM regime encoder.

    - Fixed number of states (n_states).
    - Fits on chosen features (e.g. daily returns).
    - Exposes hard regime labels (0..K-1).
    """

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False,
        )
        self.fitted = False

    def _prepare_X(self, df: pd.DataFrame, feature_cols):
        X = df[feature_cols].values.astype(np.float64)
        if not self.fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

    def fit(self, df: pd.DataFrame, feature_cols):
        X_scaled = self._prepare_X(df, feature_cols)
        self.hmm.fit(X_scaled)
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame, feature_cols):
        if not self.fitted:
            raise RuntimeError("HMMRegimeEncoder must be fitted before predict().")
        X_scaled = self._prepare_X(df, feature_cols)
        regimes = self.hmm.predict(X_scaled)
        return regimes

    def fit_predict(self, df: pd.DataFrame, feature_cols):
        self.fit(df, feature_cols)
        return self.predict(df, feature_cols)
