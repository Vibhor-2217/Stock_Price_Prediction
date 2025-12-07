# regime_encoder/vb_hmm_regime_encoder.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class VBStyleRegimeEncoder:
    """
    VB-style Gaussian HMM regime encoder (approximate).

    Key ideas:
      - Automatically selects number of regimes K using BIC
        over a range [k_min, k_max].
      - Returns BOTH hard labels (argmax) and soft posteriors
        (gamma_tk = P(z_t = k | x_1:T)).
      - Interface is similar to a VB-HMM: latent regimes with
        model evidence-based K selection and posterior probabilities.

    Internally uses hmmlearn's GaussianHMM (EM/MLE), but the
    selection & posterior behavior is VB-like.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 6,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        assert k_min >= 2
        assert k_max >= k_min
        self.k_min = k_min
        self.k_max = k_max
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.hmm: GaussianHMM | None = None
        self.n_states: int | None = None
        self.fitted = False

    # ---------- internal helpers ----------

    def _prepare_X(self, df: pd.DataFrame, feature_cols):
        X = df[feature_cols].values.astype(np.float64)
        if not self.fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

    @staticmethod
    def _num_params(k: int, d: int) -> int:
        """
        Approximate number of free parameters in a K-state, d-dim
        Gaussian HMM with full covariances:

        - initial probs: k-1
        - transitions: k*(k-1)
        - means: k*d
        - covariances (full): k * d*(d+1)/2
        """
        pi_params = k - 1
        trans_params = k * (k - 1)
        mean_params = k * d
        cov_params = k * d * (d + 1) // 2
        return pi_params + trans_params + mean_params + cov_params

    def _select_model(self, X_scaled: np.ndarray):
        N, d = X_scaled.shape
        best_bic = np.inf
        best_model = None
        best_k = None

        for k in range(self.k_min, self.k_max + 1):
            hmm = GaussianHMM(
                n_components=k,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False,
            )
            hmm.fit(X_scaled)
            logL = hmm.score(X_scaled)  # log-likelihood
            p = self._num_params(k, d)
            bic = -2.0 * logL + p * np.log(N)

            # print(f"[DEBUG] k={k}, logL={logL:.1f}, p={p}, BIC={bic:.1f}")

            if bic < best_bic:
                best_bic = bic
                best_model = hmm
                best_k = k

        return best_model, best_k

    # ---------- public API ----------

    def fit(self, df: pd.DataFrame, feature_cols):
        """
        Fit HMM and select number of regimes K using BIC.
        """
        X_scaled = self._prepare_X(df, feature_cols)
        best_model, best_k = self._select_model(X_scaled)

        self.hmm = best_model
        self.n_states = int(best_k)
        self.fitted = True
        return self

    def predict_soft(self, df: pd.DataFrame, feature_cols):
        """
        Return posterior regime probabilities gamma_tk (soft regimes).

        Output: array of shape [N, K] with rows summing to 1.
        """
        if not self.fitted or self.hmm is None:
            raise RuntimeError("VBStyleRegimeEncoder must be fitted before predict_soft().")

        X_scaled = self._prepare_X(df, feature_cols)
        gamma = self.hmm.predict_proba(X_scaled)  # [N, K]
        return gamma

    def predict_hard(self, df: pd.DataFrame, feature_cols):
        """
        Return hard regime labels (argmax over soft posteriors).
        """
        gamma = self.predict_soft(df, feature_cols)
        regimes = np.argmax(gamma, axis=1)
        return regimes

    def fit_predict_soft(self, df: pd.DataFrame, feature_cols):
        self.fit(df, feature_cols)
        return self.predict_soft(df, feature_cols)

    def fit_predict_hard(self, df: pd.DataFrame, feature_cols):
        self.fit(df, feature_cols)
        return self.predict_hard(df, feature_cols)
