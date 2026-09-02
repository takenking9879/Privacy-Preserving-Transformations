"""Privacy-preserving transforms. Parameters are fit on training data only."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from src.bottleneck import SupervisedBottleneck
from src.config import TransformSpec


def _random_orthogonal(dim: int, rng: np.random.RandomState) -> np.ndarray:
    if dim == 1:
        return np.ones((1, 1), dtype=np.float64)
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float64)


class PrivacyTransform:
    """Implements f_k(X) from the experimental protocol.

    Classical path:
        X -> G (marginal Gaussianization)
          -> W (symmetric whitening)
          -> Q_L (uniform quantization)
          -> P (optional orthonormal projection)
          -> R (secret orthogonal mixing)

    Learned path:
        X -> E_phi (supervised bottleneck)
          -> standardize
          -> Q_L
          -> R
    """

    def __init__(self, spec: TransformSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.gaussianizer: QuantileTransformer | None = None
        self.mu: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.W_inv: np.ndarray | None = None
        self.q_lo: float = -4.0
        self.q_hi: float = 4.0
        self.q_centers: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.encoder: SupervisedBottleneck | None = None
        self.h_mean: np.ndarray | None = None
        self.h_std: np.ndarray | None = None
        self.in_dim: int = 0
        self.out_dim: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PrivacyTransform":
        X = np.asarray(X, dtype=np.float64)
        self.in_dim = X.shape[1]
        spec = self.spec

        if spec.name == "raw":
            self.out_dim = self.in_dim
            return self

        if spec.bottleneck:
            if y is None:
                raise ValueError("Bottleneck transform requires y for supervised fit.")
            k = spec.output_dim(self.in_dim)
            self.encoder = SupervisedBottleneck(
                d_out=k, adversarial=spec.adversarial, seed=self.seed
            )
            self.encoder.fit(X, y)
            H = self.encoder.transform(X)
            self.h_mean = H.mean(axis=0)
            self.h_std = H.std(axis=0) + 1e-8
            Z = (H - self.h_mean) / self.h_std
            Z = self._fit_quantize(Z)
            self.out_dim = Z.shape[1]
            if spec.rotate:
                self.R = _random_orthogonal(self.out_dim, self.rng)
            return self

        Z = X
        if spec.gaussianize:
            n_q = min(1000, X.shape[0])
            self.gaussianizer = QuantileTransformer(
                n_quantiles=max(10, n_q),
                output_distribution="normal",
                subsample=X.shape[0],
                random_state=self.seed,
                copy=True,
            )
            Z = self.gaussianizer.fit_transform(Z)

        if spec.whiten:
            self.mu = Z.mean(axis=0)
            Zc = Z - self.mu
            cov = np.cov(Zc, rowvar=False, bias=False)
            cov = np.atleast_2d(cov)
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            u, s, _ = np.linalg.svd(cov, full_matrices=False)
            inv_sqrt = 1.0 / np.sqrt(s)
            sqrt_s = np.sqrt(s)
            self.W = (u * inv_sqrt) @ u.T
            self.W_inv = (u * sqrt_s) @ u.T
            Z = Zc @ self.W

        Z = self._fit_quantize(Z)

        d_cur = Z.shape[1]
        k = spec.output_dim(self.in_dim) if spec.proj_ratio is not None else d_cur
        if spec.proj_ratio is not None and k < d_cur:
            full = _random_orthogonal(d_cur, self.rng)
            self.P = full[:, :k]
            Z = Z @ self.P
            d_cur = k

        self.out_dim = d_cur
        if spec.rotate:
            self.R = _random_orthogonal(d_cur, self.rng)
        return self

    def _fit_quantize(self, Z: np.ndarray) -> np.ndarray:
        L = self.spec.quantize_levels
        if L is None:
            return Z
        edges = np.linspace(self.q_lo, self.q_hi, L + 1)
        self.q_centers = 0.5 * (edges[:-1] + edges[1:])
        return self._apply_quantize(Z)

    def _apply_quantize(self, Z: np.ndarray) -> np.ndarray:
        if self.spec.quantize_levels is None or self.q_centers is None:
            return Z
        L = self.spec.quantize_levels
        zc = np.clip(Z, self.q_lo, self.q_hi)
        width = (self.q_hi - self.q_lo) / L
        idx = np.floor((zc - self.q_lo) / width).astype(int)
        idx = np.clip(idx, 0, L - 1)
        return self.q_centers[idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        spec = self.spec
        if spec.name == "raw":
            return X.copy()

        if spec.bottleneck:
            assert self.encoder is not None and self.h_mean is not None and self.h_std is not None
            Z = (self.encoder.transform(X) - self.h_mean) / self.h_std
            Z = self._apply_quantize(Z)
            if self.R is not None:
                Z = Z @ self.R
            return Z

        Z = X
        if self.gaussianizer is not None:
            Z = self.gaussianizer.transform(Z)
        if self.W is not None and self.mu is not None:
            Z = (Z - self.mu) @ self.W
        Z = self._apply_quantize(Z)
        if self.P is not None:
            Z = Z @ self.P
        if self.R is not None:
            Z = Z @ self.R
        return Z

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_linear_part(self, Z: np.ndarray) -> np.ndarray:
        """Best-effort inverse ignoring quantization (for informed attacks)."""
        V = Z
        if self.R is not None:
            V = V @ self.R.T
        if self.P is not None:
            V = V @ np.linalg.pinv(self.P)
        if self.W_inv is not None and self.mu is not None:
            V = V @ self.W_inv + self.mu
        if self.gaussianizer is not None:
            V = self.gaussianizer.inverse_transform(V)
        return V
