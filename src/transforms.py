"""Candidate privacy-preserving transforms.

Every transform is fit on the training split only. The training party receives
anonymous numeric columns and a target that has already been passed through an
invertible TargetMap. Predictions are mapped back with TargetMap.inverse.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import QuantileTransformer

from src.encode import PublicEncoder
from src.schema import RawTable, TargetMap


def _random_orthogonal(dim: int, rng: np.random.RandomState) -> np.ndarray:
    if dim <= 1:
        return np.ones((max(dim, 1), max(dim, 1)), dtype=np.float64)
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float64)


def _make_target_map(y: np.ndarray, task: str, rng: np.random.RandomState) -> TargetMap:
    if task == "regression":
        scale = float(rng.uniform(0.35, 2.7) * rng.choice([-1.0, 1.0]))
        shift = float(rng.normal(0.0, 3.0))
        return TargetMap(task="regression", scale=scale, shift=shift)
    classes = np.unique(y.astype(int))
    perm = rng.permutation(len(classes))
    fwd = {int(c): int(perm[i]) for i, c in enumerate(classes)}
    inv = {v: k for k, v in fwd.items()}
    return TargetMap(task="classification", class_forward=fwd, class_inverse=inv)


def _hmac_bucket(key: bytes, value: object, buckets: int) -> int:
    text = "" if value is None else str(value)
    digest = hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "little") % buckets


@dataclass
class TransformResult:
    Z: np.ndarray
    y_tilde: np.ndarray
    column_ids: list[str]
    target_map: TargetMap
    extras: dict = field(default_factory=dict)


class BaseTransform:
    name = "base"
    requires_y = False
    family = "unknown"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.encoder = PublicEncoder([])
        self.target_map = TargetMap(task="regression")
        self.out_dim = 0
        self.notes: dict[str, Any] = {}

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "BaseTransform":
        raise NotImplementedError

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.target_map.transform(y)

    def inverse_y(self, y_tilde: np.ndarray) -> np.ndarray:
        return self.target_map.inverse(y_tilde)

    def apply(self, table: RawTable, idx: np.ndarray) -> TransformResult:
        frame = table.predictive_frame().iloc[idx]
        Z = self.transform_X(frame)
        y_tilde = self.transform_y(table.y[idx])
        ids = [f"c{j:03d}" for j in range(Z.shape[1])]
        return TransformResult(Z=Z, y_tilde=y_tilde, column_ids=ids, target_map=self.target_map)

    def _fit_public_encoder(self, table: RawTable, train_idx: np.ndarray) -> np.ndarray:
        self.encoder = PublicEncoder(table.columns)
        return self.encoder.fit_transform(table.predictive_frame().iloc[train_idx])

    def _init_target(self, table: RawTable, train_idx: np.ndarray, transform_target: bool) -> None:
        if transform_target:
            self.target_map = _make_target_map(table.y[train_idx], table.task, self.rng)
        else:
            if table.task == "regression":
                self.target_map = TargetMap(task="regression", scale=1.0, shift=0.0)
            else:
                classes = np.unique(table.y[train_idx].astype(int))
                fwd = {int(c): int(c) for c in classes}
                self.target_map = TargetMap(
                    task="classification", class_forward=fwd, class_inverse=fwd
                )


class IdentityTransform(BaseTransform):
    """Public encoding, original target. Utility ceiling, zero privacy."""

    name = "identity"
    family = "baseline"

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "IdentityTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, transform_target=False)
        self.out_dim = X.shape[1]
        self.notes = {"invertible": True, "prediction_remap": "identity"}
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        return self.encoder.transform(frame)


class SecretAffineTransform(BaseTransform):
    """Per-column secret scale/shift + permutation. Absorbed by linear models.

    Phase-1 utility ceiling among *named-hiding* maps: no nonlinear warp.
    Rank and known-pair attacks remain trivial — that is measured, not assumed.
    """

    name = "secret_affine"
    family = "typed"

    def __init__(self, seed: int = 0, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.transform_target = transform_target
        self.scales: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.perm: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "SecretAffineTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        d = X.shape[1]
        self.scales = self.rng.uniform(0.4, 3.5, size=d) * self.rng.choice([-1.0, 1.0], size=d)
        self.shifts = self.rng.normal(0.0, 2.0, size=d)
        self.perm = self.rng.permutation(d)
        self.out_dim = d
        self.notes = {
            "invertible": True,
            "preserves": ["linear_span", "rank_order", "tree_splits"],
            "destroys": ["units", "column_order", "signed_orientation"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.scales is not None and self.shifts is not None and self.perm is not None
        X = self.encoder.transform(frame)
        return (X * self.scales + self.shifts)[:, self.perm]


class StdRotTransform(BaseTransform):
    """Standardize then secret-rotate. Linear-friendly mixing without quantile warp."""

    name = "std_rot"
    family = "classical"

    def __init__(self, seed: int = 0, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.transform_target = transform_target
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "StdRotTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        self.R = _random_orthogonal(X.shape[1], self.rng)
        self.out_dim = X.shape[1]
        self.notes = {
            "invertible": True,
            "preserves": ["linear_span", "second_order_geometry"],
            "destroys": ["column_identity", "axis_aligned_splits"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.mean is not None and self.std is not None and self.R is not None
        return ((self.encoder.transform(frame) - self.mean) / self.std) @ self.R


class GaussTransform(BaseTransform):
    """Per-column quantile Gaussianization. Hides units/marginals, keeps rank."""

    name = "gauss"
    family = "classical"

    def __init__(self, seed: int = 0, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.transform_target = transform_target
        self.qt: QuantileTransformer | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "GaussTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        n_q = min(1000, max(10, X.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=X.shape[0],
            random_state=self.seed,
            copy=True,
        )
        self.qt.fit(X)
        self.out_dim = X.shape[1]
        self.notes = {
            "invertible": "partial (quantile inverse, cats/strings lossy)",
            "preserves": ["rank_order", "pairwise_comonotonicity"],
            "destroys": ["scale", "units", "marginal_shape"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None
        return self.qt.transform(self.encoder.transform(frame))


class GaussWhiteRotTransform(BaseTransform):
    """Gaussianize, ZCA-whiten, secret orthogonal mix. Prior-work classical map."""

    name = "gauss_white_rot"
    family = "classical"

    def __init__(self, seed: int = 0, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.transform_target = transform_target
        self.qt: QuantileTransformer | None = None
        self.mu: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "GaussWhiteRotTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        n_q = min(1000, max(10, X.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=X.shape[0],
            random_state=self.seed,
            copy=True,
        )
        U = self.qt.fit_transform(X)
        self.mu = U.mean(axis=0)
        cov = np.cov(U - self.mu, rowvar=False, bias=False)
        cov = np.atleast_2d(cov) + 1e-6 * np.eye(U.shape[1])
        u, s, _ = np.linalg.svd(cov, full_matrices=False)
        self.W = (u * (1.0 / np.sqrt(s))) @ u.T
        self.R = _random_orthogonal(U.shape[1], self.rng)
        self.out_dim = U.shape[1]
        self.notes = {
            "invertible": True,
            "preserves": ["second_order_geometry_up_to_rotation"],
            "destroys": ["axis_aligned_splits", "column_identity", "marginals"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None and self.mu is not None and self.W is not None and self.R is not None
        U = self.qt.transform(self.encoder.transform(frame))
        return ((U - self.mu) @ self.W) @ self.R


class KeyedMonotoneTransform(BaseTransform):
    """Secret per-column monotone map + column permutation + dummy noise columns.

    Tree models that split on order should retain nearly all utility.
    Linear models see scrambled scales. Known-pair correlation matching
    recovers the permutation cheaply — that is a measured weakness.
    """

    name = "keyed_monotone"
    family = "typed"

    def __init__(self, seed: int = 0, n_dummy: int = 3, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.n_dummy = n_dummy
        self.transform_target = transform_target
        self.qt: QuantileTransformer | None = None
        self.scales: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.perm: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "KeyedMonotoneTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        n_q = min(1000, max(10, X.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="uniform",
            subsample=X.shape[0],
            random_state=self.seed,
            copy=True,
        )
        self.qt.fit(X)
        d = X.shape[1]
        self.scales = self.rng.uniform(0.4, 3.5, size=d) * self.rng.choice([-1.0, 1.0], size=d)
        self.shifts = self.rng.normal(0.0, 2.0, size=d)
        self.perm = self.rng.permutation(d + self.n_dummy)
        self.out_dim = d + self.n_dummy
        self.notes = {
            "invertible": "partial (dummies irreversible; monotone inverse exists)",
            "preserves": ["within_column_order"],
            "destroys": ["units", "column_order", "string_literals"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None and self.scales is not None and self.shifts is not None
        assert self.perm is not None
        U = self.qt.transform(self.encoder.transform(frame))
        Zc = U * self.scales + self.shifts
        dummy = self.rng.normal(size=(len(Zc), self.n_dummy))
        # Dummy columns must be deterministic given X so attacks are well-defined.
        # Re-seed from row signatures of the uniform ranks.
        dummy = np.sin(U[:, :1] * 17.3 + np.arange(self.n_dummy)[None, :] * 3.1)
        dummy += 0.15 * np.cos(U[:, min(1, U.shape[1] - 1) : min(2, U.shape[1])] * 9.1)
        Z = np.hstack([Zc, dummy])
        return Z[:, self.perm]


class TypedKeyedTransform(BaseTransform):
    """Type-aware owner-side map: HMAC buckets for strings, keyed codes for
    categoricals, Gaussianization for numerics, secret permutation.
    """

    name = "typed_keyed"
    family = "typed"

    def __init__(self, seed: int = 0, string_buckets: int = 32, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.string_buckets = string_buckets
        self.transform_target = transform_target
        self.hmac_key: bytes = b""
        self.num_names: list[str] = []
        self.cat_names: list[str] = []
        self.str_names: list[str] = []
        self.cat_maps: dict[str, dict[str, float]] = {}
        self.qt: QuantileTransformer | None = None
        self.perm: np.ndarray | None = None
        self.num_scale: np.ndarray | None = None
        self.num_shift: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "TypedKeyedTransform":
        self._init_target(table, train_idx, self.transform_target)
        self.hmac_key = self.rng.randint(0, 256, size=16).astype(np.uint8).tobytes()
        cols = [c for c in table.columns if not c.identifier]
        self.num_names = [c.name for c in cols if c.kind == "numeric"]
        self.cat_names = [c.name for c in cols if c.kind == "categorical"]
        self.str_names = [c.name for c in cols if c.kind == "string"]
        frame = table.predictive_frame().iloc[train_idx]
        self.cat_maps = {}
        for name in self.cat_names:
            levels = sorted(frame[name].astype(str).unique().tolist())
            codes = self.rng.permutation(len(levels)).astype(np.float64)
            self.cat_maps[name] = {lev: float(codes[i]) for i, lev in enumerate(levels)}
        Xn = self._numeric_block(frame)
        n_q = min(1000, max(10, Xn.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=Xn.shape[0],
            random_state=self.seed,
            copy=True,
        )
        self.qt.fit(Xn)
        d = Xn.shape[1]
        self.num_scale = self.rng.uniform(0.5, 2.0, size=d)
        self.num_shift = self.rng.normal(0.0, 1.0, size=d)
        self.perm = self.rng.permutation(d)
        self.out_dim = d
        self.notes = {
            "invertible": "partial (string buckets many-to-one; cats keyed)",
            "preserves": ["numeric_rank", "cat_equality"],
            "destroys": ["string_literals", "column_names", "cat_original_codes"],
        }
        return self

    def _numeric_block(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.num_names:
            blocks.append(frame.loc[:, self.num_names].to_numpy(dtype=np.float64))
        if self.cat_names:
            cat = np.empty((len(frame), len(self.cat_names)), dtype=np.float64)
            for j, name in enumerate(self.cat_names):
                mapping = self.cat_maps[name]
                cat[:, j] = [mapping.get(str(v), -1.0) for v in frame[name].tolist()]
            blocks.append(cat)
        if self.str_names:
            hashed = np.empty((len(frame), len(self.str_names)), dtype=np.float64)
            for j, name in enumerate(self.str_names):
                hashed[:, j] = [
                    float(_hmac_bucket(self.hmac_key, v, self.string_buckets))
                    for v in frame[name].tolist()
                ]
            blocks.append(hashed)
        return np.hstack(blocks) if blocks else np.zeros((len(frame), 0))

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None and self.perm is not None
        assert self.num_scale is not None and self.num_shift is not None
        U = self.qt.transform(self._numeric_block(frame))
        return (U * self.num_scale + self.num_shift)[:, self.perm]


class SupervisedBottleneckTransform(BaseTransform):
    """Local X->H->y encoder; H is what the trainer sees (prior-work family)."""

    name = "bn_adv"
    requires_y = True
    family = "learned"

    def __init__(
        self,
        seed: int = 0,
        ratio: float = 0.6,
        adversarial: bool = True,
        noise: float = 0.0,
        transform_target: bool = True,
    ) -> None:
        super().__init__(seed)
        self.ratio = ratio
        self.adversarial = adversarial
        self.noise = noise
        self.transform_target = transform_target
        self.model = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.h_mean: np.ndarray | None = None
        self.h_std: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "SupervisedBottleneckTransform":
        from src.learned import train_bottleneck

        X = self._fit_public_encoder(table, train_idx)
        y = table.y[train_idx]
        self._init_target(table, train_idx, self.transform_target)
        k = max(4, int(round(self.ratio * X.shape[1])))
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        Xs = (X - self.x_mean) / self.x_std
        self.model = train_bottleneck(
            Xs,
            y,
            k,
            task=table.task,
            adversarial=self.adversarial,
            seed=self.seed,
        )
        H = self.model.encode(Xs)
        self.h_mean = H.mean(axis=0)
        self.h_std = H.std(axis=0) + 1e-8
        self.R = _random_orthogonal(k, self.rng)
        self.out_dim = k
        self.notes = {
            "invertible": False,
            "preserves": ["I(H;Y) by construction"],
            "destroys": ["features orthogonal to Y", "column_identity"],
            "local_training_required": True,
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.x_mean is not None
        X = self.encoder.transform(frame)
        Xs = (X - self.x_mean) / self.x_std
        H = self.model.encode(Xs)
        Z = (H - self.h_mean) / self.h_std
        if self.noise > 0:
            # Deterministic structured noise from a keyed hash of the row so
            # train/test are reproducible without storing per-row draws.
            keyed = np.sin(Xs @ self.rng.normal(size=(Xs.shape[1], Z.shape[1])))
            Z = Z + self.noise * keyed
        return Z @ self.R


class VIBTransform(BaseTransform):
    """Variational information bottleneck: stochastic Z = μ(X)+σ(X)ε, trained
    to keep I(Z;Y) while shrinking I(Z;X). Stronger theoretically than a
    deterministic bottleneck against exact inversion.
    """

    name = "vib"
    requires_y = True
    family = "learned"

    def __init__(
        self,
        seed: int = 0,
        ratio: float = 0.5,
        beta: float = 1e-3,
        transform_target: bool = True,
        sample_at_transform: bool = False,
    ) -> None:
        super().__init__(seed)
        self.ratio = ratio
        self.beta = beta
        self.transform_target = transform_target
        self.sample_at_transform = sample_at_transform
        self.model = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "VIBTransform":
        from src.learned import train_vib

        X = self._fit_public_encoder(table, train_idx)
        y = table.y[train_idx]
        self._init_target(table, train_idx, self.transform_target)
        k = max(4, int(round(self.ratio * X.shape[1])))
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        Xs = (X - self.x_mean) / self.x_std
        self.model = train_vib(Xs, y, k, task=table.task, beta=self.beta, seed=self.seed)
        self.R = _random_orthogonal(k, self.rng)
        self.out_dim = k
        self.notes = {
            "invertible": False,
            "preserves": ["I(Z;Y) up to beta"],
            "destroys": ["fine-grained X via KL bottleneck"],
            "local_training_required": True,
            "stochastic": self.sample_at_transform,
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.x_mean is not None
        X = self.encoder.transform(frame)
        Xs = (X - self.x_mean) / self.x_std
        mu, logvar = self.model.stats(Xs)
        if self.sample_at_transform:
            # Reproducible noise from a keyed function of X (not stored draws).
            eps = np.tanh(Xs @ (self.rng.normal(size=(Xs.shape[1], mu.shape[1]))))
            Z = mu + np.exp(0.5 * logvar) * eps
        else:
            Z = mu
        return Z @ self.R


class RFFTransform(BaseTransform):
    """Secret random Fourier features. Nonlinear mixing; linear inversion fails."""

    name = "rff"
    family = "nonlinear"

    def __init__(self, seed: int = 0, ratio: float = 1.5, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.ratio = ratio
        self.transform_target = transform_target
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "RFFTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        k = max(8, int(round(self.ratio * X.shape[1])))
        self.W = self.rng.normal(scale=0.35, size=(X.shape[1], k))
        self.b = self.rng.uniform(0, 2 * np.pi, size=k)
        self.out_dim = k
        self.notes = {
            "invertible": False,
            "preserves": ["approximate_shift_invariant_kernel"],
            "destroys": ["original_coordinates", "column_identity"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.W is not None and self.b is not None
        Xs = (self.encoder.transform(frame) - self.mean) / self.std
        return np.sqrt(2.0 / self.W.shape[1]) * np.cos(Xs @ self.W + self.b)


class MicroaggRotTransform(BaseTransform):
    """Replace each row with a nearby centroid, then secret-rotate.

    Caps exact row recovery (many-to-one) at a utility cost.
    """

    name = "microagg_rot"
    family = "lossy"

    def __init__(self, seed: int = 0, n_clusters: int = 80, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.n_clusters = n_clusters
        self.transform_target = transform_target
        self.qt: QuantileTransformer | None = None
        self.kmeans: MiniBatchKMeans | None = None
        self.R: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "MicroaggRotTransform":
        X = self._fit_public_encoder(table, train_idx)
        self._init_target(table, train_idx, self.transform_target)
        n_q = min(1000, max(10, X.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=X.shape[0],
            random_state=self.seed,
            copy=True,
        )
        U = self.qt.fit_transform(X)
        k = min(self.n_clusters, max(8, len(U) // 15))
        self.kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.seed, n_init=5, batch_size=256)
        self.kmeans.fit(U)
        self.mean = U.mean(axis=0)
        self.std = U.std(axis=0) + 1e-8
        self.R = _random_orthogonal(U.shape[1], self.rng)
        self.out_dim = U.shape[1]
        self.notes = {
            "invertible": False,
            "preserves": ["coarse_neighborhoods"],
            "destroys": ["within_cluster_identity"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None and self.kmeans is not None and self.R is not None
        U = self.qt.transform(self.encoder.transform(frame))
        labels = self.kmeans.predict(U)
        C = self.kmeans.cluster_centers_[labels]
        return ((C - self.mean) / self.std) @ self.R


class NoisyGaussRotTransform(BaseTransform):
    """Classical secret rotation plus isotropic noise (local-DP flavour)."""

    name = "noisy_gauss_rot"
    family = "classical"

    def __init__(self, seed: int = 0, noise: float = 0.35, transform_target: bool = True) -> None:
        super().__init__(seed)
        self.noise = noise
        self.transform_target = transform_target
        self.inner = GaussWhiteRotTransform(seed=seed, transform_target=transform_target)
        self.proj: np.ndarray | None = None

    def fit(self, table: RawTable, train_idx: np.ndarray) -> "NoisyGaussRotTransform":
        self.inner.fit(table, train_idx)
        self.encoder = self.inner.encoder
        self.target_map = self.inner.target_map
        self.out_dim = self.inner.out_dim
        # Deterministic noise directions (keyed).
        self.proj = self.rng.normal(size=(self.out_dim, self.out_dim))
        self.notes = {
            "invertible": False,
            "preserves": ["approximate_geometry"],
            "destroys": ["exact_values"],
        }
        return self

    def transform_X(self, frame: pd.DataFrame) -> np.ndarray:
        Z = self.inner.transform_X(frame)
        Xn = self.encoder.transform(frame)
        noise = np.tanh(Xn @ self.proj[: Xn.shape[1], : Z.shape[1]])
        return Z + self.noise * noise


def build_transform(name: str, seed: int) -> BaseTransform:
    catalog = {
        "identity": lambda: IdentityTransform(seed),
        "secret_affine": lambda: SecretAffineTransform(seed),
        "std_rot": lambda: StdRotTransform(seed),
        "gauss": lambda: GaussTransform(seed),
        "gauss_white_rot": lambda: GaussWhiteRotTransform(seed),
        "keyed_monotone": lambda: KeyedMonotoneTransform(seed),
        "typed_keyed": lambda: TypedKeyedTransform(seed),
        "bn_adv": lambda: SupervisedBottleneckTransform(seed, adversarial=True, noise=0.0),
        "bn_noisy": lambda: SupervisedBottleneckTransform(seed, adversarial=True, noise=0.25),
        "vib": lambda: VIBTransform(seed, beta=1e-3, sample_at_transform=False),
        "vib_stoch": lambda: VIBTransform(seed, beta=5e-3, sample_at_transform=True),
        "rff": lambda: RFFTransform(seed),
        "microagg_rot": lambda: MicroaggRotTransform(seed),
        "noisy_gauss_rot": lambda: NoisyGaussRotTransform(seed, noise=0.35),
    }
    if name not in catalog:
        raise KeyError(f"Unknown transform {name}")
    tfm = catalog[name]()
    tfm.name = name
    return tfm


CANDIDATE_TRANSFORMS = [
    "identity",
    "secret_affine",
    "std_rot",
    "gauss",
    "gauss_white_rot",
    "keyed_monotone",
    "typed_keyed",
    "bn_adv",
    "bn_noisy",
    "vib",
    "vib_stoch",
    "rff",
    "microagg_rot",
    "noisy_gauss_rot",
]

QUICK_TRANSFORMS = [
    "identity",
    "gauss",
    "gauss_white_rot",
    "keyed_monotone",
    "typed_keyed",
    "bn_adv",
    "vib",
    "rff",
]
