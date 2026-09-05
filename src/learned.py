"""Local learned encoders: supervised bottleneck and variational IB."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, task: str) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
        self.head = nn.Linear(d_out, 1 if task == "regression" else 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.head(h)


class _Decoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class _VIB(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, task: str) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, d_out)
        self.logvar = nn.Linear(hidden, d_out)
        self.head = nn.Linear(d_out, 1 if task == "regression" else 2)

    def stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.stats(x)
        std = torch.exp(0.5 * torch.clamp(logvar, -8, 4))
        z = mu + std * torch.randn_like(std)
        return z, self.head(z), mu, logvar  # z, pred, mu, logvar


class NumpyEncoder:
    def __init__(self, encode_fn) -> None:
        self._encode_fn = encode_fn

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self._encode_fn(X)

    def stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _prep_y(y: np.ndarray, task: str) -> tuple[torch.Tensor, float, float]:
    y = np.asarray(y)
    if task == "regression":
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-8)
        yt = torch.tensor(((y - y_mean) / y_std).astype(np.float32)).unsqueeze(1)
        return yt, y_mean, y_std
    return torch.tensor(y.astype(np.int64)), 0.0, 1.0


def _pred_loss(pred: torch.Tensor, yb: torch.Tensor, task: str) -> torch.Tensor:
    if task == "regression":
        return torch.mean((pred - yb) ** 2)
    return nn.functional.cross_entropy(pred, yb)


def train_bottleneck(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str,
    adversarial: bool,
    seed: int,
    hidden: int = 80,
    epochs: int = 22,
    batch_size: int = 256,
    lr: float = 1e-3,
    adv_lambda: float = 0.08,
) -> NumpyEncoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _MLPEncoder(X.shape[1], k, hidden, task)
    decoder = _Decoder(k, X.shape[1], hidden) if adversarial else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    opt_a = torch.optim.Adam(decoder.parameters(), lr=lr) if decoder is not None else None
    yt, _, _ = _prep_y(y, task)
    xt = torch.tensor(X.astype(np.float32))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(batch_size, len(X)), shuffle=True)
    model.train()
    if decoder is not None:
        decoder.train()
    for _ in range(epochs):
        for xb, yb in loader:
            h, pred = model(xb)
            if decoder is None or opt_a is None:
                loss = _pred_loss(pred, yb, task)
                opt.zero_grad()
                loss.backward()
                opt.step()
                continue
            opt_a.zero_grad()
            torch.mean((decoder(h.detach()) - xb) ** 2).backward()
            opt_a.step()
            h, pred = model(xb)
            loss = _pred_loss(pred, yb, task) - adv_lambda * torch.mean((decoder(h) - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    def encode_fn(arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model.encode(torch.tensor(arr.astype(np.float32))).numpy().astype(np.float64)

    return NumpyEncoder(encode_fn)


def train_vib(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str,
    beta: float,
    seed: int,
    hidden: int = 80,
    epochs: int = 24,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> NumpyEncoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _VIB(X.shape[1], k, hidden, task)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    yt, _, _ = _prep_y(y, task)
    xt = torch.tensor(X.astype(np.float32))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(batch_size, len(X)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            z, pred, mu, logvar = model(xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = _pred_loss(pred, yb, task) + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    class VIBEncoder(NumpyEncoder):
        def __init__(self) -> None:
            super().__init__(self.encode)

        def stats(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                mu, lv = model.stats(torch.tensor(arr.astype(np.float32)))
            return mu.numpy().astype(np.float64), lv.numpy().astype(np.float64)

        def encode(self, arr: np.ndarray) -> np.ndarray:
            mu, _ = self.stats(arr)
            return mu

    return VIBEncoder()
