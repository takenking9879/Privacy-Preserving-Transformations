"""Supervised (and optionally adversarial) predictive bottleneck encoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ENCODER_HYPERPARAMS


class MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
        self.head = nn.Linear(d_out, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.head(h)


class MLPDecoder(nn.Module):
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


class SupervisedBottleneck:
    """Fit locally on train only: X -> H -> y, optionally with an adversary."""

    def __init__(self, d_out: int, adversarial: bool = False, seed: int = 0) -> None:
        self.d_out = d_out
        self.adversarial = adversarial
        self.seed = seed
        self.hp = ENCODER_HYPERPARAMS
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.model: MLPEncoder | None = None
        self.device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedBottleneck":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)
        Xs = (X - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std

        d_in = Xs.shape[1]
        hidden = int(self.hp["hidden"])
        self.model = MLPEncoder(d_in, self.d_out, hidden)
        decoder = MLPDecoder(self.d_out, d_in, hidden) if self.adversarial else None
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.hp["lr"]),
            weight_decay=float(self.hp["weight_decay"]),
        )
        opt_a = (
            torch.optim.Adam(decoder.parameters(), lr=float(self.hp["lr"]))
            if decoder is not None
            else None
        )

        xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(
            TensorDataset(xt, yt),
            batch_size=int(self.hp["batch_size"]),
            shuffle=True,
        )
        epochs = int(self.hp["adv_epochs"] if self.adversarial else self.hp["epochs"])
        lam = float(self.hp["adv_lambda"]) if self.adversarial else 0.0
        self.model.train()
        if decoder is not None:
            decoder.train()

        for _ in range(epochs):
            for xb, yb in loader:
                h, pred = self.model(xb)
                loss_pred = torch.mean((pred - yb) ** 2)
                if decoder is None or opt_a is None:
                    opt.zero_grad()
                    loss_pred.backward()
                    opt.step()
                    continue

                x_hat = decoder(h.detach())
                loss_a = torch.mean((x_hat - xb) ** 2)
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                h, pred = self.model(xb)
                loss_pred = torch.mean((pred - yb) ** 2)
                loss_recon = torch.mean((decoder(h) - xb) ** 2)
                # Encoder: keep predictive utility, make reconstruction harder.
                loss_e = loss_pred - lam * loss_recon
                opt.zero_grad()
                loss_e.backward()
                opt.step()

        self.model.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.x_mean is not None and self.x_std is not None
        Xs = (X - self.x_mean) / self.x_std
        xt = torch.tensor(Xs, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            h = self.model.encode(xt).cpu().numpy()
        return h.astype(np.float64)
