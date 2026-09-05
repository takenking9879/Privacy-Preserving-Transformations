import numpy as np

from src.datasets import make_banking_mixed
from src.experiment import split_indices
from src.keys import generate_trainer_keypair, unwrap_release, wrap_release
from src.transforms import build_transform


def test_capsule_hides_names_and_transforms_target():
    table = make_banking_mixed(n=220, seed=7, task="regression")
    idx = np.arange(150)
    tfm = build_transform("capsule", seed=7)
    tfm.fit(table, idx)
    out = tfm.apply(table, idx)
    assert all(c.startswith("c") for c in out.column_ids)
    assert not np.allclose(out.y_tilde, table.y[idx])
    np.testing.assert_allclose(tfm.inverse_y(out.y_tilde), table.y[idx], atol=1e-8)
    assert out.Z.shape[1] < table.predictive_frame().shape[1]


def test_trainer_envelope_roundtrip():
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(12, 5))
    y = rng.normal(size=12)
    kp = generate_trainer_keypair()
    blob = wrap_release(Z, y, kp.public_pem)
    Z2, y2 = unwrap_release(blob, kp.private_pem)
    np.testing.assert_allclose(Z2, Z)
    np.testing.assert_allclose(y2, y)


def test_wrong_private_key_cannot_open_release():
    rng = np.random.RandomState(1)
    Z = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    kp1 = generate_trainer_keypair()
    kp2 = generate_trainer_keypair()
    blob = wrap_release(Z, y, kp1.public_pem)
    try:
        unwrap_release(blob, kp2.private_pem)
        opened = True
    except Exception:
        opened = False
    assert opened is False
