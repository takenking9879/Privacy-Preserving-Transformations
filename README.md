# Privacy-Preserving Transformations

Utility-preserving privacy / obfuscation for heterogeneous tabular data:

```
X_raw  ->  X_transformed
```

A model is trained **only** on the transformed table. Predictions are mapped
back to the original target space with an invertible owner-side map `g`.
Column names and semantics are not sent to the training party.

This is **not encryption**. Known-pair inversion is treated as the binding
attack, not as an afterthought.

## What is in this repo

| Path | Role |
| --- | --- |
| `src/transforms.py` | Candidate maps (classical, typed, VIB, RFF, microagg, …) |
| `src/models.py` | Linear, HGB, MLP, k-NN with frozen hyperparameters |
| `src/attacks.py` | Attacker A (known-pair) and Attacker B (black-box) |
| `src/experiment.py` | Shared splits, utility, both attackers, Pareto selection |
| `RESEARCH.md` | Literature and why families were kept or rejected |
| `REPORT.md` | Full 11-part deliverable (hand-written; not overwritten by `run.py`) |
| `tests/` | Permanent regression tests |

## Protocol

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run.py --quick    # smoke
python run.py            # full grid (2 seeds, 3 datasets, 14 transforms, 4 models)
```

Priority used for selection:

```
maximum practical privacy
        subject to
minimal predictive utility loss
```

## Architecture (recommended)

```
owner holds schema, keys, g
  strings     -> HMAC buckets
  categoricals -> keyed codes
  numerics    -> quantile map
  optional    -> local VIB / bottleneck if y is available
  then        -> secret permutation / rotation
  target      -> invertible g(y)
trainer sees  (Z with ids c000…, y_tilde)
owner returns g^{-1}(M(Z_new))
```

Default map when the owner cannot train a local encoder: **`typed_keyed`**.
If labelled data can stay on-prem for an encoder fit: **`vib`**.
Neither resists known-pair recovery of the attributes that cause `y`.
See [`REPORT.md`](REPORT.md).
