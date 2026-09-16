"""Cross-dataset fidelity + TSTR utility evaluation.

Walk every ``DatasetSpec`` in ``src.datasets.registry``, fit each registered
synthesizer (plus ``auto`` / ``forest`` / ``knn`` / ``ensemble`` when those
families exist, and ``negative_control`` when importable), and write a
portable artifact used to declare a *general* winner.

CLI
---
::

    python -m src.eval_multidataset
    python -m src.eval_multidataset --n 500 --seed 0 --quick
    SYNTH_QUICK=1 python -m src.eval_multidataset

Arguments
    ``--n``        rows per DGP (default ``$SYNTH_N`` or 500)
    ``--seed``     master seed (default ``$SYNTH_SEED`` or 0)
    ``--quick``    cap ``n`` at 150 and drop slower methods
    ``--out``      JSON path (default ``$SYNTH_OUT`` or
                   ``artifacts/multidataset.json``)
    ``--test-size`` hold-out fraction for ``evaluate_model_utility`` (0.30)

``SYNTH_QUICK=1`` / ``true`` / ``yes`` / ``on`` is the same as ``--quick``:
``n = min(n, 150)`` and extras ``mixture`` / ``forest`` / ``knn`` /
``ensemble`` are skipped when at least one core method remains.

Failing dataset/method pairs are recorded with ``status="error"`` and
skipped; the run does not abort.

JSON schema (``artifacts/multidataset.json``)
--------------------------------------------
::

    {
      "schema_version": "1.0",
      "meta": {
        "n": int,
        "seed": int,
        "quick": bool,
        "test_size": float,
        "datasets": [str, ...],
        "methods": [str, ...],
        "registered_synthesizers": [str, ...],
        "gates": {"fidelity_min": 0.70, "tstr_gap_r2_max": 0.10},
        "output_path": str
      },
      "results": [
        {
          "dataset": str,
          "method": str,
          "status": "ok" | "error",
          "error": str | null,
          "fidelity_score": float | null,
          "utility_score": float | null,
          "combined_score": float | null,
          "tstr_gap_r2": float | null,
          "gates": {
            "fidelity_score_ge_0.70": bool,
            "tstr_gap_r2_le_0.10": bool,
            "passes": bool
          } | null,
          "n_real": int | null,
          "n_synth": int | null,
          "elapsed_sec": float
        },
        ...
      ],
      "matrix": {
        "rows": [str, ...],          # methods
        "cols": [str, ...],          # datasets
        "tstr_gap_r2":  {method: {dataset: float|null}},
        "combined_score": {method: {dataset: float|null}}
      },
      "passes": {
        method: {"n_pass": int, "n_eval": int, "datasets": [str, ...]}
      },
      "winner": {
        "method": str | null,
        "median_combined_score": float | null,
        "n_scored_datasets": int,
        "n_pass": int,
        "excluded": ["identity", "negative_control"]
      },
      "leaderboard": [
        {"method": str, "median_combined_score": float, "n_pass": int, ...}
      ]
    }

The printed matrix is methods × datasets; cells are ``tstr_gap_r2``
(lower is better).  A second table prints ``combined_score``.
The GENERAL winner is the non-control method with the best median
``combined_score = fidelity_score + utility_score`` across datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec

DEFAULT_N = 500
QUICK_N_CAP = 150
DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.30
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "multidataset.json"
SCHEMA_VERSION = "1.0"

GATE_FIDELITY = 0.70
GATE_TSTR_R2_GAP = 0.10  # looser than the single-DGP 0.08 bar

CONTROL_METHODS = frozenset({"identity", "negative_control"})
# Slower / extra families dropped in --quick when a core method remains.
QUICK_SKIP_METHODS = frozenset({"mixture", "forest", "knn", "ensemble"})
EXTRA_FAMILIES = ("auto", "forest", "knn", "ensemble")

_EXTRA_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "auto": {
        "modules": (
            "src.synthesizers.auto",
            "src.synthesizers.autosynth",
            "src.synthesizers.auto_synthesizer",
        ),
        "functions": (
            "synthesize_auto",
            "synthesize",
            "generate",
            "generate_synthetic",
        ),
        "classes": ("AutoSynthesizer", "Auto"),
    },
    "forest": {
        "modules": (
            "src.synthesizers.forest",
            "src.synthesizers.random_forest",
            "src.synthesizers.rf",
        ),
        "functions": (
            "synthesize_forest",
            "synthesize_rf",
            "synthesize",
            "generate",
        ),
        "classes": (
            "ForestSynthesizer",
            "RandomForestSynthesizer",
            "RFSynthesizer",
        ),
    },
    "knn": {
        "modules": (
            "src.synthesizers.knn",
            "src.synthesizers.knn_synth",
        ),
        "functions": ("synthesize_knn", "synthesize", "generate"),
        "classes": ("KNNSynthesizer", "KnnSynthesizer"),
    },
    "ensemble": {
        "modules": ("src.synthesizers.ensemble",),
        "functions": (
            "synthesize_ensemble",
            "synthesize",
            "generate",
        ),
        "classes": (
            "EnsembleSynthesizer",
            "MixtureCartEnsemble",
        ),
    },
}

SynthesizerFn = Callable[..., pd.DataFrame]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _fmt(x: Any, width: int = 8) -> str:
    if x is None:
        return "n/a".rjust(width)
    try:
        return f"{float(x):{width}.3f}"
    except (TypeError, ValueError):
        return str(x)[:width].rjust(width)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Dataset / synthesizer discovery
# ---------------------------------------------------------------------------


def _iter_specs() -> list[DatasetSpec]:
    """Yield registered ``DatasetSpec`` objects; tolerate name-or-spec APIs."""
    try:
        from src.datasets.registry import available, get
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not import dataset registry: {exc}", RuntimeWarning)
        return []

    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    try:
        items = list(available())
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"dataset registry.available() failed: {exc}", RuntimeWarning)
        return []

    for item in items:
        spec: Optional[DatasetSpec] = None
        if isinstance(item, DatasetSpec):
            spec = item
        elif isinstance(item, str):
            try:
                spec = get(item)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"dataset get({item!r}) failed: {exc}", RuntimeWarning)
                continue
        if spec is None or not isinstance(spec, DatasetSpec):
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    return specs


def _identity_synth(
    df: pd.DataFrame,
    n: int,
    seed: int,
    include_targets: bool = True,
) -> pd.DataFrame:
    del include_targets
    work = df
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if n == len(work):
        return work.reset_index(drop=True).copy()
    replace = n > len(work)
    idx = rng.choice(len(work), size=n, replace=replace)
    return work.iloc[idx].reset_index(drop=True)


def _try_import_negative_control() -> Optional[SynthesizerFn]:
    try:
        from src.evaluate import negative_control_synth
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"negative_control_synth not importable ({exc}); skipping.",
            RuntimeWarning,
        )
        return None
    if not callable(negative_control_synth):
        return None
    return negative_control_synth


def _try_import_identity() -> SynthesizerFn:
    try:
        from src.evaluate import identity_synth
    except Exception:
        return _identity_synth
    if callable(identity_synth):
        return identity_synth
    return _identity_synth


def _pick_from_module(module: Any, spec: dict[str, tuple[str, ...]]) -> Any:
    for attr in spec.get("functions", ()):
        obj = getattr(module, attr, None)
        if callable(obj) and not isinstance(obj, type):
            return obj
    for attr in spec.get("classes", ()):
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and (
            hasattr(obj, "fit") or hasattr(obj, "sample") or callable(obj)
        ):
            return obj
    return None


def _register_extra(name: str, synthesizers: dict[str, SynthesizerFn]) -> None:
    """Best-effort: pull ``auto`` / ``forest`` / ``knn`` / ``ensemble`` in."""
    if name in synthesizers:
        return
    try:
        from src.synthesizers import registry as synth_reg
    except Exception:
        synth_reg = None

    synthesizers_reg = getattr(synth_reg, "SYNTHESIZERS", {}) if synth_reg else {}
    adapt = getattr(synth_reg, "adapt", None) if synth_reg else None
    register = getattr(synth_reg, "register", None) if synth_reg else None
    try_register = getattr(synth_reg, "try_register", None) if synth_reg else None
    known_specs = getattr(synth_reg, "_SPECS", {}) if synth_reg else {}

    if name in synthesizers_reg:
        synthesizers[name] = synthesizers_reg[name]
        return

    # Only call try_register for families the registry already knows; extras
    # like auto/forest/knn/ensemble are discovered from their own modules.
    if callable(try_register) and name in known_specs:
        try:
            if try_register(name) and name in synthesizers_reg:
                synthesizers[name] = synthesizers_reg[name]
                return
        except Exception:
            pass

    spec = _EXTRA_SPECS.get(name)
    if spec is None or adapt is None:
        return

    import importlib

    for mod_path in spec["modules"]:
        try:
            module = importlib.import_module(mod_path)
        except Exception:
            continue
        picked = _pick_from_module(module, spec)
        if picked is None:
            continue
        try:
            wrapped = adapt(picked, name)
            if callable(register):
                register(name, picked)
            synthesizers[name] = wrapped
            return
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not adapt extra synthesizer '{name}' from {mod_path}: {exc}",
                RuntimeWarning,
            )


def _collect_methods(quick: bool) -> dict[str, SynthesizerFn]:
    methods: dict[str, SynthesizerFn] = {}

    methods["identity"] = _try_import_identity()

    try:
        from src.synthesizers.registry import SYNTHESIZERS
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not import synthesizer registry: {exc}", RuntimeWarning)
        SYNTHESIZERS = {}

    for name, fn in list(SYNTHESIZERS.items()):
        if callable(fn):
            methods[str(name)] = fn

    for extra in EXTRA_FAMILIES:
        _register_extra(extra, methods)
        if extra in SYNTHESIZERS and extra not in methods:
            methods[extra] = SYNTHESIZERS[extra]

    neg = _try_import_negative_control()
    if neg is not None:
        methods["negative_control"] = neg

    if quick:
        reduced = {k: v for k, v in methods.items() if k not in QUICK_SKIP_METHODS}
        core = [k for k in reduced if k not in CONTROL_METHODS]
        if core:
            return reduced
    return methods


def _align_schema(synth: pd.DataFrame, template: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in template.columns if c not in synth.columns]
    extra = [c for c in synth.columns if c not in template.columns]
    if missing:
        warnings.warn(
            f"Synthesizer '{name}' missing columns {missing}; filling with NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        for c in missing:
            synth[c] = np.nan
    if extra:
        synth = synth.drop(columns=extra)
    return synth.loc[:, list(template.columns)].copy().reset_index(drop=True)


def _call_synth(fn: SynthesizerFn, df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    attempts: Iterable[Callable[[], Any]] = (
        lambda: fn(df, n, seed, include_targets=True),
        lambda: fn(df, n=n, seed=seed, include_targets=True),
        lambda: fn(df, n, seed),
        lambda: fn(df, n=n, seed=seed),
        lambda: fn(df, n),
        lambda: fn(df),
    )
    last_err: Optional[BaseException] = None
    for attempt in attempts:
        try:
            out = attempt()
        except TypeError as exc:
            last_err = exc
            continue
        if isinstance(out, pd.DataFrame):
            return out
        last_err = TypeError(f"synthesizer returned {type(out).__name__}, not DataFrame")
    if last_err is not None:
        raise last_err
    raise TypeError("synthesizer produced no DataFrame")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    from src.metrics import compute_statistical_fidelity

    details = compute_statistical_fidelity(real, synth)
    if not isinstance(details, dict):
        raise TypeError("compute_statistical_fidelity must return a dict")
    return details


def _compute_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    test_size: float,
) -> dict[str, Any]:
    from src.metrics import evaluate_model_utility

    target_clf = spec.target_clf if spec.target_clf else "__no_y_class__"
    kwargs: dict[str, Any] = {
        "real": real,
        "synth": synth,
        "target_reg": spec.target_reg,
        "target_clf": target_clf,
        "test_size": float(test_size),
        "seed": int(seed),
    }
    try:
        return evaluate_model_utility(**kwargs)
    except TypeError:
        return evaluate_model_utility(real, synth)


def _extract_tstr_gap(util: dict[str, Any]) -> Optional[float]:
    gap = _as_float(util.get("tstr_gap_r2"))
    if gap is not None:
        return gap
    reg = util.get("regression")
    if isinstance(reg, dict):
        gap = _as_float(reg.get("tstr_gap_r2"))
        if gap is not None:
            return gap
        mean = reg.get("mean")
        if isinstance(mean, dict):
            gap = _as_float(mean.get("tstr_gap_r2"))
            if gap is not None:
                return gap
    return None


def _gates(fidelity_score: Optional[float], tstr_gap_r2: Optional[float]) -> dict[str, Any]:
    fid_ok = fidelity_score is not None and float(fidelity_score) >= GATE_FIDELITY
    gap_ok = tstr_gap_r2 is not None and float(tstr_gap_r2) <= GATE_TSTR_R2_GAP
    return {
        "fidelity_score_ge_0.70": bool(fid_ok),
        "tstr_gap_r2_le_0.10": bool(gap_ok),
        "passes": bool(fid_ok and gap_ok),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_pair(
    spec: DatasetSpec,
    method: str,
    fn: SynthesizerFn,
    real: pd.DataFrame,
    seed: int,
    test_size: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "dataset": spec.name,
        "method": method,
        "status": "ok",
        "error": None,
        "fidelity_score": None,
        "utility_score": None,
        "combined_score": None,
        "tstr_gap_r2": None,
        "gates": None,
        "n_real": int(len(real)),
        "n_synth": None,
        "elapsed_sec": None,
    }
    try:
        synth = _call_synth(fn, real, n=len(real), seed=seed)
        if not isinstance(synth, pd.DataFrame):
            raise TypeError(f"{method} returned {type(synth).__name__}, not DataFrame")
        synth = _align_schema(synth, real, method)
        row["n_synth"] = int(len(synth))

        fid = _compute_fidelity(real, synth)
        util = _compute_utility(real, synth, spec, seed=seed, test_size=test_size)

        fid_score = _as_float(fid.get("fidelity_score"))
        util_score = _as_float(util.get("utility_score"))
        gap = _extract_tstr_gap(util)
        row["fidelity_score"] = fid_score
        row["utility_score"] = util_score
        if fid_score is not None and util_score is not None:
            row["combined_score"] = float(fid_score) + float(util_score)
        row["tstr_gap_r2"] = gap
        row["gates"] = _gates(fid_score, gap)
    except Exception as exc:  # noqa: BLE001 — never crash the run
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        warnings.warn(
            f"{spec.name}/{method} failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    row["elapsed_sec"] = float(time.perf_counter() - t0)
    return row


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _summarize(
    results: list[dict[str, Any]],
    methods: list[str],
    datasets: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    matrix_gap: dict[str, dict[str, Optional[float]]] = {m: {} for m in methods}
    matrix_combo: dict[str, dict[str, Optional[float]]] = {m: {} for m in methods}
    for m in methods:
        for d in datasets:
            matrix_gap[m][d] = None
            matrix_combo[m][d] = None
    for row in results:
        if row.get("status") != "ok":
            continue
        m = row["method"]
        d = row["dataset"]
        if m in matrix_gap and d in matrix_gap[m]:
            matrix_gap[m][d] = _as_float(row.get("tstr_gap_r2"))
            matrix_combo[m][d] = _as_float(row.get("combined_score"))

    passes: dict[str, dict[str, Any]] = {}
    leaderboard: list[dict[str, Any]] = []
    for m in methods:
        scored = [
            r
            for r in results
            if r.get("method") == m and r.get("status") == "ok"
        ]
        passed = [
            r
            for r in scored
            if isinstance(r.get("gates"), dict) and r["gates"].get("passes")
        ]
        combos = [
            float(r["combined_score"])
            for r in scored
            if r.get("combined_score") is not None
        ]
        gaps = [
            float(r["tstr_gap_r2"])
            for r in scored
            if r.get("tstr_gap_r2") is not None
        ]
        entry = {
            "method": m,
            "n_eval": len(scored),
            "n_pass": len(passed),
            "datasets": [r["dataset"] for r in passed],
            "median_combined_score": _median(combos),
            "median_tstr_gap_r2": _median(gaps),
            "is_control": m in CONTROL_METHODS,
        }
        passes[m] = {
            "n_pass": entry["n_pass"],
            "n_eval": entry["n_eval"],
            "datasets": list(entry["datasets"]),
        }
        leaderboard.append(entry)

    contenders = [e for e in leaderboard if not e["is_control"] and e["median_combined_score"] is not None]
    contenders.sort(
        key=lambda e: (
            -(e["median_combined_score"] or float("-inf")),
            -(e["n_pass"] or 0),
            e["median_tstr_gap_r2"]
            if e["median_tstr_gap_r2"] is not None
            else float("inf"),
            e["method"],
        )
    )
    leaderboard.sort(
        key=lambda e: (
            e["is_control"],
            -(e["median_combined_score"] if e["median_combined_score"] is not None else float("-inf")),
            e["method"],
        )
    )

    if contenders:
        best = contenders[0]
        winner = {
            "method": best["method"],
            "median_combined_score": best["median_combined_score"],
            "n_scored_datasets": best["n_eval"],
            "n_pass": best["n_pass"],
            "excluded": sorted(CONTROL_METHODS),
        }
    else:
        winner = {
            "method": None,
            "median_combined_score": None,
            "n_scored_datasets": 0,
            "n_pass": 0,
            "excluded": sorted(CONTROL_METHODS),
        }

    matrix = {
        "rows": methods,
        "cols": datasets,
        "tstr_gap_r2": matrix_gap,
        "combined_score": matrix_combo,
    }
    return matrix, passes, winner, leaderboard


def _print_matrix(
    title: str,
    cell: dict[str, dict[str, Optional[float]]],
    methods: list[str],
    datasets: list[str],
    note: str,
) -> None:
    col_w = max(8, max((len(d) for d in datasets), default=8))
    name_w = max(16, max((len(m) for m in methods), default=16))
    header = f"{'method':<{name_w}}" + "".join(d.rjust(col_w + 1) for d in datasets)
    print()
    print("=" * max(78, len(header)))
    print(title)
    print(note)
    print("=" * max(78, len(header)))
    print(header)
    print("-" * max(78, len(header)))
    for m in methods:
        cells = "".join(_fmt(cell.get(m, {}).get(d), col_w).rjust(col_w + 1) for d in datasets)
        print(f"{m:<{name_w}}{cells}")
    print("=" * max(78, len(header)))


def _print_summary(
    winner: dict[str, Any],
    passes: dict[str, dict[str, Any]],
    methods: list[str],
    n_datasets: int,
) -> None:
    print()
    print("PASSES  (fidelity_score >= 0.70  and  tstr_gap_r2 <= 0.10)")
    print("-" * 60)
    for m in methods:
        block = passes.get(m) or {}
        n_pass = int(block.get("n_pass") or 0)
        n_eval = int(block.get("n_eval") or 0)
        ds = ", ".join(block.get("datasets") or []) or "—"
        print(f"  {m:<20} {n_pass}/{n_eval} of {n_datasets} datasets   [{ds}]")
    print()
    w = winner.get("method")
    med = winner.get("median_combined_score")
    if w:
        print(
            f"GENERAL winner: {w}  "
            f"(median combined_score={med:.4f} across "
            f"{winner.get('n_scored_datasets')} datasets; "
            f"excluded {', '.join(winner.get('excluded') or [])})"
        )
    else:
        print("GENERAL winner: none (no non-control method produced a score)")
    print()


def run_evaluation(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    quick: bool = False,
    out_path: Optional[Path] = None,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict[str, Any]:
    specs = _iter_specs()
    methods = _collect_methods(quick=quick)
    method_names = list(methods.keys())
    dataset_names = [s.name for s in specs]

    print(
        f"[eval_multidataset] n={n}  seed={seed}  quick={quick}  "
        f"datasets={dataset_names or '[]'}  methods={method_names or '[]'}",
        flush=True,
    )
    if not specs:
        warnings.warn("No DatasetSpec objects registered; writing an empty artifact.")
    if not methods:
        warnings.warn("No synthesizers collected; writing an empty artifact.")

    results: list[dict[str, Any]] = []
    for spec in specs:
        try:
            real = spec.sample(int(n), int(seed))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Dataset '{spec.name}' generate failed: {exc}", RuntimeWarning)
            for method in method_names:
                results.append(
                    {
                        "dataset": spec.name,
                        "method": method,
                        "status": "error",
                        "error": f"dataset_generate: {type(exc).__name__}: {exc}",
                        "fidelity_score": None,
                        "utility_score": None,
                        "combined_score": None,
                        "tstr_gap_r2": None,
                        "gates": None,
                        "n_real": None,
                        "n_synth": None,
                        "elapsed_sec": 0.0,
                    }
                )
            continue

        print(f"[eval_multidataset] dataset={spec.name}  rows={len(real)}", flush=True)
        for i, method in enumerate(method_names):
            fn = methods[method]
            job_seed = int(seed) + 10 + (sum(map(ord, method)) % 50) + i
            print(f"           method={method} …", flush=True)
            rec = _evaluate_pair(spec, method, fn, real, seed=job_seed, test_size=test_size)
            print(
                f"           status={rec['status']}  "
                f"fid={_fmt(rec.get('fidelity_score'))}  "
                f"util={_fmt(rec.get('utility_score'))}  "
                f"gap={_fmt(rec.get('tstr_gap_r2'))}  "
                f"elapsed={rec.get('elapsed_sec'):.2f}s",
                flush=True,
            )
            results.append(rec)

    matrix, passes, winner, leaderboard = _summarize(results, method_names, dataset_names)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "n": int(n),
            "seed": int(seed),
            "quick": bool(quick),
            "test_size": float(test_size),
            "datasets": dataset_names,
            "methods": method_names,
            "registered_synthesizers": [
                m for m in method_names if m not in CONTROL_METHODS
            ],
            "gates": {
                "fidelity_min": GATE_FIDELITY,
                "tstr_gap_r2_max": GATE_TSTR_R2_GAP,
            },
            "output_path": str(out_path) if out_path is not None else None,
        },
        "results": results,
        "matrix": matrix,
        "passes": passes,
        "winner": winner,
        "leaderboard": leaderboard,
    }

    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
        payload["meta"]["output_path"] = str(dest)
        print(f"[eval_multidataset] wrote {dest}", flush=True)

    _print_matrix(
        "TSTR gap R²  (rows=methods, cols=datasets; lower is better)",
        matrix["tstr_gap_r2"],
        method_names,
        dataset_names,
        "cell = tstr_gap_r2   (n/a = skipped / failed pair)",
    )
    _print_matrix(
        "combined_score = fidelity_score + utility_score  (higher is better)",
        matrix["combined_score"],
        method_names,
        dataset_names,
        "cell = combined_score",
    )
    _print_summary(winner, passes, method_names, n_datasets=len(dataset_names))
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate every DatasetSpec × synthesizer; write artifacts/multidataset.json."
        )
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Rows per DGP (default: $SYNTH_N or {DEFAULT_N})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Master seed (default: $SYNTH_SEED or {DEFAULT_SEED})",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Cap n at 150 and drop slower methods (also SYNTH_QUICK=1).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"JSON output path (default: $SYNTH_OUT or {DEFAULT_OUT})",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Hold-out fraction for evaluate_model_utility (default {DEFAULT_TEST_SIZE}).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    quick = bool(args.quick or _env_flag("SYNTH_QUICK"))
    n = args.n if args.n is not None else _env_int("SYNTH_N", DEFAULT_N)
    if quick:
        n = min(int(n), QUICK_N_CAP)
    seed = args.seed if args.seed is not None else _env_int("SYNTH_SEED", DEFAULT_SEED)
    out = args.out or os.environ.get("SYNTH_OUT") or str(DEFAULT_OUT)
    return run_evaluation(
        n=int(n),
        seed=int(seed),
        quick=quick,
        out_path=Path(out),
        test_size=float(args.test_size),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
