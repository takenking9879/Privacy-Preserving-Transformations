"""Write ``reports/RESULTS.md`` from evaluation artifacts and optional CSVs.

CLI::

    python -m src.report

Missing ``artifacts/evaluation.json`` or ``data/*.csv`` is tolerated: the
markdown still records what was found and which success gates could not
be scored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

# ---------------------------------------------------------------------------
# Success gates (keep these numbers in lockstep with evaluate / tests)
# ---------------------------------------------------------------------------
GATE_TSTR_GAP_MAX = 0.08
GATE_FIDELITY_MIN = 0.70
GATE_NEG_FIDELITY_MAX = 0.55
GATE_NEG_TSTR_GAP_MIN = 0.15

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "artifacts" / "evaluation.json"
DEFAULT_DATA = ROOT / "data"
DEFAULT_REPORT = ROOT / "reports" / "RESULTS.md"
DEFAULT_PLOTS = ROOT / "reports" / "figures"

_NEG_NAME_RE = re.compile(
    r"(negative|independen|shuffle|iid|permut|scrambl|noise|jitter|unique.?jitter)",
    re.IGNORECASE,
)
_WINNER_KEYS = (
    "winner",
    "winning_synthesizer",
    "winning_method",
    "best_synthesizer",
    "best_method",
    "selected",
)
_NAME_KEYS = ("name", "synthesizer", "method", "model", "id", "label")
_FIDELITY_KEYS = (
    "fidelity_score",
    "fidelity",
    "score_fidelity",
    "overall_fidelity",
)
_GAP_KEYS = (
    "tstr_gap_r2",
    "best_tstr_gap_r2",
    "gap_r2",
    "r2_gap",
    "tstr_trtr_gap_r2",
    "utility_gap_r2",
)
_RF_TSTR_KEYS = (
    "tstr_r2_rf",
    "tstr_rf_r2",
    "rf_tstr_r2",
    "tstr.rf.r2",
    "utility.rf.tstr_r2",
)
_RF_TRTR_KEYS = (
    "trtr_r2_rf",
    "trtr_rf_r2",
    "rf_trtr_r2",
    "trtr.rf.r2",
    "utility.rf.trtr_r2",
)
_LIN_TSTR_KEYS = (
    "tstr_r2_linear",
    "tstr_r2_ridge",
    "tstr_linear_r2",
    "linear_tstr_r2",
    "ridge_tstr_r2",
    "tstr.linear.r2",
    "tstr.ridge.r2",
    "utility.linear.tstr_r2",
)
_LIN_TRTR_KEYS = (
    "trtr_r2_linear",
    "trtr_r2_ridge",
    "trtr_linear_r2",
    "linear_trtr_r2",
    "ridge_trtr_r2",
    "trtr.linear.r2",
    "trtr.ridge.r2",
    "utility.linear.trtr_r2",
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("r2", "score", "value", "mean", "R2"):
            if key in value:
                return _as_float(value[key])
        return None
    if isinstance(value, (list, tuple)) and value:
        return _as_float(value[0])
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first(obj: Mapping[str, Any] | None, keys: Iterable[str]) -> Any:
    if not isinstance(obj, Mapping):
        return None
    for key in keys:
        if "." in key:
            got = _dig(obj, key)
            if got is not None:
                return got
        if key in obj and obj[key] is not None:
            return obj[key]
        low = {str(k).lower(): k for k in obj}
        if key.lower() in low:
            return obj[low[key.lower()]]
    return None


def _first_float(obj: Mapping[str, Any] | None, keys: Iterable[str]) -> float | None:
    return _as_float(_first(obj, keys))


def _looks_negative(name: str, row: Mapping[str, Any]) -> bool:
    if row.get("is_negative_control") is True or row.get("negative_control") is True:
        return True
    if row.get("role") == "negative_control" or row.get("kind") == "negative":
        return True
    if row.get("type") == "negative_control":
        return True
    return bool(_NEG_NAME_RE.search(str(name or "")))


def _model_block(row: Mapping[str, Any], *aliases: str) -> Mapping[str, Any] | None:
    for container_key in ("models", "utility", "tstr", "regression"):
        block = row.get(container_key)
        if isinstance(block, Mapping):
            for alias in aliases:
                if alias in block and isinstance(block[alias], Mapping):
                    return block[alias]
    for alias in aliases:
        block = row.get(alias)
        if isinstance(block, Mapping):
            return block
    return None


def _tstr_trtr_from_block(block: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(block, Mapping):
        return None, None
    tstr = _first_float(block, ("tstr_r2", "tstr", "r2_tstr", "r2"))
    trtr = _first_float(block, ("trtr_r2", "trtr", "r2_trtr"))
    # Nested {tstr: {r2: ...}}
    if tstr is None and isinstance(block.get("tstr"), Mapping):
        tstr = _first_float(block["tstr"], ("r2", "score", "value"))
    if trtr is None and isinstance(block.get("trtr"), Mapping):
        trtr = _first_float(block["trtr"], ("r2", "score", "value"))
    return tstr, trtr


@dataclass
class SynthRow:
    name: str
    fidelity_score: float | None = None
    tstr_r2_rf: float | None = None
    trtr_r2_rf: float | None = None
    tstr_r2_linear: float | None = None
    trtr_r2_linear: float | None = None
    tstr_gap_r2_rf: float | None = None
    tstr_gap_r2_linear: float | None = None
    tstr_gap_r2: float | None = None
    is_negative_control: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def best_gap(self) -> float | None:
        """Smaller signed gap (TRTR − TSTR) is better. Gate uses this value."""
        candidates = [
            g
            for g in (self.tstr_gap_r2_rf, self.tstr_gap_r2_linear, self.tstr_gap_r2)
            if g is not None
        ]
        if not candidates:
            return None
        # Prefer the two model-specific gaps when both exist.
        model_gaps = [g for g in (self.tstr_gap_r2_rf, self.tstr_gap_r2_linear) if g is not None]
        if model_gaps:
            return min(model_gaps)
        return self.tstr_gap_r2


def _gap(trtr: float | None, tstr: float | None) -> float | None:
    if trtr is None or tstr is None:
        return None
    return float(trtr) - float(tstr)


def parse_synth_row(item: Any, fallback_name: str = "unknown") -> SynthRow:
    if not isinstance(item, Mapping):
        return SynthRow(name=str(fallback_name), raw={"value": item})
    name = _first(item, _NAME_KEYS)
    name = str(name) if name is not None else str(fallback_name)

    fidelity = _first_float(item, _FIDELITY_KEYS)
    if fidelity is None and isinstance(item.get("fidelity"), Mapping):
        fidelity = _first_float(item["fidelity"], _FIDELITY_KEYS + ("score",))

    tstr_rf = _first_float(item, _RF_TSTR_KEYS)
    trtr_rf = _first_float(item, _RF_TRTR_KEYS)
    tstr_lin = _first_float(item, _LIN_TSTR_KEYS)
    trtr_lin = _first_float(item, _LIN_TRTR_KEYS)

    rf_block = _model_block(item, "rf", "random_forest", "RandomForest", "hgb")
    lin_block = _model_block(item, "linear", "ridge", "ols", "lr")
    if tstr_rf is None or trtr_rf is None:
        bt, br = _tstr_trtr_from_block(rf_block)
        tstr_rf = tstr_rf if tstr_rf is not None else bt
        trtr_rf = trtr_rf if trtr_rf is not None else br
    if tstr_lin is None or trtr_lin is None:
        bt, br = _tstr_trtr_from_block(lin_block)
        tstr_lin = tstr_lin if tstr_lin is not None else bt
        trtr_lin = trtr_lin if trtr_lin is not None else br

    gap_rf = _first_float(
        item, ("tstr_gap_r2_rf", "gap_r2_rf", "rf_tstr_gap_r2")
    )
    gap_lin = _first_float(
        item, ("tstr_gap_r2_linear", "tstr_gap_r2_ridge", "gap_r2_linear")
    )
    if gap_rf is None:
        gap_rf = _gap(trtr_rf, tstr_rf)
    if gap_lin is None:
        gap_lin = _gap(trtr_lin, tstr_lin)
    if gap_rf is None and rf_block is not None:
        gap_rf = _first_float(rf_block, ("gap", "tstr_gap_r2", "gap_r2"))
    if gap_lin is None and lin_block is not None:
        gap_lin = _first_float(lin_block, ("gap", "tstr_gap_r2", "gap_r2"))

    gap = _first_float(item, _GAP_KEYS)
    if gap is None:
        model_gaps = [g for g in (gap_rf, gap_lin) if g is not None]
        gap = min(model_gaps) if model_gaps else None

    return SynthRow(
        name=name,
        fidelity_score=fidelity,
        tstr_r2_rf=tstr_rf,
        trtr_r2_rf=trtr_rf,
        tstr_r2_linear=tstr_lin,
        trtr_r2_linear=trtr_lin,
        tstr_gap_r2_rf=gap_rf,
        tstr_gap_r2_linear=gap_lin,
        tstr_gap_r2=gap,
        is_negative_control=_looks_negative(name, item),
        raw=dict(item),
    )


def _as_row_list(payload: Mapping[str, Any]) -> list[SynthRow]:
    rows: list[SynthRow] = []
    for key in ("leaderboard", "results", "synthesizers", "methods", "candidates"):
        block = payload.get(key)
        if isinstance(block, list):
            for i, item in enumerate(block):
                rows.append(parse_synth_row(item, fallback_name=f"method_{i}"))
            break
        if isinstance(block, Mapping):
            for name, item in block.items():
                if isinstance(item, Mapping):
                    merged = {"name": name, **item}
                else:
                    merged = {"name": name, "value": item}
                rows.append(parse_synth_row(merged, fallback_name=str(name)))
            break
    if not rows and any(k in payload for k in _FIDELITY_KEYS + _NAME_KEYS + _GAP_KEYS):
        rows.append(parse_synth_row(payload))
    return rows


def _winner_name(payload: Mapping[str, Any], rows: list[SynthRow]) -> str | None:
    raw = _first(payload, _WINNER_KEYS)
    if isinstance(raw, Mapping):
        raw = _first(raw, _NAME_KEYS)
    if raw is not None:
        return str(raw)
    scored = [r for r in rows if not r.is_negative_control]
    if not scored:
        return None

    def sort_key(r: SynthRow) -> tuple:
        fid = r.fidelity_score if r.fidelity_score is not None else -1.0
        gap = r.best_gap() if r.best_gap() is not None else 99.0
        return (-fid, gap, r.name)

    scored.sort(key=sort_key)
    return scored[0].name


def _pick_row(rows: list[SynthRow], name: str | None) -> SynthRow | None:
    if not name:
        return None
    for r in rows:
        if r.name == name:
            return r
    low = str(name).lower()
    for r in rows:
        if r.name.lower() == low:
            return r
    return None


def _negative_row(payload: Mapping[str, Any], rows: list[SynthRow]) -> SynthRow | None:
    for key in ("negative_control", "negative", "control"):
        block = payload.get(key)
        if isinstance(block, Mapping):
            row = parse_synth_row(block, fallback_name="negative_control")
            row.is_negative_control = True
            return row
        if isinstance(block, list) and block:
            row = parse_synth_row(block[0], fallback_name="negative_control")
            row.is_negative_control = True
            return row
        if isinstance(block, str):
            found = _pick_row(rows, block)
            if found:
                found.is_negative_control = True
                return found
    named = [r for r in rows if r.is_negative_control]
    return named[0] if named else None


@dataclass
class GateResult:
    name: str
    passed: bool | None
    detail: str


def evaluate_gates(winner: SynthRow | None, negative: SynthRow | None) -> list[GateResult]:
    gates: list[GateResult] = []

    if winner is None:
        gates.append(
            GateResult(
                "TSTR R² gap ≤ 0.08 (best of RF / linear)",
                None,
                "No winning synthesizer in the artifact — gate not scored.",
            )
        )
        gates.append(
            GateResult(
                "fidelity_score ≥ 0.70 (winner)",
                None,
                "No winning synthesizer in the artifact — gate not scored.",
            )
        )
    else:
        gap = winner.best_gap()
        if gap is None:
            gates.append(
                GateResult(
                    "TSTR R² gap ≤ 0.08 (best of RF / linear)",
                    None,
                    f"{winner.name}: TSTR/TRTR R² missing for both RF and linear.",
                )
            )
        else:
            ok = gap <= GATE_TSTR_GAP_MAX
            bits = []
            if winner.tstr_gap_r2_rf is not None:
                bits.append(f"RF gap={winner.tstr_gap_r2_rf:.4f}")
            if winner.tstr_gap_r2_linear is not None:
                bits.append(f"linear gap={winner.tstr_gap_r2_linear:.4f}")
            bits.append(f"best gap={gap:.4f} (need ≤ {GATE_TSTR_GAP_MAX:.2f})")
            gates.append(
                GateResult(
                    "TSTR R² gap ≤ 0.08 (best of RF / linear)",
                    ok,
                    f"{winner.name}: " + "; ".join(bits),
                )
            )

        fid = winner.fidelity_score
        if fid is None:
            gates.append(
                GateResult(
                    "fidelity_score ≥ 0.70 (winner)",
                    None,
                    f"{winner.name}: fidelity_score missing.",
                )
            )
        else:
            gates.append(
                GateResult(
                    "fidelity_score ≥ 0.70 (winner)",
                    fid >= GATE_FIDELITY_MIN,
                    f"{winner.name}: fidelity_score={fid:.4f} (need ≥ {GATE_FIDELITY_MIN:.2f})",
                )
            )

    if negative is None:
        gates.append(
            GateResult(
                "negative control is weak (fidelity < 0.55 or TSTR gap > 0.15)",
                None,
                "No negative-control row found — gate not scored.",
            )
        )
    else:
        fid = negative.fidelity_score
        gap = negative.best_gap()
        fid_ok = fid is not None and fid < GATE_NEG_FIDELITY_MAX
        gap_ok = gap is not None and gap > GATE_NEG_TSTR_GAP_MIN
        if fid is None and gap is None:
            passed: bool | None = None
            detail = f"{negative.name}: neither fidelity_score nor tstr_gap_r2 present."
        else:
            passed = bool(fid_ok or gap_ok)
            fid_txt = "missing" if fid is None else f"{fid:.4f}"
            gap_txt = "missing" if gap is None else f"{gap:.4f}"
            detail = (
                f"{negative.name}: fidelity_score={fid_txt} "
                f"(pass if < {GATE_NEG_FIDELITY_MAX:.2f}) OR "
                f"tstr_gap_r2={gap_txt} (pass if > {GATE_NEG_TSTR_GAP_MIN:.2f}). "
                f"Triggered via fidelity={fid_ok}, gap={gap_ok}."
            )
        gates.append(
            GateResult(
                "negative control is weak (fidelity < 0.55 or TSTR gap > 0.15)",
                passed,
                detail,
            )
        )
    return gates


def load_evaluation(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else {"leaderboard": payload}


def load_tables(data_dir: Path) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
    real = None
    synth: dict[str, pd.DataFrame] = {}
    if not data_dir.is_dir():
        return real, synth
    original = data_dir / "original.csv"
    if original.is_file():
        try:
            real = pd.read_csv(original)
        except Exception as exc:
            warnings.warn(f"Could not read {original}: {exc}")
    for csv_path in sorted(data_dir.glob("synthetic_*.csv")):
        try:
            name = csv_path.stem.replace("synthetic_", "", 1) or csv_path.stem
            synth[name] = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"Could not read {csv_path}: {exc}")
    return real, synth


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _gate_emoji(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "N/A"


def _interpret_section() -> str:
    return """## How to interpret (Jorge)

This is a **modeling-substitute** test, not a row-by-row disguise. There is no pairing
between real row `i` and synthetic row `i'`. Looking at a single customer and asking
"did we copy them?" is the wrong question.

1. **`fidelity_score` (0–1, higher is better).** Blend of 1-D margins, Spearman
   correlation structure, and whether the `X → y` relationship survived. The
   winner must be **≥ 0.70**. A pretty histogram is not enough if joints die.
2. **TSTR vs TRTR R² gap.** Train on synthetic, test on a real hold-out (TSTR).
   Compare to train-on-real / test-on-real (TRTR). We report **Random Forest and
   linear/Ridge**. The gate uses the **best** (smaller) of those two gaps and
   requires **≤ 0.08**. If RF gap is 0.05 and linear gap is 0.12, the gate still
   passes — the synthetic table is good enough for at least one frozen model
   family. If *both* gaps are large, `X'` lost the signal that predicts `y`.
3. **Negative control.** A column-wise shuffle (or similar junk generator) must
   look *bad*: `fidelity_score < 0.55` **or** TSTR gap `> 0.15`. If junk data
   passes, the metric is too soft and we should not trust a CART/copula "win".
4. **Plots.** Overlay histograms of `y`, `income`, `risk_score` should sit on
   top of each other. Heatmaps should keep the same red/blue blocks (especially
   `risk_score`–`credit_util`–`y`). The scatter should keep the hockey-stick:
   expected loss rising once `risk_score` is high.

**En corto:** si el winner pasa los tres gates, `X'` sirve para entrenar un
modelo y llevarlo a datos reales sin un desplome grande de R². Si el control
negativo *también* pasa, no celebres — las métricas están mal calibradas.
"""


def build_markdown(
    *,
    eval_path: Path,
    payload: Mapping[str, Any] | None,
    rows: list[SynthRow],
    winner: SynthRow | None,
    negative: SynthRow | None,
    gates: list[GateResult],
    real: pd.DataFrame | None,
    synth_tables: Mapping[str, pd.DataFrame],
    plot_paths: list[Path],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    decided = [g.passed for g in gates if g.passed is not None]
    if not decided:
        overall = "UNSCORED (missing artifacts)"
    elif all(decided):
        overall = "ALL SCORED GATES PASSED"
    else:
        overall = "ONE OR MORE GATES FAILED"

    lines: list[str] = [
        "# Synthetic-data fidelity — RESULTS",
        "",
        f"_Generated {now} by `python -m src.report`._",
        "",
        f"**Overall:** {overall}",
        "",
        "## Success gates",
        "",
        "Fixed thresholds (do not drift these):",
        "",
        f"- Best TSTR R² gap vs TRTR **≤ {GATE_TSTR_GAP_MAX:.2f}** (regression; report RF and linear; gate on the better model).",
        f"- `fidelity_score` **≥ {GATE_FIDELITY_MIN:.2f}** for the winning synthesizer.",
        f"- Negative control: `fidelity_score` **< {GATE_NEG_FIDELITY_MAX:.2f}** **OR** `tstr_gap_r2` **> {GATE_NEG_TSTR_GAP_MIN:.2f}**.",
        "",
    ]

    gate_rows = [
        [
            g.name,
            _gate_emoji(g.passed),
            g.detail,
        ]
        for g in gates
    ]
    lines.append(_md_table(["Gate", "Status", "Detail"], gate_rows))
    lines.append("")

    if payload is None:
        lines.extend(
            [
                "## Evaluation artifact",
                "",
                f"`{eval_path}` was **not found** (or failed to parse). "
                "Leaderboard numbers below are empty until `src.evaluate` writes the JSON.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Evaluation artifact",
                "",
                f"Loaded `{eval_path}` "
                + (f"({len(payload)} top-level keys)." if payload else "."),
                "",
            ]
        )

    lines.extend(["## Leaderboard", ""])
    if not rows:
        lines.append("No synthesizer rows were found in the evaluation artifact.")
        lines.append("")
    else:
        table_rows: list[list[str]] = []
        for r in rows:
            mark = " ← winner" if winner is not None and r.name == winner.name else ""
            neg = " yes" if r.is_negative_control else ""
            table_rows.append(
                [
                    f"{r.name}{mark}",
                    _fmt(r.fidelity_score),
                    _fmt(r.tstr_r2_rf),
                    _fmt(r.trtr_r2_rf),
                    _fmt(r.tstr_gap_r2_rf),
                    _fmt(r.tstr_r2_linear),
                    _fmt(r.trtr_r2_linear),
                    _fmt(r.tstr_gap_r2_linear),
                    _fmt(r.best_gap()),
                    neg.strip() or "—",
                ]
            )
        lines.append(
            _md_table(
                [
                    "synthesizer",
                    "fidelity",
                    "TSTR R² RF",
                    "TRTR R² RF",
                    "gap RF",
                    "TSTR R² linear",
                    "TRTR R² linear",
                    "gap linear",
                    "best gap",
                    "neg. ctrl",
                ],
                table_rows,
            )
        )
        lines.append("")
        lines.append(
            "Gap = TRTR R² − TSTR R². The **best gap** is `min(gap_RF, gap_linear)` "
            "and is the number used by the 0.08 gate."
        )
        lines.append("")

    lines.extend(["## Winner", ""])
    if winner is None:
        lines.append("No winning synthesizer could be identified.")
    else:
        lines.append(f"**{winner.name}**")
        lines.append("")
        lines.append(
            f"- fidelity_score = {_fmt(winner.fidelity_score)} "
            f"(gate ≥ {GATE_FIDELITY_MIN:.2f})"
        )
        lines.append(
            f"- RF: TSTR={_fmt(winner.tstr_r2_rf)} TRTR={_fmt(winner.trtr_r2_rf)} "
            f"gap={_fmt(winner.tstr_gap_r2_rf)}"
        )
        lines.append(
            f"- linear: TSTR={_fmt(winner.tstr_r2_linear)} TRTR={_fmt(winner.trtr_r2_linear)} "
            f"gap={_fmt(winner.tstr_gap_r2_linear)}"
        )
        lines.append(
            f"- best-model gap = {_fmt(winner.best_gap())} "
            f"(gate ≤ {GATE_TSTR_GAP_MAX:.2f})"
        )
    lines.append("")

    lines.extend(["## Negative control", ""])
    if negative is None:
        lines.append("No negative-control synthesizer was present.")
    else:
        lines.append(f"**{negative.name}**")
        lines.append("")
        lines.append(f"- fidelity_score = {_fmt(negative.fidelity_score)}")
        lines.append(f"- tstr_gap_r2 (best model) = {_fmt(negative.best_gap())}")
        lines.append(
            f"- protocol pass if fidelity < {GATE_NEG_FIDELITY_MAX:.2f} "
            f"or gap > {GATE_NEG_TSTR_GAP_MIN:.2f}"
        )
    lines.append("")

    lines.extend(["## Data tables", ""])
    if real is None:
        lines.append("`data/original.csv` not found (tolerated).")
    else:
        lines.append(
            f"`data/original.csv`: {real.shape[0]} rows × {real.shape[1]} columns "
            f"(`{', '.join(map(str, real.columns[:12]))}"
            f"{'…' if real.shape[1] > 12 else ''}`)."
        )
    if not synth_tables:
        lines.append("`data/synthetic_<name>.csv` not found (tolerated).")
    else:
        for name, frame in synth_tables.items():
            lines.append(
                f"`data/synthetic_{name}.csv`: {frame.shape[0]} rows × {frame.shape[1]} columns."
            )
    lines.append("")

    lines.extend(["## Plots", ""])
    if not plot_paths:
        lines.append(
            "No comparison PNGs were written (need both real and at least one "
            "synthetic table). When present, `save_comparison_plots` writes:"
        )
        lines.append("")
        for name in (
            "marginal_y.png",
            "marginal_income.png",
            "marginal_risk_score.png",
            "corr_heatmap_real.png",
            "corr_heatmap_synth.png",
            "scatter_y_vs_risk_score.png",
        ):
            lines.append(f"- `{name}`")
    else:
        lines.append("Wrote:")
        lines.append("")
        for path in plot_paths:
            lines.append(f"- `{path}`")
    lines.append("")
    lines.append(_interpret_section())
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    *,
    eval_path: Path | None = None,
    data_dir: Path | None = None,
    report_path: Path | None = None,
    plots_dir: Path | None = None,
) -> Path:
    eval_path = Path(eval_path) if eval_path else DEFAULT_EVAL
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA
    report_path = Path(report_path) if report_path else DEFAULT_REPORT
    plots_dir = Path(plots_dir) if plots_dir else DEFAULT_PLOTS

    payload = load_evaluation(eval_path)
    rows = _as_row_list(payload) if payload else []
    winner = _pick_row(rows, _winner_name(payload, rows) if payload else None)
    if winner is None and rows:
        non_neg = [r for r in rows if not r.is_negative_control]
        winner = (non_neg or rows)[0]
    negative = _negative_row(payload or {}, rows)
    gates = evaluate_gates(winner, negative)

    real, synth_tables = load_tables(data_dir)
    plot_paths: list[Path] = []
    if real is not None and synth_tables:
        try:
            from .plots import save_comparison_plots
        except ImportError:  # pragma: no cover - script-style fallback
            from src.plots import save_comparison_plots

        # Prefer the winner's CSV when the name matches; otherwise first file.
        chosen = None
        if winner is not None:
            for key, frame in synth_tables.items():
                if key.lower() == winner.name.lower() or winner.name.lower() in key.lower():
                    chosen = frame
                    break
        if chosen is None:
            chosen = next(iter(synth_tables.values()))
        try:
            plot_paths = save_comparison_plots(real, chosen, plots_dir)
        except Exception as exc:
            warnings.warn(f"Plot generation skipped: {exc}")

    text = build_markdown(
        eval_path=eval_path,
        payload=payload,
        rows=rows,
        winner=winner,
        negative=negative,
        gates=gates,
        real=real,
        synth_tables=synth_tables,
        plot_paths=plot_paths,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize synthetic-data fidelity into reports/RESULTS.md"
    )
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="evaluation.json path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="CSV directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="RESULTS.md path")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS, help="PNG output dir")
    args = parser.parse_args(argv)
    path = write_report(
        eval_path=args.eval,
        data_dir=args.data_dir,
        report_path=args.out,
        plots_dir=args.plots_dir,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
