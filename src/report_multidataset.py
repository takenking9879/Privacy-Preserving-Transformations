"""Write ``reports/MULTI_DATASET.md`` from ``artifacts/multidataset.json``.

CLI::

    python -m src.report_multidataset

Missing ``artifacts/multidataset.json`` is tolerated: the markdown still
records what was found, empty tables, and the generality note for Jorge.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Suite gates (lockstep with src.eval_multidataset)
GATE_FIDELITY = 0.70
GATE_TSTR_R2_GAP = 0.10

CONTROL_METHODS = frozenset({"identity", "negative_control"})

# A method is *general* only if it passes all three — not just credit.
GENERALITY_DATASETS = ("interactions", "multimodal", "imbalanced")
_GENERALITY_ALIASES: dict[str, str] = {
    "interaction": "interactions",
    "interactions": "interactions",
    "multimodal": "multimodal",
    "multi_modal": "multimodal",
    "multi-modal": "multimodal",
    "imbalanced": "imbalanced",
    "imbalance": "imbalanced",
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "artifacts" / "multidataset.json"
DEFAULT_REPORT = ROOT / "reports" / "MULTI_DATASET.md"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("value", "score", "mean", "r2"):
            if key in value:
                return _as_float(value[key])
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _norm_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _canon_dataset(name: Any) -> str:
    raw = _norm_name(name)
    return _GENERALITY_ALIASES.get(raw, raw)


def _is_control(name: str) -> bool:
    return _norm_name(name) in CONTROL_METHODS


def _fmt(value: Optional[float], digits: int = 4) -> str:
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


def _yes_no(flag: Optional[bool]) -> str:
    if flag is True:
        return "yes"
    if flag is False:
        return "no"
    return "—"


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class PairRow:
    dataset: str
    method: str
    status: str = "ok"
    error: Optional[str] = None
    fidelity_score: Optional[float] = None
    utility_score: Optional[float] = None
    combined_score: Optional[float] = None
    tstr_gap_r2: Optional[float] = None
    passes: Optional[bool] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_control(self) -> bool:
        return _is_control(self.method)

    def gate_pass(self) -> bool:
        if self.passes is True:
            return True
        if self.status != "ok":
            return False
        fid = self.fidelity_score
        gap = self.tstr_gap_r2
        if fid is None or gap is None:
            return False
        return fid >= GATE_FIDELITY and gap <= GATE_TSTR_R2_GAP


def _parse_pair(item: Mapping[str, Any]) -> PairRow:
    dataset = str(item.get("dataset") or item.get("dgp") or item.get("name") or "")
    method = str(item.get("method") or item.get("synthesizer") or item.get("model") or "")
    status = str(item.get("status") or "ok")
    fid = _as_float(item.get("fidelity_score") if "fidelity_score" in item else item.get("fidelity"))
    util = _as_float(item.get("utility_score"))
    combo = _as_float(item.get("combined_score"))
    if combo is None and fid is not None and util is not None:
        combo = float(fid) + float(util)
    gap = _as_float(item.get("tstr_gap_r2") if "tstr_gap_r2" in item else item.get("gap"))
    gates = item.get("gates")
    passes: Optional[bool] = None
    if isinstance(gates, Mapping):
        passes = _as_bool(gates.get("passes"))
        if passes is None:
            fid_ok = _as_bool(gates.get("fidelity_score_ge_0.70"))
            gap_ok = _as_bool(gates.get("tstr_gap_r2_le_0.10"))
            if fid_ok is not None and gap_ok is not None:
                passes = bool(fid_ok and gap_ok)
    return PairRow(
        dataset=dataset,
        method=method,
        status=status,
        error=str(item["error"]) if item.get("error") not in (None, "") else None,
        fidelity_score=fid,
        utility_score=util,
        combined_score=combo,
        tstr_gap_r2=gap,
        passes=passes,
        raw=dict(item),
    )


def _pairs_from_results(payload: Mapping[str, Any]) -> list[PairRow]:
    block = payload.get("results")
    if isinstance(block, list):
        return [_parse_pair(item) for item in block if isinstance(item, Mapping)]
    return []


def _pairs_from_matrix(payload: Mapping[str, Any]) -> list[PairRow]:
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping):
        return []
    combo = matrix.get("combined_score")
    gaps = matrix.get("tstr_gap_r2")
    if not isinstance(combo, Mapping) and not isinstance(gaps, Mapping):
        return []
    combo = combo if isinstance(combo, Mapping) else {}
    gaps = gaps if isinstance(gaps, Mapping) else {}
    methods = list(matrix.get("rows") or combo.keys() or gaps.keys())
    datasets: list[str] = list(matrix.get("cols") or [])
    if not datasets:
        seen: list[str] = []
        for src in (combo, gaps):
            for cells in src.values() if isinstance(src, Mapping) else []:
                if isinstance(cells, Mapping):
                    for d in cells:
                        if d not in seen:
                            seen.append(str(d))
        datasets = seen
    rows: list[PairRow] = []
    for method in methods:
        method = str(method)
        combo_row = combo.get(method) if isinstance(combo.get(method), Mapping) else {}
        gap_row = gaps.get(method) if isinstance(gaps.get(method), Mapping) else {}
        for dataset in datasets:
            dataset = str(dataset)
            c = _as_float(combo_row.get(dataset)) if combo_row else None
            g = _as_float(gap_row.get(dataset)) if gap_row else None
            if c is None and g is None:
                continue
            rows.append(
                PairRow(
                    dataset=dataset,
                    method=method,
                    combined_score=c,
                    tstr_gap_r2=g,
                )
            )
    return rows


def load_pairs(payload: Mapping[str, Any] | None) -> list[PairRow]:
    if not payload:
        return []
    rows = _pairs_from_results(payload)
    if rows:
        return rows
    return _pairs_from_matrix(payload)


def _ordered_names(
    pairs: list[PairRow],
    *,
    kind: str,
    payload: Mapping[str, Any] | None,
) -> list[str]:
    preferred: list[str] = []
    if payload:
        meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
        key = "datasets" if kind == "dataset" else "methods"
        raw = None
        if isinstance(meta, Mapping):
            raw = meta.get(key)
        if raw is None and kind == "dataset":
            matrix = payload.get("matrix")
            if isinstance(matrix, Mapping):
                raw = matrix.get("cols")
        if raw is None and kind == "method":
            matrix = payload.get("matrix")
            if isinstance(matrix, Mapping):
                raw = matrix.get("rows")
        if isinstance(raw, list):
            preferred = [str(x) for x in raw if x not in (None, "")]
    seen: list[str] = []
    for name in preferred:
        if name not in seen:
            seen.append(name)
    for pair in pairs:
        name = pair.dataset if kind == "dataset" else pair.method
        if name and name not in seen:
            seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def _median(values: Iterable[float]) -> Optional[float]:
    nums = [float(v) for v in values]
    if not nums:
        return None
    return float(statistics.median(nums))


def per_dataset_winners(
    pairs: list[PairRow],
    datasets: list[str],
) -> list[dict[str, Any]]:
    """Highest ``combined_score`` among non-control methods, per dataset."""
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        candidates = [
            p
            for p in pairs
            if p.dataset == dataset
            and not p.is_control
            and p.status == "ok"
            and p.combined_score is not None
        ]
        if not candidates:
            # Fall back to any scored non-control row (fidelity only).
            candidates = [
                p
                for p in pairs
                if p.dataset == dataset and not p.is_control and p.status == "ok"
            ]
        if not candidates:
            out.append(
                {
                    "dataset": dataset,
                    "method": None,
                    "combined_score": None,
                    "fidelity_score": None,
                    "tstr_gap_r2": None,
                    "passes": None,
                }
            )
            continue
        candidates.sort(
            key=lambda p: (
                -(p.combined_score if p.combined_score is not None else float("-inf")),
                -int(p.gate_pass()),
                p.tstr_gap_r2 if p.tstr_gap_r2 is not None else float("inf"),
                p.method,
            )
        )
        best = candidates[0]
        out.append(
            {
                "dataset": dataset,
                "method": best.method,
                "combined_score": best.combined_score,
                "fidelity_score": best.fidelity_score,
                "tstr_gap_r2": best.tstr_gap_r2,
                "passes": best.gate_pass(),
            }
        )
    return out


def combined_matrix(
    pairs: list[PairRow],
    methods: list[str],
    datasets: list[str],
) -> dict[str, dict[str, Optional[float]]]:
    grid: dict[str, dict[str, Optional[float]]] = {
        m: {d: None for d in datasets} for m in methods
    }
    for p in pairs:
        if p.method in grid and p.dataset in grid[p.method]:
            if p.status == "ok" or p.combined_score is not None:
                grid[p.method][p.dataset] = p.combined_score
    return grid


def method_summaries(
    pairs: list[PairRow],
    methods: list[str],
    datasets: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in methods:
        scored = [p for p in pairs if p.method == method and p.status == "ok"]
        passed = [p for p in scored if p.gate_pass()]
        passed_names = [p.dataset for p in passed]
        passed_canon = {_canon_dataset(n) for n in passed_names}
        combo_vals = [p.combined_score for p in scored if p.combined_score is not None]
        evaluated_canon = {
            _canon_dataset(p.dataset)
            for p in scored
            if p.dataset
        }
        veto = {
            name: (name in passed_canon) if name in evaluated_canon else None
            for name in GENERALITY_DATASETS
        }
        # All three veto DGPs must have been evaluated *and* passed.
        # Controls never count as general even if they clear the numeric gates.
        is_general = (not _is_control(method)) and all(
            veto[name] is True for name in GENERALITY_DATASETS
        )
        summaries.append(
            {
                "method": method,
                "median_combined_score": _median(combo_vals),
                "n_scored": len(combo_vals),
                "n_eval": len(scored),
                "n_pass": len(passed),
                "passed_datasets": passed_names,
                "veto": veto,
                "is_general": is_general,
                "is_control": _is_control(method),
            }
        )
    return summaries


def declare_general_winner(
    summaries: list[dict[str, Any]],
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pick a GENERAL winner: must pass interactions + multimodal + imbalanced.

    Among qualifying non-control methods, prefer highest median combined
    score, then most dataset passes, then name.  Credit-only wins do not
    qualify.  If the artifact names a winner that fails the veto, it is
    recorded as ``artifact_winner`` but not as the general winner.
    """
    artifact_winner: Optional[str] = None
    if payload and isinstance(payload.get("winner"), Mapping):
        raw = payload["winner"].get("method")
        if raw:
            artifact_winner = str(raw)
    elif payload and isinstance(payload.get("winner"), str):
        artifact_winner = str(payload["winner"])

    contenders = [
        s
        for s in summaries
        if not s["is_control"]
        and s["is_general"]
        and s["median_combined_score"] is not None
    ]
    contenders.sort(
        key=lambda s: (
            -(s["median_combined_score"] or float("-inf")),
            -(s["n_pass"] or 0),
            s["method"],
        )
    )
    if contenders:
        best = contenders[0]
        return {
            "method": best["method"],
            "median_combined_score": best["median_combined_score"],
            "n_pass": best["n_pass"],
            "n_eval": best["n_eval"],
            "passed_datasets": list(best["passed_datasets"]),
            "is_general": True,
            "artifact_winner": artifact_winner,
            "reason": (
                "Passes interactions AND multimodal AND imbalanced "
                "(not just credit); highest median combined_score among "
                "qualifying methods."
            ),
        }

    # No method cleared the generality veto.
    scored = [
        s
        for s in summaries
        if not s["is_control"] and s["median_combined_score"] is not None
    ]
    scored.sort(
        key=lambda s: (
            -(s["median_combined_score"] or float("-inf")),
            -(s["n_pass"] or 0),
            s["method"],
        )
    )
    best_median = scored[0]["method"] if scored else artifact_winner
    return {
        "method": None,
        "median_combined_score": None,
        "n_pass": 0,
        "n_eval": 0,
        "passed_datasets": [],
        "is_general": False,
        "artifact_winner": artifact_winner,
        "best_median_method": best_median,
        "reason": (
            "No non-control method passed interactions AND multimodal AND "
            "imbalanced. Winning credit (or a high median on easy DGPs) "
            "is not a general claim."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _jorge_note() -> str:
    return """## Note for Jorge / Nota para Jorge

**EN.** A method is *general* only if it **passes on `interactions` AND `multimodal` AND `imbalanced`**.
A win on `credit` (CART looking strong on the legacy DGP) is a local result, not transfer.
Do not call the suite winner “general” because it topped the credit column or had a high median
on additive / near-elliptical tables. Those three DGPs are the veto: product / threshold
structure, mixture modes, and a rare class. Fail any one of them and the method is a specialist,
not a portable synthesizer.

**ES.** Un método es *general* solo si **pasa en `interactions` Y `multimodal` Y `imbalanced`**.
Ganar en `credit` no basta: es una victoria local, no transferencia.
No declares ganador general a quien solo brilla en crédito o en tablas casi aditivas.
Esos tres DGPs son el veto (interacciones, modos, clase rara). Si falla uno, es especialista,
no un sintetizador portable.
"""


def build_markdown(
    *,
    eval_path: Path,
    payload: Mapping[str, Any] | None,
    pairs: list[PairRow],
    datasets: list[str],
    methods: list[str],
    winners: list[dict[str, Any]],
    matrix: dict[str, dict[str, Optional[float]]],
    summaries: list[dict[str, Any]],
    general: dict[str, Any],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    n = meta.get("n") if isinstance(meta, Mapping) else None
    seed = meta.get("seed") if isinstance(meta, Mapping) else None
    quick = meta.get("quick") if isinstance(meta, Mapping) else None

    lines: list[str] = [
        "# Multi-dataset synthesizer report",
        "",
        f"_Generated {now} by `python -m src.report_multidataset`._",
        "",
        "Suite gates (same numbers as `src.eval_multidataset`):",
        "",
        f"- `fidelity_score` **≥ {GATE_FIDELITY:.2f}**",
        f"- `tstr_gap_r2` **≤ {GATE_TSTR_R2_GAP:.2f}**",
        "- A pair **passes** only if both gates hold.",
        "- **General** = pass on `interactions` **and** `multimodal` **and** `imbalanced` "
        "(not just `credit`). Controls `identity` / `negative_control` are excluded from winners.",
        "",
    ]

    if payload is None:
        lines.extend(
            [
                "## Evaluation artifact",
                "",
                f"`{eval_path}` was **not found** (or failed to parse). "
                "Tables below are empty until `python -m src.eval_multidataset` writes the JSON.",
                "",
            ]
        )
    else:
        bits = [f"Loaded `{eval_path}`"]
        if n is not None:
            bits.append(f"n={n}")
        if seed is not None:
            bits.append(f"seed={seed}")
        if quick is not None:
            bits.append(f"quick={quick}")
        bits.append(f"{len(datasets)} datasets × {len(methods)} methods")
        bits.append(f"{len(pairs)} pair rows")
        lines.extend(
            [
                "## Evaluation artifact",
                "",
                " · ".join(bits) + ".",
                "",
            ]
        )

    # --- per-dataset winner ---
    lines.extend(["## Per-dataset winner", ""])
    if not datasets:
        lines.append("No datasets in the artifact.")
        lines.append("")
    else:
        table_rows: list[list[str]] = []
        for w in winners:
            table_rows.append(
                [
                    str(w["dataset"]),
                    str(w["method"] or "—"),
                    _fmt(w.get("combined_score")),
                    _fmt(w.get("fidelity_score")),
                    _fmt(w.get("tstr_gap_r2")),
                    _yes_no(w.get("passes")),
                ]
            )
        lines.append(
            _md_table(
                [
                    "dataset",
                    "winner",
                    "combined",
                    "fidelity",
                    "tstr_gap_r2",
                    "pass",
                ],
                table_rows,
            )
        )
        lines.append("")
        lines.append(
            "Winner per DGP = non-control method with the highest "
            "`combined_score = fidelity_score + utility_score` "
            "(ties: gate pass, then smaller TSTR gap)."
        )
        lines.append("")

    # --- method × dataset combined scores ---
    lines.extend(["## Method × dataset combined scores", ""])
    if not methods or not datasets:
        lines.append("No method × dataset matrix could be built.")
        lines.append("")
    else:
        headers = ["method", *datasets]
        table_rows = []
        for method in methods:
            cells = [_fmt(matrix.get(method, {}).get(d)) for d in datasets]
            table_rows.append([method, *cells])
        lines.append(_md_table(headers, table_rows))
        lines.append("")
        lines.append(
            "`combined_score = fidelity_score + utility_score` (higher is better). "
            "Em dash = missing / error pair."
        )
        lines.append("")

    # --- median score per method ---
    lines.extend(["## Median combined score per method", ""])
    if not summaries:
        lines.append("No methods to score.")
        lines.append("")
    else:
        ranked = sorted(
            summaries,
            key=lambda s: (
                s["is_control"],
                -(s["median_combined_score"] if s["median_combined_score"] is not None else float("-inf")),
                s["method"],
            ),
        )
        table_rows = [
            [
                s["method"],
                _fmt(s["median_combined_score"]),
                str(s["n_scored"]),
                "yes" if s["is_control"] else "—",
            ]
            for s in ranked
        ]
        lines.append(
            _md_table(
                ["method", "median_combined", "n_scored", "control"],
                table_rows,
            )
        )
        lines.append("")
        lines.append(
            "Median is over datasets with a finite `combined_score`. "
            "Controls are listed last and never win the suite."
        )
        lines.append("")

    # --- pass-count ---
    lines.extend(
        [
            "## Pass-count per method",
            "",
            f"A pair passes when `fidelity_score ≥ {GATE_FIDELITY:.2f}` "
            f"**and** `tstr_gap_r2 ≤ {GATE_TSTR_R2_GAP:.2f}`.",
            "",
        ]
    )
    if not summaries:
        lines.append("No methods to score.")
        lines.append("")
    else:
        ranked = sorted(
            summaries,
            key=lambda s: (
                s["is_control"],
                -(s["n_pass"] or 0),
                -(s["median_combined_score"] if s["median_combined_score"] is not None else float("-inf")),
                s["method"],
            ),
        )
        table_rows = []
        for s in ranked:
            veto = s["veto"]
            table_rows.append(
                [
                    s["method"],
                    f"{s['n_pass']}/{s['n_eval']}",
                    ", ".join(s["passed_datasets"]) or "—",
                    _yes_no(veto.get("interactions")),
                    _yes_no(veto.get("multimodal")),
                    _yes_no(veto.get("imbalanced")),
                    _yes_no(s["is_general"]),
                ]
            )
        lines.append(
            _md_table(
                [
                    "method",
                    "pass-count",
                    "datasets passed",
                    "interactions",
                    "multimodal",
                    "imbalanced",
                    "general?",
                ],
                table_rows,
            )
        )
        lines.append("")
        lines.append(
            "`general?` is yes only when the method passed **all three** veto "
            "DGPs (`interactions`, `multimodal`, `imbalanced`). Missing veto "
            "columns stay em dash and block the seal."
        )
        lines.append("")

    # --- general winner ---
    lines.extend(["## General winner", ""])
    method = general.get("method")
    if method:
        lines.append(f"**{method}**")
        lines.append("")
        lines.append(
            f"- median combined_score = {_fmt(general.get('median_combined_score'))}"
        )
        lines.append(
            f"- pass-count = {general.get('n_pass')}/{general.get('n_eval')}"
        )
        passed = general.get("passed_datasets") or []
        lines.append(
            "- passed datasets: " + (", ".join(passed) if passed else "—")
        )
        lines.append(f"- {general.get('reason')}")
    else:
        lines.append("**None.** No method qualifies as general.")
        lines.append("")
        if general.get("best_median_method"):
            lines.append(
                f"Best median `combined_score` (not a general claim): "
                f"`{general['best_median_method']}`."
            )
        if general.get("artifact_winner"):
            lines.append(
                f"Artifact `winner.method` was `{general['artifact_winner']}` "
                "(median combined_score in `eval_multidataset`; that is **not** "
                "the generality seal)."
            )
        if general.get("reason"):
            lines.append("")
            lines.append(str(general["reason"]))
    lines.append("")
    lines.append(_jorge_note())
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — tolerate bad JSON
        warnings.warn(f"Could not parse {path}: {exc}")
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"results": payload}
    warnings.warn(f"{path} is not a JSON object; ignoring.")
    return None


def write_report(
    *,
    eval_path: Path | None = None,
    report_path: Path | None = None,
) -> Path:
    eval_path = Path(eval_path) if eval_path else DEFAULT_EVAL
    report_path = Path(report_path) if report_path else DEFAULT_REPORT

    payload = load_artifact(eval_path)
    pairs = load_pairs(payload)
    datasets = _ordered_names(pairs, kind="dataset", payload=payload)
    methods = _ordered_names(pairs, kind="method", payload=payload)
    winners = per_dataset_winners(pairs, datasets)
    matrix = combined_matrix(pairs, methods, datasets)
    summaries = method_summaries(pairs, methods, datasets)
    general = declare_general_winner(summaries, payload)

    text = build_markdown(
        eval_path=eval_path,
        payload=payload,
        pairs=pairs,
        datasets=datasets,
        methods=methods,
        winners=winners,
        matrix=matrix,
        summaries=summaries,
        general=general,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize artifacts/multidataset.json into reports/MULTI_DATASET.md"
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=DEFAULT_EVAL,
        help="multidataset.json path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT,
        help="MULTI_DATASET.md path",
    )
    args = parser.parse_args(argv)
    path = write_report(eval_path=args.eval, report_path=args.out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
