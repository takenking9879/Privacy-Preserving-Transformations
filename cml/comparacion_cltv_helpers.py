"""
Utilidades para comparar dos snapshots de CLTV futuros efectivo.

Pensado para pegarse / importarse en CML. Las queries viven en el notebook
(Impala + %%sql); aquí solo está el contraste estadístico y las gráficas.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


CANONICAL_NUMERIC = [
    "plz_estimado_asignado",
    "cdp",
    "ticket_promedio",
    "cdp_ajustada",
    "monto_topado",
    "plz_estimado",
    "tasa_estimada",
    "mto_capital",
    "factor_prepago",
    "tasa_estimada_ajustada",
    "plz_estimado_ajustado",
    "intereses",
    "bhs_score",
    "reservas_esperadas",
    "costo_calles_esperadas",
    "costo_llamada_esperadas",
    "costo_sms_esperadas",
    "positivos",
    "negativos",
    "value",
    "esperanza_efe",
    "cltv",
]

CATEGORICAL = ["familia", "num_periodo_sem"]
ONLY_IN_B = ["intereses2"]

# Factores típicos de escala (proporción vs %, puntos vs bps, etc.)
SCALE_FACTORS = [0.01, 0.1, 10.0, 100.0, 1000.0, 10000.0]

PSI_OK = 0.10
PSI_WARN = 0.25


def _safe_name(col: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", col)


def sql_count(table: str) -> str:
    return f"""
SELECT
  count(*) AS n_rows,
  ndv(id_master) AS n_id_master,
  count(*) - count(id_master) AS n_id_master_null
FROM {table}
""".strip()


def sql_nulos(table: str, columns: Iterable[str]) -> str:
    parts = ["count(*) AS n_rows"]
    for col in columns:
        alias = _safe_name(col)
        parts.append(f"(count(*) - count({col})) AS {alias}__nulls")
        parts.append(f"count({col}) AS {alias}__n")
    return "SELECT\n  " + ",\n  ".join(parts) + f"\nFROM {table}"


def sql_perfil_numerico(table: str, columns: Iterable[str]) -> str:
    """Un solo scan de Impala: nulos, ceros y percentiles aproximados."""
    parts = ["count(*) AS n_rows"]
    for col in columns:
        alias = _safe_name(col)
        parts.extend(
            [
                f"(count(*) - count({col})) AS {alias}__nulls",
                f"sum(cast({col} = 0 AS bigint)) AS {alias}__zeros",
                f"avg(cast({col} AS double)) AS {alias}__mean",
                f"stddev(cast({col} AS double)) AS {alias}__std",
                f"min(cast({col} AS double)) AS {alias}__min",
                f"max(cast({col} AS double)) AS {alias}__max",
                f"appx_median(cast({col} AS double)) AS {alias}__median",
                f"percentile_approx(cast({col} AS double), 0.01) AS {alias}__p01",
                f"percentile_approx(cast({col} AS double), 0.05) AS {alias}__p05",
                f"percentile_approx(cast({col} AS double), 0.25) AS {alias}__p25",
                f"percentile_approx(cast({col} AS double), 0.75) AS {alias}__p75",
                f"percentile_approx(cast({col} AS double), 0.95) AS {alias}__p95",
                f"percentile_approx(cast({col} AS double), 0.99) AS {alias}__p99",
            ]
        )
    return "SELECT\n  " + ",\n  ".join(parts) + f"\nFROM {table}"


def sql_familia(table: str) -> str:
    return f"""
SELECT
  cast(familia AS string) AS familia,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM {table}
GROUP BY 1
ORDER BY n DESC
""".strip()


def sql_periodo(table: str) -> str:
    return f"""
SELECT
  cast(num_periodo_sem AS string) AS num_periodo_sem,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM {table}
GROUP BY 1
ORDER BY n DESC
""".strip()


def sql_overlap(table_a: str, table_b: str) -> str:
    return f"""
SELECT
  (SELECT count(*) FROM {table_a}) AS n_a,
  (SELECT count(*) FROM {table_b}) AS n_b,
  (SELECT ndv(id_master) FROM {table_a}) AS ndv_a,
  (SELECT ndv(id_master) FROM {table_b}) AS ndv_b,
  (
    SELECT count(*)
    FROM {table_a} a
    INNER JOIN {table_b} b
      ON a.id_master = b.id_master
  ) AS n_inner_join,
  (
    SELECT ndv(a.id_master)
    FROM {table_a} a
    INNER JOIN {table_b} b
      ON a.id_master = b.id_master
  ) AS ndv_overlap
""".strip()


def sql_muestra(table: str, columns: Iterable[str], n: int = 150000, p: float = 0.20) -> str:
    cols = ",\n  ".join(columns)
    return f"""
SELECT
  {cols}
FROM {table}
WHERE rand() < {p}
LIMIT {n}
""".strip()


def melt_wide_profile(wide: pd.DataFrame, n_rows_col: str = "n_rows") -> pd.DataFrame:
    """Convierte el SELECT ancho de Impala a formato largo (variable, metrica)."""
    if wide is None or len(wide) == 0:
        return pd.DataFrame(columns=["variable", "metric", "value"])
    row = wide.iloc[0].to_dict()
    n_rows = row.get(n_rows_col, np.nan)
    records = []
    for key, value in row.items():
        if key == n_rows_col:
            continue
        if "__" not in str(key):
            continue
        variable, metric = str(key).rsplit("__", 1)
        records.append({"variable": variable, "metric": metric, "value": value, "n_rows": n_rows})
    return pd.DataFrame.from_records(records)


def align_samples(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Homologa nombres y tipos para que las dos muestras sean comparables."""
    a = df_a.copy()
    b = df_b.copy()
    b = b.rename(columns={"esperanza_efe_fin": "esperanza_efe"})

    for frame in (a, b):
        if "num_periodo_sem" in frame.columns:
            frame["num_periodo_sem"] = frame["num_periodo_sem"].astype("string")
        if "familia" in frame.columns:
            frame["familia"] = frame["familia"].astype("string")
        if "plz_estimado_asignado" in frame.columns:
            frame["plz_estimado_asignado"] = pd.to_numeric(
                frame["plz_estimado_asignado"], errors="coerce"
            )
        if "bhs_score" in frame.columns:
            frame["bhs_score"] = pd.to_numeric(frame["bhs_score"], errors="coerce")

    return a, b


def _finite(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    return arr[np.isfinite(arr)]


def _zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.mean(arr)
    sd = np.std(arr, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.min(arr), np.max(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def detect_scale(a: np.ndarray, b: np.ndarray) -> dict:
    """Detecta si se ven la misma forma pero en distinta unidad (0-1 vs 0-100, etc.)."""
    out = {
        "median_ratio": np.nan,
        "mean_ratio": np.nan,
        "range_ratio": np.nan,
        "iqr_ratio": np.nan,
        "possible_factor": None,
        "hint": "sin evidencia clara de factor de escala común",
    }
    if len(a) < 30 or len(b) < 30:
        out["hint"] = "muestra insuficiente para diagnosticar escala"
        return out

    med_a, med_b = np.median(a), np.median(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    rng_a, rng_b = np.ptp(a), np.ptp(b)
    iqr_a = np.subtract(*np.percentile(a, [75, 25]))
    iqr_b = np.subtract(*np.percentile(b, [75, 25]))

    def _ratio(x, y):
        if y == 0 or not np.isfinite(x) or not np.isfinite(y):
            return np.nan
        return float(x / y)

    out["median_ratio"] = _ratio(med_a, med_b)
    out["mean_ratio"] = _ratio(mean_a, mean_b)
    out["range_ratio"] = _ratio(rng_a, rng_b)
    out["iqr_ratio"] = _ratio(iqr_a, iqr_b)

    a01 = float(np.nanmin(a) >= -0.05 and np.nanmax(a) <= 1.05)
    b01 = float(np.nanmin(b) >= -0.05 and np.nanmax(b) <= 1.05)
    a100 = float(np.nanmin(a) >= -0.5 and np.nanmax(a) <= 105)
    b100 = float(np.nanmin(b) >= -0.5 and np.nanmax(b) <= 105)

    ratios = [
        v
        for v in (out["median_ratio"], out["mean_ratio"], out["iqr_ratio"])
        if v is not None and np.isfinite(v) and v > 0
    ]
    if ratios:
        ratio = float(np.median(ratios))
        for factor in SCALE_FACTORS:
            if abs(math.log10(ratio / factor)) < 0.15:
                out["possible_factor"] = factor
                if factor < 1:
                    inv = 1.0 / factor
                    out["hint"] = (
                        f"posible diferencia de escala: A ≈ B / {inv:g} "
                        f"(razón A/B ≈ {ratio:.3g})"
                    )
                else:
                    out["hint"] = (
                        f"posible diferencia de escala: A ≈ B × {factor:g} "
                        f"(razón A/B ≈ {ratio:.3g})"
                    )
                break
        else:
            if 0.7 <= ratio <= 1.4:
                out["hint"] = f"escalas compatibles (razón ≈ {ratio:.3g})"
            else:
                out["hint"] = f"escalas distintas (razón ≈ {ratio:.3g}), no es un factor redondo típico"

    if a01 and b100 and not b01:
        out["hint"] = "A parece proporción [0,1] y B porcentaje [0,100]"
        out["possible_factor"] = 0.01
    elif b01 and a100 and not a01:
        out["hint"] = "B parece proporción [0,1] y A porcentaje [0,100]"
        out["possible_factor"] = 100.0

    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if len(expected) < 50 or len(actual) < 50:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return np.nan
    # Extiende extremos; si no, valores fuera de rango (p. ej. 0-1 vs 0-100) no entran.
    breaks = breaks.astype(float)
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    e_pct = np.histogram(expected, bins=breaks)[0] / len(expected)
    a_pct = np.histogram(actual, bins=breaks)[0] / len(actual)
    e_pct = np.clip(e_pct, 1e-4, None)
    a_pct = np.clip(a_pct, 1e-4, None)
    e_pct = e_pct / e_pct.sum()
    a_pct = a_pct / a_pct.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def interpret_psi(value: float) -> str:
    if not np.isfinite(value):
        return "PSI no calculable"
    if value < PSI_OK:
        return "similar (PSI < 0.10)"
    if value < PSI_WARN:
        return "cambio moderado (0.10–0.25)"
    return "cambio fuerte (PSI ≥ 0.25)"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    n = len(a) + len(b) - 2
    if n <= 0:
        return np.nan
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / n
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def compare_numeric_series(
    s_a: pd.Series,
    s_b: pd.Series,
    name: str,
    max_test_n: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Compara una variable numérica: nulos, escala, forma y tests."""
    rng = rng or np.random.default_rng(42)
    a_all = pd.to_numeric(s_a, errors="coerce")
    b_all = pd.to_numeric(s_b, errors="coerce")
    a = _finite(a_all)
    b = _finite(b_all)

    out = {
        "variable": name,
        "n_a": int(len(s_a)),
        "n_b": int(len(s_b)),
        "null_pct_a": float(a_all.isna().mean()) if len(s_a) else np.nan,
        "null_pct_b": float(b_all.isna().mean()) if len(s_b) else np.nan,
        "zero_pct_a": float(np.mean(a == 0)) if len(a) else np.nan,
        "zero_pct_b": float(np.mean(b == 0)) if len(b) else np.nan,
        "mean_a": float(np.mean(a)) if len(a) else np.nan,
        "mean_b": float(np.mean(b)) if len(b) else np.nan,
        "median_a": float(np.median(a)) if len(a) else np.nan,
        "median_b": float(np.median(b)) if len(b) else np.nan,
        "std_a": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan,
        "std_b": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
        "min_a": float(np.min(a)) if len(a) else np.nan,
        "min_b": float(np.min(b)) if len(b) else np.nan,
        "max_a": float(np.max(a)) if len(a) else np.nan,
        "max_b": float(np.max(b)) if len(b) else np.nan,
        "cv_a": np.nan,
        "cv_b": np.nan,
        "ks_stat": np.nan,
        "ks_p": np.nan,
        "ks_stat_z": np.nan,
        "ks_p_z": np.nan,
        "mw_stat": np.nan,
        "mw_p": np.nan,
        "levene_stat": np.nan,
        "levene_p": np.nan,
        "wasserstein": np.nan,
        "wasserstein_z": np.nan,
        "psi": np.nan,
        "psi_z": np.nan,
        "cohens_d": np.nan,
        "median_rel_diff": np.nan,
    }
    out.update({f"scale_{k}": v for k, v in detect_scale(a, b).items()})

    if len(a) > 1 and out["mean_a"] not in (0, np.nan):
        out["cv_a"] = out["std_a"] / out["mean_a"] if out["mean_a"] != 0 else np.nan
    if len(b) > 1 and out["mean_b"] not in (0, np.nan):
        out["cv_b"] = out["std_b"] / out["mean_b"] if out["mean_b"] != 0 else np.nan
    if np.isfinite(out["median_a"]) and out["median_a"] != 0 and np.isfinite(out["median_b"]):
        out["median_rel_diff"] = (out["median_b"] - out["median_a"]) / abs(out["median_a"])

    if len(a) < 30 or len(b) < 30 or stats is None:
        out["psi"] = psi(a, b) if len(a) and len(b) else np.nan
        out["psi_label"] = interpret_psi(out["psi"])
        return out

    aa = a if len(a) <= max_test_n else rng.choice(a, size=max_test_n, replace=False)
    bb = b if len(b) <= max_test_n else rng.choice(b, size=max_test_n, replace=False)
    za, zb = _zscore(aa), _zscore(bb)
    # En enteros de baja cardinalidad (bhs_score, plazos) el z-score inventa un
    # corrimiento de forma que no existe; ahí nos quedamos con el PSI/KS crudo.
    low_card = len(np.unique(np.round(aa, 8))) <= 20 and len(np.unique(np.round(bb, 8))) <= 20

    ks = stats.ks_2samp(aa, bb, method="asymp")
    ks_z = stats.ks_2samp(za, zb, method="asymp")
    mw = stats.mannwhitneyu(aa, bb, alternative="two-sided")
    lev = stats.levene(aa, bb)

    out.update(
        {
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "ks_stat_z": float(ks.statistic if low_card else ks_z.statistic),
            "ks_p_z": float(ks.pvalue if low_card else ks_z.pvalue),
            "mw_stat": float(mw.statistic),
            "mw_p": float(mw.pvalue),
            "levene_stat": float(lev.statistic),
            "levene_p": float(lev.pvalue),
            "wasserstein": float(stats.wasserstein_distance(aa, bb)),
            "wasserstein_z": float(stats.wasserstein_distance(za, zb)),
            "psi": psi(a, b),
            "psi_z": psi(a, b) if low_card else psi(za, zb),
            "cohens_d": cohens_d(aa, bb),
        }
    )
    out["psi_label"] = interpret_psi(out["psi"])
    out["psi_z_label"] = interpret_psi(out["psi_z"])
    return out


def compare_all_numeric(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    columns = list(columns or CANONICAL_NUMERIC)
    rows = []
    for col in columns:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        rows.append(compare_numeric_series(df_a[col], df_b[col], col))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Ordena por desacuerdo de forma (PSI estandarizado, luego KS-z)
    sort_cols = [c for c in ("psi_z", "ks_stat_z", "psi") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False, na_position="last")
    return out.reset_index(drop=True)


def compare_categorical(s_a: pd.Series, s_b: pd.Series, name: str, top_n: int = 30) -> dict:
    a = s_a.astype("string")
    b = s_b.astype("string")
    pa = a.value_counts(dropna=False, normalize=True)
    pb = b.value_counts(dropna=False, normalize=True)
    levels = pa.index.union(pb.index)
    ta = a.value_counts(dropna=False).reindex(levels, fill_value=0)
    tb = b.value_counts(dropna=False).reindex(levels, fill_value=0)
    table = np.vstack([ta.to_numpy(), tb.to_numpy()])

    result = {
        "variable": name,
        "n_levels_a": int(a.nunique(dropna=False)),
        "n_levels_b": int(b.nunique(dropna=False)),
        "null_pct_a": float(a.isna().mean()),
        "null_pct_b": float(b.isna().mean()),
        "chi2": np.nan,
        "chi2_p": np.nan,
        "cramers_v": np.nan,
        "tv_distance": float(0.5 * (pa.reindex(levels, fill_value=0) - pb.reindex(levels, fill_value=0)).abs().sum()),
    }
    if stats is not None and table.shape[1] >= 2 and table.sum() > 0:
        chi2, p, _, _ = stats.chi2_contingency(table)
        n = table.sum()
        r, c = table.shape
        k = min(r - 1, c - 1)
        result["chi2"] = float(chi2)
        result["chi2_p"] = float(p)
        result["cramers_v"] = float(math.sqrt(chi2 / (n * k))) if k and n else np.nan

    mix = (
        pd.DataFrame({"pct_a": pa, "pct_b": pb})
        .fillna(0)
        .assign(abs_diff=lambda d: (d["pct_a"] - d["pct_b"]).abs())
        .sort_values("abs_diff", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "level"})
    )
    result["top_shifts"] = mix
    return result


def verdict_row(row: pd.Series) -> str:
    """Lectura práctica: p-value en N grande engaña; usamos PSI / KS / escala."""
    hints = []
    factor = row.get("scale_possible_factor")
    if pd.notna(factor):
        if factor < 1:
            hints.append(f"escala A ≈ B/{1.0/float(factor):g}")
        else:
            hints.append(f"escala A ≈ B×{float(factor):g}")
    psi_v = row.get("psi")
    psi_z = row.get("psi_z")
    ks_z = row.get("ks_stat_z")
    if pd.notna(psi_v) and psi_v >= PSI_WARN and (pd.isna(psi_z) or psi_z >= PSI_OK):
        if pd.notna(psi_z) and psi_z < PSI_OK:
            hints.append("misma forma, distinta escala/ubicación")
        else:
            hints.append("distribución distinta")
    elif pd.notna(psi_z) and psi_z >= PSI_WARN:
        hints.append("forma distinta aun estandarizando")
    elif pd.notna(psi_v) and psi_v < PSI_OK and (pd.isna(psi_z) or psi_z < PSI_OK):
        hints.append("se parecen")
    elif pd.notna(ks_z) and ks_z < 0.05:
        hints.append("forma similar (KS-z bajo)")
    return "; ".join(hints) if hints else "revisar a detalle"


def add_verdict(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["lectura"] = out.apply(verdict_row, axis=1)
    return out


def make_demo_frames(n: int = 8000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Datos sintéticos para probar el flujo sin Impala."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)

    def _frame(scale_tasa: float, cltv_shift: float, extra: bool) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "id_master": ids,
                "plz_estimado_asignado": rng.integers(4, 36, size=n),
                "cdp": rng.beta(2, 5, size=n),
                "ticket_promedio": rng.lognormal(6.5, 0.4, size=n),
                "cdp_ajustada": rng.beta(2.2, 5, size=n),
                "monto_topado": rng.lognormal(8.2, 0.35, size=n),
                "plz_estimado": rng.normal(18, 6, size=n).clip(1, 48),
                "tasa_estimada": rng.normal(0.35, 0.08, size=n).clip(0.05, 0.9) * scale_tasa,
                "mto_capital": rng.lognormal(8.0, 0.4, size=n),
                "factor_prepago": rng.beta(2, 8, size=n),
                "tasa_estimada_ajustada": rng.normal(0.32, 0.07, size=n).clip(0.05, 0.9) * scale_tasa,
                "plz_estimado_ajustado": rng.normal(16, 5, size=n).clip(1, 48),
                "intereses": rng.lognormal(6.0, 0.5, size=n),
                "bhs_score": rng.integers(1, 10, size=n),
                "familia": rng.choice(["A", "B", "C", "D"], size=n, p=[0.4, 0.3, 0.2, 0.1]),
                "reservas_esperadas": rng.lognormal(5.5, 0.6, size=n),
                "costo_calles_esperadas": rng.lognormal(3.2, 0.5, size=n),
                "costo_llamada_esperadas": rng.lognormal(2.8, 0.4, size=n),
                "costo_sms_esperadas": rng.lognormal(1.5, 0.5, size=n),
                "positivos": rng.lognormal(7.2, 0.45, size=n),
                "negativos": rng.lognormal(5.8, 0.5, size=n),
                "value": rng.normal(100, 40, size=n),
                "esperanza_efe": rng.normal(80 + cltv_shift, 30, size=n),
                "cltv": rng.normal(120 + cltv_shift, 50, size=n),
                "num_periodo_sem": rng.choice(["202501", "202452"], size=n),
            }
        )
        # ~3% nulos en un par de columnas
        mask = rng.random(n) < 0.03
        df.loc[mask, "cdp"] = np.nan
        if extra:
            df["intereses2"] = df["intereses"] * 0.92
            df = df.rename(columns={"esperanza_efe": "esperanza_efe_fin"})
        return df

    return _frame(1.0, 0.0, extra=False), _frame(100.0, 8.0, extra=True)


DISPLAY_COLS = [
    "variable",
    "null_pct_a",
    "null_pct_b",
    "median_a",
    "median_b",
    "mean_a",
    "mean_b",
    "min_a",
    "max_a",
    "min_b",
    "max_b",
    "scale_hint",
    "psi",
    "psi_label",
    "psi_z",
    "ks_stat",
    "ks_stat_z",
    "cohens_d",
    "median_rel_diff",
    "lectura",
]
