"""Genera comparacion_cltv_impala.ipynb con escaping correcto."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# Comparación CLTV futuros efectivo: `202501` vs `202452`

Contrasta dos tablas de Impala para ver si se parecen lo suficiente (nulos, escala, forma y un overlay de densidades).

| Snapshot | Tabla |
|---|---|
| **A** 202501 | `ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501` |
| **B** 202452 | `ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452` |

Son semanas consecutivas (52 de 2024 vs 1 de 2025): la población no debería moverse tanto. Si algo se ve muy distinto, casi seguro es definición / pipeline, no el cliente.

**Cómo usarlo en CML**
1. Sube este notebook y `comparacion_cltv_helpers.py` al mismo directorio (o a `/home/cdsw`).
2. Corre las celdas `%%sql` con el magic de Impala que ya usas.
3. Si tu magic soporta `-o df`, las celdas ya lo traen. Si no:
   - `ipython-sql` / jupysql: `df = _.DataFrame()` en la celda de abajo.
   - o `%%sql df <<` en lugar de `%%sql -o df`.
4. Impala calcula perfiles sobre **toda** la tabla. Python solo usa una **muestra** para tests y gráficas.

**Lectura importante:** con cientos de miles de filas casi todo sale `p < 0.05`. No te guíes por el p-value. Mira PSI, KS, Cohen's d y si hay un factor de escala (0–1 vs 0–100)."""
)

code(
    """from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Si subiste el .py junto al notebook, úsalo.
HERE = Path.cwd()
for candidate in (HERE, HERE / "cml", Path("/home/cdsw"), Path("/home/cdsw/cml")):
    if (candidate / "comparacion_cltv_helpers.py").exists():
        sys.path.insert(0, str(candidate))
        break

from comparacion_cltv_helpers import (
    CANONICAL_NUMERIC,
    CATEGORICAL,
    DISPLAY_COLS,
    add_verdict,
    align_samples,
    compare_all_numeric,
    compare_categorical,
    make_demo_frames,
    melt_wide_profile,
    sql_count,
    sql_nulos,
    sql_perfil_numerico,
)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

TABLE_A = "ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501"
TABLE_B = "ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452"
LABEL_A = "202501"
LABEL_B = "202452"

# False = Impala real. True = data sintética para probar Python sin cluster.
USE_DEMO = False

COLS_A = [
    "id_master", "plz_estimado_asignado", "cdp", "ticket_promedio", "cdp_ajustada",
    "monto_topado", "plz_estimado", "tasa_estimada", "mto_capital", "factor_prepago",
    "tasa_estimada_ajustada", "plz_estimado_ajustado", "intereses", "bhs_score",
    "familia", "reservas_esperadas", "costo_calles_esperadas", "costo_llamada_esperadas",
    "costo_sms_esperadas", "positivos", "negativos", "value", "esperanza_efe",
    "cltv", "num_periodo_sem",
]
COLS_B = [
    "id_master", "plz_estimado_asignado", "cdp", "ticket_promedio", "cdp_ajustada",
    "monto_topado", "plz_estimado", "tasa_estimada", "mto_capital", "factor_prepago",
    "tasa_estimada_ajustada", "plz_estimado_ajustado", "intereses", "intereses2",
    "bhs_score", "familia", "reservas_esperadas", "costo_calles_esperadas",
    "costo_llamada_esperadas", "costo_sms_esperadas", "positivos", "negativos",
    "value", "esperanza_efe_fin", "cltv", "num_periodo_sem",
]
NUM_A = [c for c in COLS_A if c not in ("id_master", "familia", "num_periodo_sem")]
NUM_B = [c for c in COLS_B if c not in ("id_master", "familia", "num_periodo_sem")]

print("Queries listas. Si quieres copiarlas a una celda %%sql:")
print(sql_count(TABLE_A))"""
)

md(
    """## 1. Volúmenes

Si `-o` no crea el DataFrame, corre `count_a = _.DataFrame()` (o el equivalente de tu magic) en una celda Python inmediata."""
)

code(
    """%%sql -o count_a
SELECT
  count(*) AS n_rows,
  ndv(id_master) AS n_id_master,
  count(*) - count(id_master) AS n_id_master_null
FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501"""
)

code(
    """%%sql -o count_b
SELECT
  count(*) AS n_rows,
  ndv(id_master) AS n_id_master,
  count(*) - count(id_master) AS n_id_master_null
FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452"""
)

code(
    """if USE_DEMO:
    print("USE_DEMO=True: se saltan los counts de Impala")
else:
    display(pd.concat(
        [count_a.assign(tabla=LABEL_A), count_b.assign(tabla=LABEL_B)],
        ignore_index=True,
    ))"""
)

md(
    """## 2. Nulos (tabla completa, un scan por snapshot)

Diferencias de tipo que ya vimos en el esquema:
- `plz_estimado_asignado`: double vs int
- `num_periodo_sem`: string vs int
- `esperanza_efe` vs `esperanza_efe_fin`
- `intereses2` solo existe en 202452"""
)

code(
    """print(sql_nulos(TABLE_A, COLS_A))
print("\\n----- B -----\\n")
print(sql_nulos(TABLE_B, COLS_B))"""
)

code(
    """%%sql -o nulos_a
SELECT
  count(*) AS n_rows,
  (count(*) - count(id_master)) AS id_master__nulls,
  (count(*) - count(plz_estimado_asignado)) AS plz_estimado_asignado__nulls,
  (count(*) - count(cdp)) AS cdp__nulls,
  (count(*) - count(ticket_promedio)) AS ticket_promedio__nulls,
  (count(*) - count(cdp_ajustada)) AS cdp_ajustada__nulls,
  (count(*) - count(monto_topado)) AS monto_topado__nulls,
  (count(*) - count(plz_estimado)) AS plz_estimado__nulls,
  (count(*) - count(tasa_estimada)) AS tasa_estimada__nulls,
  (count(*) - count(mto_capital)) AS mto_capital__nulls,
  (count(*) - count(factor_prepago)) AS factor_prepago__nulls,
  (count(*) - count(tasa_estimada_ajustada)) AS tasa_estimada_ajustada__nulls,
  (count(*) - count(plz_estimado_ajustado)) AS plz_estimado_ajustado__nulls,
  (count(*) - count(intereses)) AS intereses__nulls,
  (count(*) - count(bhs_score)) AS bhs_score__nulls,
  (count(*) - count(familia)) AS familia__nulls,
  (count(*) - count(reservas_esperadas)) AS reservas_esperadas__nulls,
  (count(*) - count(costo_calles_esperadas)) AS costo_calles_esperadas__nulls,
  (count(*) - count(costo_llamada_esperadas)) AS costo_llamada_esperadas__nulls,
  (count(*) - count(costo_sms_esperadas)) AS costo_sms_esperadas__nulls,
  (count(*) - count(positivos)) AS positivos__nulls,
  (count(*) - count(negativos)) AS negativos__nulls,
  (count(*) - count(value)) AS value__nulls,
  (count(*) - count(esperanza_efe)) AS esperanza_efe__nulls,
  (count(*) - count(cltv)) AS cltv__nulls,
  (count(*) - count(num_periodo_sem)) AS num_periodo_sem__nulls
FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501"""
)

code(
    """%%sql -o nulos_b
SELECT
  count(*) AS n_rows,
  (count(*) - count(id_master)) AS id_master__nulls,
  (count(*) - count(plz_estimado_asignado)) AS plz_estimado_asignado__nulls,
  (count(*) - count(cdp)) AS cdp__nulls,
  (count(*) - count(ticket_promedio)) AS ticket_promedio__nulls,
  (count(*) - count(cdp_ajustada)) AS cdp_ajustada__nulls,
  (count(*) - count(monto_topado)) AS monto_topado__nulls,
  (count(*) - count(plz_estimado)) AS plz_estimado__nulls,
  (count(*) - count(tasa_estimada)) AS tasa_estimada__nulls,
  (count(*) - count(mto_capital)) AS mto_capital__nulls,
  (count(*) - count(factor_prepago)) AS factor_prepago__nulls,
  (count(*) - count(tasa_estimada_ajustada)) AS tasa_estimada_ajustada__nulls,
  (count(*) - count(plz_estimado_ajustado)) AS plz_estimado_ajustado__nulls,
  (count(*) - count(intereses)) AS intereses__nulls,
  (count(*) - count(intereses2)) AS intereses2__nulls,
  (count(*) - count(bhs_score)) AS bhs_score__nulls,
  (count(*) - count(familia)) AS familia__nulls,
  (count(*) - count(reservas_esperadas)) AS reservas_esperadas__nulls,
  (count(*) - count(costo_calles_esperadas)) AS costo_calles_esperadas__nulls,
  (count(*) - count(costo_llamada_esperadas)) AS costo_llamada_esperadas__nulls,
  (count(*) - count(costo_sms_esperadas)) AS costo_sms_esperadas__nulls,
  (count(*) - count(positivos)) AS positivos__nulls,
  (count(*) - count(negativos)) AS negativos__nulls,
  (count(*) - count(value)) AS value__nulls,
  (count(*) - count(esperanza_efe_fin)) AS esperanza_efe__nulls,
  (count(*) - count(cltv)) AS cltv__nulls,
  (count(*) - count(num_periodo_sem)) AS num_periodo_sem__nulls
FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452"""
)

code(
    """if not USE_DEMO:
    na = melt_wide_profile(nulos_a).rename(columns={"value": LABEL_A})
    nb = melt_wide_profile(nulos_b).rename(columns={"value": LABEL_B})
    nulos = na.merge(nb, on=["variable", "metric"], how="outer")
    nulos = nulos[nulos["metric"] == "nulls"].copy()
    n_a = float(nulos_a.iloc[0]["n_rows"])
    n_b = float(nulos_b.iloc[0]["n_rows"])
    nulos["pct_a"] = nulos[LABEL_A] / n_a
    nulos["pct_b"] = nulos[LABEL_B] / n_b
    nulos["diff_pp"] = (nulos["pct_b"] - nulos["pct_a"]) * 100
    nulos = nulos.sort_values("diff_pp", key=lambda s: s.abs(), ascending=False)
    display(nulos[["variable", LABEL_A, LABEL_B, "pct_a", "pct_b", "diff_pp"]])"""
)

md(
    """## 3. Perfil numérico en Impala (media, mediana, colas)

`percentile_approx` y `appx_median` evitan un full sort. Un scan por tabla. La celda de abajo manda el SELECT generado por `sql_perfil_numerico` al magic `%%sql`."""
)

code(
    """SQL_PERFIL_A = sql_perfil_numerico(TABLE_A, NUM_A)
SQL_PERFIL_B = sql_perfil_numerico(TABLE_B, NUM_B)
print(SQL_PERFIL_A[:1500], "\\n...\\n")"""
)

code(
    """from IPython import get_ipython

ip = get_ipython()
if USE_DEMO:
    print("USE_DEMO=True: se salta el perfil Impala")
elif ip is None:
    raise RuntimeError("Necesitas un kernel IPython/CML para %%sql")
else:
    ip.run_cell_magic("sql", "-o perfil_a", SQL_PERFIL_A)
    ip.run_cell_magic("sql", "-o perfil_b", SQL_PERFIL_B)"""
)

code(
    """if not USE_DEMO:
    pa = melt_wide_profile(perfil_a)
    pb = melt_wide_profile(perfil_b)
    pb["variable"] = pb["variable"].replace({"esperanza_efe_fin": "esperanza_efe"})
    perfil = pa.merge(
        pb,
        on=["variable", "metric"],
        how="outer",
        suffixes=(f"_{LABEL_A}", f"_{LABEL_B}"),
    )
    wide = perfil.pivot(
        index="variable",
        columns="metric",
        values=[f"value_{LABEL_A}", f"value_{LABEL_B}"],
    )
    wide.columns = [f"{metric}_{side}" for side, metric in wide.columns]
    wide = wide.reset_index()
    for metric in ["mean", "median", "min", "max"]:
        ca, cb = f"{metric}_{LABEL_A}", f"{metric}_{LABEL_B}"
        if ca in wide.columns and cb in wide.columns:
            wide[f"ratio_{metric}"] = wide[ca] / wide[cb].replace(0, np.nan)
    display(wide.sort_values("variable"))"""
)

md("## 4. Cruce de `id_master` y categóricas")

code(
    """%%sql -o overlap_ids
SELECT
  (SELECT count(*) FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501) AS n_a,
  (SELECT count(*) FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452) AS n_b,
  (SELECT ndv(id_master) FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501) AS ndv_a,
  (SELECT ndv(id_master) FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452) AS ndv_b,
  (
    SELECT ndv(a.id_master)
    FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501 a
    INNER JOIN ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452 b
      ON a.id_master = b.id_master
  ) AS ndv_overlap"""
)

code(
    """%%sql -o familia_a
SELECT
  cast(familia AS string) AS familia,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501
GROUP BY 1
ORDER BY n DESC"""
)

code(
    """%%sql -o familia_b
SELECT
  cast(familia AS string) AS familia,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452
GROUP BY 1
ORDER BY n DESC"""
)

code(
    """%%sql -o periodo_a
SELECT
  cast(num_periodo_sem AS string) AS num_periodo_sem,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501
GROUP BY 1
ORDER BY n DESC"""
)

code(
    """%%sql -o periodo_b
SELECT
  cast(num_periodo_sem AS string) AS num_periodo_sem,
  count(*) AS n,
  count(*) / sum(count(*)) OVER () AS pct
FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452
GROUP BY 1
ORDER BY n DESC"""
)

code(
    """if not USE_DEMO:
    display(overlap_ids)
    fam = familia_a.merge(
        familia_b, on="familia", how="outer", suffixes=(f"_{LABEL_A}", f"_{LABEL_B}")
    ).fillna(0)
    fam["diff_pp"] = (fam[f"pct_{LABEL_B}"] - fam[f"pct_{LABEL_A}"]) * 100
    display(fam.sort_values("diff_pp", key=lambda s: s.abs(), ascending=False))
    per = periodo_a.merge(
        periodo_b, on="num_periodo_sem", how="outer", suffixes=(f"_{LABEL_A}", f"_{LABEL_B}")
    ).fillna(0)
    display(per)"""
)

md(
    """## 5. Muestra para tests y gráficas

No bajes la tabla completa. `rand() < 0.20` + `LIMIT` alcanza para KS / PSI / densidades.

Homologación al leer B: `esperanza_efe_fin AS esperanza_efe` y casts de tipos (`plz_estimado_asignado`, `num_periodo_sem`)."""
)

code(
    """%%sql -o df_a
SELECT
  id_master,
  cast(plz_estimado_asignado AS double) AS plz_estimado_asignado,
  cdp,
  ticket_promedio,
  cdp_ajustada,
  monto_topado,
  plz_estimado,
  tasa_estimada,
  mto_capital,
  factor_prepago,
  tasa_estimada_ajustada,
  plz_estimado_ajustado,
  intereses,
  bhs_score,
  cast(familia AS string) AS familia,
  reservas_esperadas,
  costo_calles_esperadas,
  costo_llamada_esperadas,
  costo_sms_esperadas,
  positivos,
  negativos,
  value,
  esperanza_efe,
  cltv,
  cast(num_periodo_sem AS string) AS num_periodo_sem
FROM ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501
WHERE rand() < 0.20
LIMIT 150000"""
)

code(
    """%%sql -o df_b
SELECT
  id_master,
  cast(plz_estimado_asignado AS double) AS plz_estimado_asignado,
  cdp,
  ticket_promedio,
  cdp_ajustada,
  monto_topado,
  plz_estimado,
  tasa_estimada,
  mto_capital,
  factor_prepago,
  tasa_estimada_ajustada,
  plz_estimado_ajustado,
  intereses,
  intereses2,
  bhs_score,
  cast(familia AS string) AS familia,
  reservas_esperadas,
  costo_calles_esperadas,
  costo_llamada_esperadas,
  costo_sms_esperadas,
  positivos,
  negativos,
  value,
  esperanza_efe_fin AS esperanza_efe,
  cltv,
  cast(num_periodo_sem AS string) AS num_periodo_sem
FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_202452
WHERE rand() < 0.20
LIMIT 150000"""
)

code(
    """if USE_DEMO:
    df_a, df_b = make_demo_frames(n=8000)
    print("Demo sintético: B tiene tasa en escala x100 y un shift suave en CLTV/esperanza")

df_a, df_b = align_samples(df_a, df_b)
print(df_a.shape, df_b.shape)
print("solo en A", sorted(set(df_a.columns) - set(df_b.columns)))
print("solo en B", sorted(set(df_b.columns) - set(df_a.columns)))
df_a.head(3)"""
)

md(
    """## 6. ¿Se parecen? escala + forma + significancia

Por variable numérica:
- **Nulos / ceros / min-max / mediana** — para cachar escala (0–1 vs 0–100, tasa vs %).
- **PSI** — estándar de crédito. `< 0.10` similar, `0.10–0.25` moderado, `≥ 0.25` cambio fuerte.
- **PSI y KS sobre z-score** — comparan la *forma* ignorando unidad y ubicación. Si el PSI crudo es alto y el z-score es bajo, son la misma distribución en otra escala.
- **Cohen's d y diff relativa de medianas** — tamaño del cambio, no solo si es “significativo”.
- **Mann-Whitney / Levene** — ubicación y dispersión; el p-value aquí es referencia, no veredicto."""
)

code(
    """summary = add_verdict(compare_all_numeric(df_a, df_b, CANONICAL_NUMERIC))
view = summary[[c for c in DISPLAY_COLS if c in summary.columns]].copy()
for c in ["null_pct_a", "null_pct_b"]:
    if c in view:
        view[c] = view[c] * 100
display(view)

print("\\nVariables que más se salen (PSI-z o escala rara):")
flag = summary[
    summary["lectura"].str.contains("distinta|escala|fuerte", case=False, na=False)
    | (summary["psi"] >= 0.10)
]
display(flag[[c for c in DISPLAY_COLS if c in flag.columns]])"""
)

code(
    """cat_reports = {}
for col in CATEGORICAL + ["bhs_score"]:
    if col in df_a.columns and col in df_b.columns:
        cat_reports[col] = compare_categorical(df_a[col], df_b[col], col)
        r = cat_reports[col]
        print(f"\\n=== {col} ===")
        print(
            f"niveles A/B={r['n_levels_a']}/{r['n_levels_b']}  "
            f"Cramér V={r['cramers_v']:.3f}  TV={r['tv_distance']:.3f}  p={r['chi2_p']:.3g}"
        )
        display(r["top_shifts"])"""
)

md(
    """## 7. Gráficas

Para cada variable: densidad cruda (cachar escala) y densidad en z-score (cachar forma). Si las crudas no se solapan pero las z-score sí, el problema es unidad, no la distribución."""
)

code(
    """import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

OUT_DIR = Path("comparacion_cltv_out")
OUT_DIR.mkdir(exist_ok=True)


def _xy(s):
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) > 80000:
        v = v.sample(80000, random_state=42)
    return v


def plot_var(col, q_lo=0.005, q_hi=0.995):
    a = _xy(df_a[col])
    b = _xy(df_b[col])
    if a.empty or b.empty:
        print(f"sin datos: {col}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    lo = min(a.quantile(q_lo), b.quantile(q_lo))
    hi = max(a.quantile(q_hi), b.quantile(q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))

    if HAS_SNS:
        sns.kdeplot(a, ax=axes[0], label=LABEL_A, clip=(lo, hi), common_norm=False)
        sns.kdeplot(b, ax=axes[0], label=LABEL_B, clip=(lo, hi), common_norm=False)
    else:
        axes[0].hist(a, bins=40, density=True, alpha=0.45, range=(lo, hi), label=LABEL_A)
        axes[0].hist(b, bins=40, density=True, alpha=0.45, range=(lo, hi), label=LABEL_B)
    axes[0].set_title(f"{col} — densidad cruda")
    axes[0].legend()

    za = (a - a.mean()) / (a.std() or 1)
    zb = (b - b.mean()) / (b.std() or 1)
    if HAS_SNS:
        sns.kdeplot(za, ax=axes[1], label=LABEL_A)
        sns.kdeplot(zb, ax=axes[1], label=LABEL_B)
    else:
        axes[1].hist(za, bins=40, density=True, alpha=0.45, range=(-3.5, 3.5), label=LABEL_A)
        axes[1].hist(zb, bins=40, density=True, alpha=0.45, range=(-3.5, 3.5), label=LABEL_B)
    axes[1].set_title(f"{col} — forma (z-score)")
    axes[1].legend()

    def _ecdf(x, ax, label):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=label, drawstyle="steps-post")

    _ecdf(a[(a >= lo) & (a <= hi)], axes[2], LABEL_A)
    _ecdf(b[(b >= lo) & (b <= hi)], axes[2], LABEL_B)
    axes[2].set_title(f"{col} — ECDF (colas recortadas)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{col}.png", dpi=120)
    plt.show()
    plt.close(fig)


priority = []
if "variable" in summary.columns:
    priority = summary["variable"].head(8).tolist()
for extra in ["cltv", "esperanza_efe", "tasa_estimada", "intereses", "cdp", "value"]:
    if extra in df_a.columns and extra not in priority:
        priority.append(extra)

for col in priority:
    plot_var(col)

print(f"PNG en {OUT_DIR.resolve()}")"""
)

code(
    """cols = [c for c in CANONICAL_NUMERIC if c in df_a.columns and c in df_b.columns]
n = len(cols)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), squeeze=False)
for i, col in enumerate(cols):
    ax = axes[i // ncols][i % ncols]
    a = _xy(df_a[col])
    b = _xy(df_b[col])
    if a.empty or b.empty:
        ax.set_title(col)
        continue
    za = (a - a.mean()) / (a.std() or 1)
    zb = (b - b.mean()) / (b.std() or 1)
    try:
        ax.boxplot([za, zb], tick_labels=[LABEL_A, LABEL_B], showfliers=False)
    except TypeError:
        ax.boxplot([za, zb], labels=[LABEL_A, LABEL_B], showfliers=False)
    ax.set_title(col, fontsize=9)
    ax.axhline(0, color="grey", lw=0.6)
for j in range(i + 1, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
fig.suptitle("Boxplots en z-score (forma/ubicación relativa, no la unidad original)", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "boxplots_zscore.png", dpi=120)
plt.show()"""
)

md(
    """## 8. Si hay overlap de clientes: comparación pareada

Útil solo si `ndv_overlap` es material. Si son vintages casi disjuntos, ignora esta parte."""
)

code(
    """key = "id_master"
paired_cols = [
    c
    for c in ["cltv", "esperanza_efe", "tasa_estimada", "intereses", "cdp", "value"]
    if c in df_a.columns
]
if key in df_a.columns:
    paired = df_a[[key] + paired_cols].merge(
        df_b[[key] + paired_cols],
        on=key,
        how="inner",
        suffixes=(f"_{LABEL_A}", f"_{LABEL_B}"),
    )
    print(f"ids en la muestra que cruzan: {len(paired):,}")
    if len(paired) >= 50:
        rows = []
        for c in paired_cols:
            x = pd.to_numeric(paired[f"{c}_{LABEL_A}"], errors="coerce")
            y = pd.to_numeric(paired[f"{c}_{LABEL_B}"], errors="coerce")
            m = x.notna() & y.notna()
            rows.append(
                {
                    "variable": c,
                    "n_par": int(m.sum()),
                    "corr": float(x[m].corr(y[m])) if m.sum() > 2 else np.nan,
                    "median_diff": float((y[m] - x[m]).median()) if m.sum() else np.nan,
                    "median_rel": float(((y[m] - x[m]) / x[m].replace(0, np.nan)).median())
                    if m.sum()
                    else np.nan,
                }
            )
        display(pd.DataFrame(rows))
    else:
        print("Poco overlap en la muestra; el JOIN de Impala de la sección 4 es el dato bueno.")"""
)

md(
    """## Cómo leer el resultado (para decidir si “se parecen lo suficiente”)

1. **Esquema / nulos primero.** Si una columna cambia de 0% a 30% null, no compares distribución todavía.
2. **Escala.** `tasa_*` o factores 0–1 vs 0–100 salen en `scale_hint` y en la densidad cruda. En ese caso el KS crudo va a gritar y el KS/PSI en z-score te dice si la forma sí es la misma.
3. **PSI < 0.10** en crudo (o en z-score si hubo rescale) es la barra práctica de “suficientemente parecido” en crédito.
4. **p-values** de KS/MW con 150k filas casi siempre son 0. No los uses como semáforo.
5. Columnas que no tienen par: `intereses2` solo en 202452. `esperanza_efe` ↔ `esperanza_efe_fin` ya se alineó.

Si quieres forzar la misma unidad antes de PSI (ejemplo tasas):

```python
# df_b["tasa_estimada"] = df_b["tasa_estimada"] / 100
# df_b["tasa_estimada_ajustada"] = df_b["tasa_estimada_ajustada"] / 100
```"""
)

nb.cells = cells
out = Path(__file__).with_name("comparacion_cltv_impala.ipynb")
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
