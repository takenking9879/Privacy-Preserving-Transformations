# Pega esta celda en CML. Usa el magic de Impala (%%sql) por debajo.
from IPython import get_ipython
from IPython.display import display, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ip = get_ipython()
if ip is not None:
    try:
        ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

TABLE_A = "ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501"
TABLE_B = "ws_ektcomd_analitica.tt_1034848_cltv_futuros_efectivo_202523"
LABEL_A, LABEL_B = "202501", "202523"
N = 100000
ALPHA = 0.05

NUM = [
    "plz_estimado_asignado", "cdp", "ticket_promedio", "cdp_ajustada",
    "monto_topado", "plz_estimado", "tasa_estimada", "mto_capital",
    "factor_prepago", "tasa_estimada_ajustada", "plz_estimado_ajustado",
    "intereses", "bhs_score", "reservas_esperadas", "costo_calles_esperadas",
    "costo_llamada_esperadas", "costo_sms_esperadas", "positivos",
    "negativos", "value", "esperanza_efe", "cltv",
]


def run_sql(query, name):
    # El %%sql de Impala/jupysql no acepta -o; se usa: %%sql df <<
    ip = get_ipython()
    ip.run_cell_magic("sql", f"{name} <<", query)
    out = ip.user_ns[name]
    return out.DataFrame() if hasattr(out, "DataFrame") else out


df_a = run_sql(f"""
SELECT
  cast(plz_estimado_asignado AS double) AS plz_estimado_asignado,
  cdp, ticket_promedio, cdp_ajustada, monto_topado, plz_estimado,
  tasa_estimada, mto_capital, factor_prepago, tasa_estimada_ajustada,
  plz_estimado_ajustado, intereses, bhs_score,
  cast(familia AS string) AS familia,
  reservas_esperadas, costo_calles_esperadas, costo_llamada_esperadas,
  costo_sms_esperadas, positivos, negativos, value,
  esperanza_efe, cltv,
  cast(num_periodo_sem AS string) AS num_periodo_sem
FROM {TABLE_A}
WHERE rand() < 0.2
LIMIT {N}
""", "df_a")

df_b = run_sql(f"""
SELECT
  cast(plz_estimado_asignado AS double) AS plz_estimado_asignado,
  cast(cdp AS double) AS cdp,
  ticket_promedio,
  cast(cdp_ajustada AS double) AS cdp_ajustada,
  monto_topado, plz_estimado,
  tasa_estimada, mto_capital, factor_prepago, tasa_estimada_ajustada,
  plz_estimado_ajustado, intereses, bhs_score,
  cast(familia AS string) AS familia,
  reservas_esperadas, costo_calles_esperadas, costo_llamada_esperadas,
  costo_sms_esperadas, positivos, negativos, value,
  esperanza_efe, cltv,
  cast(num_periodo_sem AS string) AS num_periodo_sem
FROM {TABLE_B}
WHERE rand() < 0.2
LIMIT {N}
""", "df_b")

print(f"n {LABEL_A}={len(df_a):,}  n {LABEL_B}={len(df_b):,}")
print("solo en B: intereses2, esperanza_efe_ (no se comparan)")

print("\n=== NULOS ===")
for col in NUM + ["familia", "num_periodo_sem"]:
    pa = df_a[col].isna().mean() * 100
    pb = df_b[col].isna().mean() * 100
    print(f"{col:30s}  {LABEL_A} {pa:5.2f}%   {LABEL_B} {pb:5.2f}%")

print(f"\n=== NUMERICAS  (KS, alpha={ALPHA}) ===")
for col in NUM:
    a = pd.to_numeric(df_a[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(df_b[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 30 or len(b) < 30:
        print(f"{col}: muy pocos datos, skip")
        continue
    ratio = (a.median() / b.median()) if b.median() != 0 else np.nan
    ks = stats.ks_2samp(a, b)
    sig = "Si es un cambio significativo" if ks.pvalue < ALPHA else "No es un cambio significativo"
    print(
        f"{col}: p={ks.pvalue:.4g}  KS={ks.statistic:.3f}  "
        f"med {LABEL_A}={a.median():.4g} [{a.min():.4g}, {a.max():.4g}]  "
        f"med {LABEL_B}={b.median():.4g} [{b.min():.4g}, {b.max():.4g}]  "
        f"razon_medianas={ratio:.3g}  → {sig}"
    )

print(f"\n=== CATEGORICAS  (chi2, alpha={ALPHA}) ===")
for col in ["familia", "num_periodo_sem", "bhs_score"]:
    ta = df_a[col].astype("string").value_counts(dropna=False)
    tb = df_b[col].astype("string").value_counts(dropna=False)
    idx = ta.index.union(tb.index)
    table = np.vstack([ta.reindex(idx, fill_value=0), tb.reindex(idx, fill_value=0)])
    chi2, p, _, _ = stats.chi2_contingency(table)
    sig = "Si es un cambio significativo" if p < ALPHA else "No es un cambio significativo"
    print(f"{col}: p={p:.4g}  chi2={chi2:.2f}  → {sig}")
    mix = pd.DataFrame({LABEL_A: ta / ta.sum(), LABEL_B: tb / tb.sum()}).fillna(0)
    print(mix.assign(diff=lambda d: d[LABEL_B] - d[LABEL_A]).sort_values("diff", key=lambda s: s.abs(), ascending=False).head(8))

print("\n=== GRAFICAS ===")
ncols = 4
nrows = int(np.ceil(len(NUM) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
for ax, col in zip(axes.ravel(), NUM):
    a = pd.to_numeric(df_a[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = pd.to_numeric(df_b[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    both = np.concatenate([x for x in (a, b) if len(x)])
    if len(both) == 0:
        ax.set_title(f"{col} (sin datos)")
        continue
    lo, hi = np.nanpercentile(both, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
        if lo == hi:
            lo, hi = lo - 1, hi + 1
    ax.hist(a, bins=40, density=True, alpha=0.45, range=(lo, hi), label=LABEL_A)
    ax.hist(b, bins=40, density=True, alpha=0.45, range=(lo, hi), label=LABEL_B)
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)
for ax in axes.ravel()[len(NUM):]:
    ax.axis("off")
fig.suptitle(f"Densidades {LABEL_A} vs {LABEL_B} (colas 1-99)", y=1.01)
try:
    fig.tight_layout()
except Exception:
    pass
png = "/tmp/comparacion_cltv.png"
fig.savefig(png, dpi=110, bbox_inches="tight")
display(fig)
display(Image(png))
plt.show()
print(f"grafica guardada en {png}")
