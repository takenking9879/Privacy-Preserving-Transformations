# Pega esta celda. Mismo truco que JP: min/max/avg/appx_median en Impala (tabla completa)
# + histogramas overlay de esas mismas variables.
from IPython import get_ipython
from IPython.display import display, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ip = get_ipython()
if ip is not None:
    try:
        ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

TABLE_A = "ws_ektcomd_analitica.tt_1117091_CLTV_futuros_efectivo_202501"
TABLE_B = "ws_ektcomd_analitica.tt_1034848_cltv_futuros_efectivo_202523"

# Las que empeoraron o se ven con otra definición
VARS = [
    "cdp",
    "cdp_ajustada",
    "costo_calles_esperadas",
    "costo_llamada_esperadas",
    "costo_sms_esperadas",
    "bhs_score",
    "esperanza_efe",
    "cltv",
]


def run_sql(query, name):
    ip = get_ipython()
    ip.run_cell_magic("sql", f"{name} <<", query)
    out = ip.user_ns[name]
    return out.DataFrame() if hasattr(out, "DataFrame") else out


tablas = []
for var in VARS:
    df = run_sql(
        f"""
SELECT
  '202501' AS tabla,
  min({var}) AS min_v,
  max({var}) AS max_v,
  avg({var}) AS avg_v,
  appx_median({var}) AS median_v
FROM {TABLE_A}
UNION ALL
SELECT
  '202523',
  min({var}),
  max({var}),
  avg({var}),
  appx_median({var})
FROM {TABLE_B}
""",
        f"perfil_{var}",
    )
    df.insert(1, "variable", var)
    print(f"\n=== {var} ===")
    display(df)
    tablas.append(df)

resumen = pd.concat(tablas, ignore_index=True)
print("\n=== RESUMEN ===")
display(resumen)

cols = ", ".join(f"cast({v} AS double) AS {v}" for v in VARS)
df_a = run_sql(
    f"SELECT {cols} FROM {TABLE_A} WHERE rand() < 0.2 LIMIT 100000",
    "df_a_perfil",
)
df_b = run_sql(
    f"SELECT {cols} FROM {TABLE_B} WHERE rand() < 0.2 LIMIT 100000",
    "df_b_perfil",
)

print("\n=== GRAFICAS ===")
ncols = 4
nrows = int(np.ceil(len(VARS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4 * nrows))
for ax, col in zip(np.ravel(axes), VARS):
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
    ax.hist(a, bins=40, density=True, alpha=0.45, range=(lo, hi), label="202501")
    ax.hist(b, bins=40, density=True, alpha=0.45, range=(lo, hi), label="202523")
    ax.set_title(col, fontsize=9)
    ax.legend(fontsize=7)
for ax in np.ravel(axes)[len(VARS):]:
    ax.axis("off")
fig.suptitle("Densidades 202501 vs 202523 (colas 1-99)", y=1.01)
try:
    fig.tight_layout()
except Exception:
    pass
png = "/tmp/perfil_impala_vars.png"
fig.savefig(png, dpi=110, bbox_inches="tight")
display(fig)
display(Image(png))
plt.show()
print(f"grafica guardada en {png}")
