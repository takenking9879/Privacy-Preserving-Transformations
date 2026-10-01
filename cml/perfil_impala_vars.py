# Pega esta celda. Mismo truco que JP: min/max/avg/appx_median en Impala (tabla completa).
from IPython import get_ipython
import pandas as pd

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
