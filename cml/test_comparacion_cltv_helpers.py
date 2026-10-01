"""Pruebas locales de los helpers (sin Impala)."""

import numpy as np
import pandas as pd

from comparacion_cltv_helpers import (
    add_verdict,
    align_samples,
    compare_all_numeric,
    compare_categorical,
    detect_scale,
    make_demo_frames,
    melt_wide_profile,
    psi,
    sql_perfil_numerico,
)


def test_align_and_scale_detection():
    raw_a, raw_b = make_demo_frames(n=4000, seed=3)
    df_a, df_b = align_samples(raw_a, raw_b)
    assert "esperanza_efe" in df_b.columns
    assert "esperanza_efe_fin" not in df_b.columns
    assert "intereses2" in df_b.columns

    scale = detect_scale(df_a["tasa_estimada"].to_numpy(), df_b["tasa_estimada"].to_numpy())
    assert scale["possible_factor"] == 0.01 or abs(scale["median_ratio"] - 0.01) < 0.005


def test_similar_shape_after_zscore():
    raw_a, raw_b = make_demo_frames(n=5000, seed=1)
    df_a, df_b = align_samples(raw_a, raw_b)
    summary = add_verdict(compare_all_numeric(df_a, df_b))
    tasa = summary.set_index("variable").loc["tasa_estimada"]
    # Crudo debe verse distinto por escala x100; la forma estandarizada no.
    assert tasa["psi"] >= 0.10
    assert tasa["psi_z"] < 0.10
    assert "escala" in str(tasa["lectura"]).lower() or "forma" in str(tasa["lectura"]).lower()

    ticket = summary.set_index("variable").loc["ticket_promedio"]
    assert ticket["psi"] < 0.10


def test_psi_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert psi(x, x.copy()) < 0.02


def test_melt_and_sql_shape():
    sql = sql_perfil_numerico("db.t", ["cdp", "cltv"])
    assert "percentile_approx(cast(cdp AS double), 0.99)" in sql
    wide = pd.DataFrame(
        [{"n_rows": 100, "cdp__nulls": 3, "cdp__mean": 0.2, "cltv__nulls": 0, "cltv__mean": 10}]
    )
    long = melt_wide_profile(wide)
    assert set(long["variable"]) == {"cdp", "cltv"}
    assert {"nulls", "mean"} <= set(long["metric"])


def test_categorical():
    a = pd.Series(["A", "A", "B", "C"])
    b = pd.Series(["A", "B", "B", "B"])
    r = compare_categorical(a, b, "familia")
    assert r["tv_distance"] > 0
    assert "top_shifts" in r


if __name__ == "__main__":
    test_align_and_scale_detection()
    test_similar_shape_after_zscore()
    test_psi_identical()
    test_melt_and_sql_shape()
    test_categorical()
    print("ok")
