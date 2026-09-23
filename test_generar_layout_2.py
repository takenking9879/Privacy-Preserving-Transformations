# -*- coding: utf-8 -*-
"""Pruebas de las reglas del sugeridor (sin Hive de prod)."""
from datetime import date
from generar_layout_2 import fecha_hoy_num, mes_nbco_por_regla, mes_de_semana


def test_fecha_hoy_num():
    assert fecha_hoy_num(date(2026, 9, 2)) == 20260902
    assert fecha_hoy_num(date(2026, 9, 23)) == 20260923


def test_nbco_dia_5():
    # Abril completo aparece el 5 de mayo
    assert mes_nbco_por_regla(date(2026, 5, 5)) == 202604
    assert mes_nbco_por_regla(date(2026, 5, 4)) == 202603
    assert mes_nbco_por_regla(date(2026, 1, 5)) == 202512
    assert mes_nbco_por_regla(date(2026, 1, 4)) == 202511
    assert mes_nbco_por_regla(date(2026, 9, 23)) == 202608


def test_mes_de_semana():
    assert mes_de_semana(202630) // 100 == 2026


def test_sugerir_con_tablas_falsas():
    import os
    import shutil
    import tempfile

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    from generar_layout_2 import fuentes, sugerir_parametros

    warehouse = tempfile.mkdtemp(prefix="sug-wh-")
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test_sugerir")
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        src = fuentes()
        for db in (
            "cd_baz_bdclientes",
            "ws_celcobd_analitica",
            "ws_ektcomd_analitica",
            "ma_bdbaz",
            "ec_baz_bdclientes",
        ):
            spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

        # Calendario: domingos ene-jul 2026 (semanas 202601..202630)
        fechas = []
        for sem, fec, mes in (
            (202622, 20260531, 202605),
            (202626, 20260628, 202606),
            (202627, 20260705, 202607),
            (202630, 20260726, 202607),
        ):
            fechas.append((sem, fec, mes, "7"))
        spark.createDataFrame(
            fechas, "num_periodo_sem INT, fec_num INT, num_periodo_mes INT, num_dia_sem STRING"
        ).write.mode("overwrite").saveAsTable(src["fechas"])

        spark.createDataFrame(
            [(202630, "1-2-0-1")], "fecha_salida INT, cliente_unico STRING"
        ).write.mode("overwrite").saveAsTable(src["pivote"])

        spark.createDataFrame(
            [(202626,)], "num_periodo_sem INT"
        ).write.mode("overwrite").saveAsTable(src["cerebro"])

        spark.createDataFrame(
            [("c", 202627)], "cliente_unico STRING, num_periodo_sem INT"
        ).write.mode("overwrite").saveAsTable(src["lae"])

        spark.createDataFrame(
            [("m", 202605)], "id_master STRING, num_periodo_mes INT"
        ).write.mode("overwrite").saveAsTable(src["nbco"])

        # CLTV existe 202622 y 202626 (el bug: calendario tiene 202626)
        spark.createDataFrame([("M1", 1.0)], "id_master STRING, cltv DOUBLE").write.mode(
            "overwrite"
        ).saveAsTable("ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_202622_v2")
        spark.createDataFrame([("M1", 1.0)], "id_master STRING, cltv DOUBLE").write.mode(
            "overwrite"
        ).saveAsTable("ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_202626_v2")

        sug = sugerir_parametros(spark, fecha_hoy=date(2026, 9, 23), imprimir=True)
        assert sug.semana_cmp == 202630, sug.semana_cmp
        assert sug.semana_cltv == 202626, sug.semana_cltv
        assert sug.semana_lae == 202627, sug.semana_lae
        # 23-sep → regla NBCO agosto, pero tabla solo tiene mayo y mes insumo es julio
        assert sug.mes_nbco == 202605, sug.mes_nbco
        print("OK sugerir_parametros con tablas falsas", sug.llamada)
    finally:
        spark.stop()
        shutil.rmtree(warehouse, ignore_errors=True)


if __name__ == "__main__":
    test_fecha_hoy_num()
    test_nbco_dia_5()
    test_mes_de_semana()
    print("OK reglas sugeridor")
    test_sugerir_con_tablas_falsas()
