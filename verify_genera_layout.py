# -*- coding: utf-8 -*-
"""
Compara la versión BASE (lógica original) vs genera_layout.py optimizada
usando tablas sintéticas en un Spark local.

Uso:
    python3 verify_genera_layout.py
    python3 verify_genera_layout.py --n 20000
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from genera_layout import genera_layout


SEMANA = 202401
SEMANA_CMP = 202401
MES = 202401
SEMANA_LAE = 202401
FECHA_NUMS = [20240101, 20240102, 20240103, 20240104, 20240105, 20240106, 20240107]


def crear_spark(warehouse: str) -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("verify_genera_layout")
        .config("spark.sql.warehouse.dir", warehouse)
        .config("spark.driver.memory", "3g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _write(df, name: str):
    df.write.mode("overwrite").saveAsTable(name)


def crear_catalogo(spark: SparkSession):
    for db in (
        "cd_baz_bdclientes",
        "ws_celcobd_analitica",
        "ec_baz_bdclientes",
        "ws_ektcomd_analitica",
        "ws_aarent_analitica",
        "ma_bdbaz",
    ):
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")


def _cu(i: int) -> str:
    return f"1-2-{i % 50}-{i:08d}"


def _master(i: int) -> str:
    return f"M{i:08d}"


def _icu(i: int, k: int = 0) -> str:
    return f"ICU{i:08d}{k}"


def _ruido_cuarteles(spark: SparkSession, n_ruido: int):
    return (
        spark.range(n_ruido)
        .select(
            F.concat(F.lit("9-9-9-"), F.lpad(F.col("id").cast("string"), 8, "0")).alias("id_cliente"),
            F.concat(F.lit("QN"), (F.col("id") % 9).cast("string")).alias("cuartel"),
            (F.lit(20220101) + (F.col("id") % 365).cast("int")).alias("fec_surtimiento"),
        )
    )


def _ruido_digital(spark: SparkSession, n_ruido: int):
    return (
        spark.range(n_ruido)
        .select(
            F.concat(F.lit("9-9-9-"), F.lpad(F.col("id").cast("string"), 8, "0")).alias("id_cliente_unico"),
            F.concat(F.lit("ICUNOISE"), F.lpad(F.col("id").cast("string"), 8, "0")).alias("id_icu"),
            F.lit("2023-06-01 00:00:00").alias("tms_alta"),
        )
    )


def _ruido_txn(spark: SparkSession, n_ruido: int):
    return (
        spark.range(n_ruido)
        .select(
            F.concat(F.lit("ICUNOISE"), F.lpad(F.col("id").cast("string"), 8, "0")).alias("id_icu"),
            F.lit("2024-01-03 12:00:00").alias("tms_operacion"),
        )
    )


def poblar_tablas(spark: SparkSession, n: int, incluir_duplicados: bool = False, ruido: int = 0):
    """
    Construye un universo determinista.

    De los n clientes del pivote:
      - 10% sin cerebro  -> se caen en el inner join por id_master
      - de los restantes, 10% sin nbco -> también se caen
      - el resto sobrevive y debe coincidir en base vs optimizada
    """
    crear_catalogo(spark)
    fecha_corte = datetime(2024, 1, 7)
    fecha_str_corte = fecha_corte.strftime("%Y-%m-%d %H:%M:%S")

    fechas = spark.createDataFrame(
        [(SEMANA, fec) for fec in FECHA_NUMS],
        "num_periodo_sem INT, fec_num INT",
    )
    _write(fechas, "cd_baz_bdclientes.cd_gen_fechas_cat")

    pivote_rows = [(_cu(i), SEMANA_CMP) for i in range(n)]
    _write(
        spark.createDataFrame(pivote_rows, "cliente_unico STRING, fecha_salida INT"),
        "ws_celcobd_analitica.ta_338082_servilleta_total_cu_240826",
    )

    cerebro_rows = []
    for i in range(n):
        if i % 10 == 0:
            continue
        cerebro_rows.append(
            (
                _master(i),
                _cu(i),
                i % 2,
                i % 20,
                float((i % 7) * 100),
                f"T{i % 3}",
                i % 2,
                i % 120,
                SEMANA,
            )
        )
    if incluir_duplicados and cerebro_rows:
        cerebro_rows.append(cerebro_rows[0])

    _write(
        spark.createDataFrame(
            cerebro_rows,
            "id_master STRING, id_cte_unico STRING, ind_inac_24meses INT, "
            "semanas_inactivo INT, est_ingresos_indirectos DOUBLE, tipo_sol_cte STRING, "
            "marca_cliente_bueno INT, antig_tl INT, num_periodo_sem INT",
        ),
        "ec_baz_bdclientes.ec_cre_comportamental_layout_cerebro_full",
    )

    lae_rows = []
    for i in range(n):
        if i % 4 == 0:
            continue
        lae_rows.append(
            (_cu(i), 0.01 * (i % 9), 0.02 * (i % 8), 0.03 * (i % 7),
             0.04 * (i % 6), 0.05 * (i % 5), f"S{i % 4}", SEMANA_LAE)
        )
    _write(
        spark.createDataFrame(
            lae_rows,
            "cliente_unico STRING, PD3 DOUBLE, pd5 DOUBLE, pd7 DOUBLE, "
            "pd10 DOUBLE, pd20 DOUBLE, segmento STRING, num_periodo_sem INT",
        ),
        "ws_celcobd_analitica.lae_interno_mensual",
    )

    hog, con, efe = [], [], []
    for i in range(n):
        if i % 10 == 0:
            continue
        if i % 3 != 0:
            hog.append((_master(i), float(100 + i)))
        if i % 5 != 0:
            con.append((_master(i), float(200 + i)))
        if i % 7 != 0:
            efe.append((_master(i), float(300 + i)))

    _write(spark.createDataFrame(hog, "id_master STRING, cltv DOUBLE"),
           f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_{SEMANA}_v2")
    _write(spark.createDataFrame(con, "id_master STRING, cltv DOUBLE"),
           f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{SEMANA}_v2")
    _write(spark.createDataFrame(efe, "id_master STRING, cltv DOUBLE"),
           f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_{SEMANA}_v2")

    activo = [(_master(i), float(50 + i % 17)) for i in range(n) if i % 10 != 0 and i % 6 != 0]
    _write(
        spark.createDataFrame(activo, "id_master STRING, clvpa DOUBLE"),
        f"ws_celcobd_analitica.`1034848_cltvpa_final_{SEMANA}`",
    )

    rbs = []
    semanas_rbs = [202302, 202350, 202401, 202201]
    for i in range(n):
        if i % 10 == 0:
            continue
        for w in semanas_rbs:
            rbs.append((
                _master(i), w,
                float(i % 11), float(i % 13), float(i % 17),
                float(i % 19), float(i % 23), float(i % 3),
                float(i % 5), float(i % 7), float(i % 29),
                int(i % 40),
            ))
    df_rbs = spark.createDataFrame(
        rbs,
        "id_master STRING, num_periodo_sem INT, rentabilidad_credito DOUBLE, "
        "interes_pagado DOUBLE, intereses_cobrados DOUBLE, reserva DOUBLE, "
        "margen_cliente DOUBLE, costo_gestion_calle DOUBLE, "
        "costo_gestion_llamada DOUBLE, costo_gestion_sms DOUBLE, "
        "papm DOUBLE, atraso INT",
    )
    if ruido > 0:
        df_rbs = df_rbs.unionByName(
            spark.range(ruido).select(
                F.concat(F.lit("MNOISE"), F.lpad(F.col("id").cast("string"), 8, "0")).alias("id_master"),
                F.lit(202350).alias("num_periodo_sem"),
                F.lit(1.0).alias("rentabilidad_credito"),
                F.lit(1.0).alias("interes_pagado"),
                F.lit(1.0).alias("intereses_cobrados"),
                F.lit(1.0).alias("reserva"),
                F.lit(1.0).alias("margen_cliente"),
                F.lit(1.0).alias("costo_gestion_calle"),
                F.lit(1.0).alias("costo_gestion_llamada"),
                F.lit(1.0).alias("costo_gestion_sms"),
                F.lit(1.0).alias("papm"),
                F.lit(1).alias("atraso"),
            )
        )
    _write(df_rbs, "ws_aarent_analitica.cu_renta_credito_operaciones_cliente")

    nbco = []
    for i in range(n):
        if i % 10 == 0:
            continue
        if (i // 10) % 10 == 0:
            continue
        nbco.append((
            _master(i),
            float((i % 5) / 4.0),
            "user_x",
            "2024-01-01",
            MES,
            f"seg_{i % 3}",
            i % 2,
        ))
    _write(
        spark.createDataFrame(
            nbco,
            "id_master STRING, p1_conectividad DOUBLE, fcusuariocreacion STRING, "
            "fdfechacreacion STRING, num_periodo_mes INT, nbco_seg STRING, flag_nbco INT",
        ),
        "ma_bdbaz.ml_cre_recomendacion_nbco_existentes",
    )

    digital = []
    txn = []
    for i in range(n):
        if i % 3 != 0:
            continue
        alta_ok = fecha_corte - timedelta(days=10 + (i % 40))
        digital.append((_cu(i), _icu(i, 0), alta_ok.strftime("%Y-%m-%d %H:%M:%S")))
        if i % 9 == 0:
            digital.append((_cu(i), _icu(i, 1), alta_ok.strftime("%Y-%m-%d %H:%M:%S")))
        if i % 11 == 0:
            alta_tarde = fecha_corte + timedelta(days=3)
            digital.append((_cu(i), _icu(i, 2), alta_tarde.strftime("%Y-%m-%d %H:%M:%S")))

        if i % 2 == 0:
            op = fecha_corte - timedelta(days=i % 20)
            txn.append((_icu(i, 0), op.strftime("%Y-%m-%d %H:%M:%S")))
        if i % 9 == 0:
            op_old = fecha_corte - timedelta(days=40)
            txn.append((_icu(i, 1), op_old.strftime("%Y-%m-%d %H:%M:%S")))
        txn.append((_icu(i, 99), fecha_str_corte))

    hist_por_cu = 12 if n >= 500 else 4
    cuarteles = []
    for i in range(n):
        if i % 5 == 0:
            continue
        for k in range(hist_por_cu):
            fec = 20230101 + k * 20
            if fec > 20240107:
                fec = 20231215
            cuarteles.append((_cu(i), f"Q{k % 6}", fec))
        cuarteles.append((_cu(i), "QMAX", 20240105))
        cuarteles.append((_cu(i), "QAFTER", 20240120))
        cuarteles.append((_cu(i), "QOLD", 19010101))

    df_digital = spark.createDataFrame(
        digital, "id_cliente_unico STRING, id_icu STRING, tms_alta STRING"
    )
    df_txn = spark.createDataFrame(txn, "id_icu STRING, tms_operacion STRING")
    df_cuarteles = spark.createDataFrame(
        cuarteles, "id_cliente STRING, cuartel STRING, fec_surtimiento INT"
    )

    if ruido > 0:
        df_digital = df_digital.unionByName(_ruido_digital(spark, ruido))
        df_txn = df_txn.unionByName(_ruido_txn(spark, ruido))
        df_cuarteles = df_cuarteles.unionByName(_ruido_cuarteles(spark, ruido))

    _write(df_digital, "cd_baz_bdclientes.cd_dig_clientes")
    _write(df_txn, "cd_baz_bdclientes.cd_dig_txn_financieras")
    _write(df_cuarteles, "ws_celcobd_analitica.tt_1117735_pedidoshistoricos_cuartel")


def genera_layout_base(semana_cmp, semana_cltv, mes, semana_lae, modo, spark, escribir=False, tabla_out=None):
    """Reproducción fiel de la función original (sin recortes ni AQE extra)."""
    semana = semana_cltv

    fechas = spark.sql(
        f"SELECT * FROM cd_baz_bdclientes.cd_gen_fechas_cat WHERE num_periodo_sem={semana} "
    )
    dias_semana_num = [row.fec_num for row in fechas.select("fec_num").collect()]
    fecha_num = sorted(dias_semana_num)[-1]
    fecha_str = f"{str(fecha_num)[:4]}-{str(fecha_num)[4:6]}-{str(fecha_num)[6:]}"
    semana_ini = int((int(semana / 100) - 1) * 100 + semana % 100)

    base_pivote = spark.sql(
        f"""
        SELECT cliente_unico, fecha_salida
        FROM ws_celcobd_analitica.ta_338082_servilleta_total_cu_240826
        WHERE fecha_salida={semana_cmp}
        """
    )

    cerebro = spark.sql(
        f"""
        SELECT id_master, id_cte_unico AS cliente_unico,
            ind_inac_24meses,
            semanas_inactivo,
            est_ingresos_indirectos,
            tipo_sol_cte,
            marca_cliente_bueno,
            antig_tl,
            num_periodo_sem
        FROM ec_baz_bdclientes.ec_cre_comportamental_layout_cerebro_full
        WHERE num_periodo_sem={semana}
        """
    )
    ventana = Window.partitionBy("cliente_unico").orderBy(F.rand())
    cerebro = cerebro.withColumn("numero_fila", F.row_number().over(ventana))
    cerebro = cerebro.filter(F.col("numero_fila") == 1).drop("numero_fila")

    lae = spark.sql(
        f"""
        SELECT cliente_unico, PD3, pd5, pd7, pd10, pd20, segmento
        FROM ws_celcobd_analitica.lae_interno_mensual
        WHERE num_periodo_sem={semana_lae}
        """
    )

    cltv_futuro_hog = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_hog FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_{semana}_v2"
    )
    cltv_futuro_con = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_con FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2"
    )
    cltv_futuro_efe = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_efe FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_{semana}_v2"
    )
    cltv_futuro_mov = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_mov FROM ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2"
    )

    cltv_real_rbs = spark.sql(
        f"""
        SELECT
            id_master,
            SUM(rentabilidad_credito) AS rentabilidad_credito,
            SUM(interes_pagado) AS interes_pagado,
            SUM(intereses_cobrados) AS intereses_cobrados,
            SUM(reserva) AS reservas,
            SUM(margen_cliente) AS margen_cliente,
            SUM(costo_gestion_calle) AS costo_gestion_calle,
            SUM(costo_gestion_llamada) AS costo_gestion_llamada,
            SUM(costo_gestion_sms) AS costo_gestion_sms,
            SUM(papm) AS papm,
            MAX(atraso) AS max_atraso
        FROM ws_aarent_analitica.cu_renta_credito_operaciones_cliente
        WHERE num_periodo_sem > {semana_ini} AND num_periodo_sem <= {semana}
        GROUP BY id_master
        """
    )

    cltv_activo = spark.sql(
        f"SELECT id_master, clvpa AS cltv_activo FROM ws_celcobd_analitica.`1034848_cltvpa_final_{semana}`"
    )

    nbco = spark.sql(
        f"SELECT * FROM ma_bdbaz.ml_cre_recomendacion_nbco_existentes WHERE num_periodo_mes={mes}"
    )
    nbco = nbco.drop("fcusuariocreacion", "fdfechacreacion", "num_periodo_mes")

    digital = spark.sql(
        f"""
        SELECT id_cliente_unico AS cliente_unico, id_icu, tms_alta, 1 AS ind_digital
        FROM cd_baz_bdclientes.cd_dig_clientes
        WHERE tms_alta <= '{fecha_str}'
        """
    )

    uso = spark.sql(
        f"""
        SELECT DISTINCT id_icu, 1 AS ind_digital_uso
        FROM cd_baz_bdclientes.cd_dig_txn_financieras
        WHERE tms_operacion > DATE_ADD('{fecha_str}', -30) AND tms_operacion <= '{fecha_str}'
        """
    )

    digital_uso = digital.join(uso, on="id_icu", how="left")
    digital_uso_agg = digital_uso.groupby("cliente_unico").agg(
        F.max(F.col("tms_alta")).alias("tms_alta"),
        F.max(F.col("ind_digital")).alias("ind_digital"),
        F.max(F.col("ind_digital_uso")).alias("ind_digital_uso"),
    )

    aux_cuarteles = spark.sql(
        f"""
        SELECT id_cliente AS cliente_unico, MAX(fec_surtimiento) AS fec_surtimiento
        FROM ws_celcobd_analitica.tt_1117735_pedidoshistoricos_cuartel
        WHERE fec_surtimiento > 19010101 AND fec_surtimiento <= {fecha_num}
        GROUP BY id_cliente
        """
    )

    cuarteles_pedido = spark.sql(
        """
        SELECT id_cliente AS cliente_unico, cuartel, fec_surtimiento
        FROM ws_celcobd_analitica.tt_1117735_pedidoshistoricos_cuartel
        """
    )

    cuarteles_cliente = cuarteles_pedido.join(
        aux_cuarteles, on=["cliente_unico", "fec_surtimiento"], how="inner"
    )
    cuarteles_cliente = cuarteles_cliente.select("cliente_unico", "cuartel").distinct()
    cuarteles_cliente = cuarteles_cliente.withColumn("numero_fila", F.row_number().over(ventana))
    cuarteles_cliente = cuarteles_cliente.filter(F.col("numero_fila") == 1).drop("numero_fila")

    base_pivote_s0 = base_pivote.join(cerebro, on=["cliente_unico"], how="left")
    base_pivote_s1 = base_pivote_s0.join(lae, on=["cliente_unico"], how="left")
    base_pivote_s2 = base_pivote_s1.join(digital_uso_agg, on="cliente_unico", how="left")
    base_pivote_s3 = base_pivote_s2.join(cuarteles_cliente, on="cliente_unico", how="left")

    base_pivote_master = nbco.join(cltv_futuro_hog, on=["id_master"], how="left")
    base_pivote_master_s1 = base_pivote_master.join(cltv_futuro_con, on=["id_master"], how="left")
    base_pivote_master_s2 = base_pivote_master_s1.join(cltv_futuro_mov, on=["id_master"], how="left")
    base_pivote_master_s3 = base_pivote_master_s2.join(cltv_futuro_efe, on=["id_master"], how="left")
    base_pivote_master_s4 = base_pivote_master_s3.join(cltv_real_rbs, on=["id_master"], how="left")
    base_pivote_master_s5 = base_pivote_master_s4.join(cltv_activo, on="id_master", how="left")

    base_pivote_final = base_pivote_s3.join(base_pivote_master_s5, on=["id_master"], how="inner")

    base_pivote_final = base_pivote_final.withColumn("id_pais", F.split(F.col("cliente_unico"), "-", -1)[0])
    base_pivote_final = base_pivote_final.withColumn("id_canal", F.split(F.col("cliente_unico"), "-", -1)[1])
    base_pivote_final = base_pivote_final.withColumn("id_sucursal", F.split(F.col("cliente_unico"), "-", -1)[2])
    base_pivote_final = base_pivote_final.withColumn("id_num_folio", F.split(F.col("cliente_unico"), "-", -1)[3])

    base_pivote_final = base_pivote_final.withColumn("cltv_f_con", F.col("cltv_f_con") * F.col("p1_conectividad"))
    base_pivote_final = base_pivote_final.withColumn("cltv_f_hog", F.col("cltv_f_hog") * (F.lit(1) - F.col("p1_conectividad")))
    base_pivote_final = base_pivote_final.drop("cliente_unico")

    if escribir:
        (
            base_pivote_final.write.format("parquet")
            .mode(modo)
            .partitionBy("num_periodo_sem")
            .saveAsTable(tabla_out)
        )
    return base_pivote_final


def _as_comparable(df):
    cols = sorted(df.columns)
    return df.select(*cols)


def comparar(df_base, df_opt):
    b = _as_comparable(df_base)
    o = _as_comparable(df_opt)

    if b.columns != o.columns:
        raise AssertionError(f"Columnas distintas\nBASE={b.columns}\nOPT={o.columns}")

    n_b, n_o = b.count(), o.count()
    if n_b != n_o:
        raise AssertionError(f"Conteo distinto: base={n_b} opt={n_o}")

    solo_base = b.subtract(o).count()
    solo_opt = o.subtract(b).count()
    if solo_base or solo_opt:
        print("Muestra solo-base:")
        b.subtract(o).show(10, truncate=False)
        print("Muestra solo-opt:")
        o.subtract(b).show(10, truncate=False)
        raise AssertionError(
            f"Filas distintas: solo_base={solo_base} solo_opt={solo_opt}"
        )
    return n_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="clientes sintéticos para equivalencia")
    parser.add_argument("--n-perf", type=int, default=4000, help="clientes sintéticos para timing")
    parser.add_argument("--ruido", type=int, default=1_500_000, help="filas extra fuera del pivote")
    args = parser.parse_args()

    warehouse = tempfile.mkdtemp(prefix="spark-wh-")
    spark = None
    try:
        spark = crear_spark(warehouse)

        print("=" * 72)
        print(f"EQUIVALENCIA  n={args.n} (sin duplicados ambiguos de rand())")
        print("=" * 72)
        poblar_tablas(spark, args.n, incluir_duplicados=False, ruido=0)

        t0 = time.time()
        df_base = genera_layout_base(SEMANA_CMP, SEMANA, MES, SEMANA_LAE, "overwrite", spark)
        df_base.persist()
        n_base = df_base.count()
        t_base = time.time() - t0

        t1 = time.time()
        df_opt = genera_layout(
            SEMANA_CMP, SEMANA, MES, SEMANA_LAE, "overwrite",
            spark=spark, escribir=False, refrescar=False,
        )
        n_opt = df_opt.count()
        t_opt = time.time() - t1

        n = comparar(df_base, df_opt)
        print(f"OK equivalencia: {n} filas idénticas")
        print(f"tiempo base={t_base:.2f}s  opt={t_opt:.2f}s  (n={args.n}, filas_out={n_base})")

        print("=" * 72)
        print(f"PERFORMANCE   n={args.n_perf}  ruido={args.ruido}")
        print("=" * 72)
        spark.catalog.clearCache()
        poblar_tablas(spark, args.n_perf, incluir_duplicados=False, ruido=args.ruido)

        t0 = time.time()
        df_base_p = genera_layout_base(SEMANA_CMP, SEMANA, MES, SEMANA_LAE, "overwrite", spark)
        n_base_p = df_base_p.count()
        t_base_p = time.time() - t0

        spark.catalog.clearCache()
        t1 = time.time()
        df_opt_p = genera_layout(
            SEMANA_CMP, SEMANA, MES, SEMANA_LAE, "overwrite",
            spark=spark, escribir=False, refrescar=False,
        )
        n_opt_p = df_opt_p.count()
        t_opt_p = time.time() - t1

        comparar(df_base_p, df_opt_p)
        speedup = t_base_p / t_opt_p if t_opt_p else float("inf")
        print(f"OK perf equivalencia: {n_opt_p} filas")
        print(
            f"PERF base={t_base_p:.2f}s  opt={t_opt_p:.2f}s  "
            f"speedup={speedup:.2f}x  filas_out={n_base_p}"
        )
        print("ALL CHECKS PASSED")
    finally:
        if spark is not None:
            spark.stop()
        shutil.rmtree(warehouse, ignore_errors=True)


if __name__ == "__main__":
    main()
