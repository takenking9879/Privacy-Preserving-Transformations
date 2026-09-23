# -*- coding: utf-8 -*-
"""
genera_layout — versión optimizada (CPU / RAM / shuffle).

Copia este archivo a tu notebook o job. La firma pública se mantiene:

    genera_layout(semana_cmp, semana_cltv, mes, semana_lae, modo)

Ajustes que debes revisar antes de correr:
  1) TABLA_OUT  — tabla Hive/Parquet de destino.
  2) Las variables TBL_* al inicio de la función (nombres FROM).
  3) TBL_CLTV_FUTURO_MOV apunta a la tabla *_con_* igual que el código original.
     Si era un typo y existe *_mov_*, cámbiala.

Equivalencia vs. la versión base:
  - Mismos filtros, agregaciones, joins (left/inner) y transformaciones finales.
  - El original desempataba con rand(): SI hay 2+ filas candidatas del mismo
    cliente, NI SIQUIERA la base da lo mismo entre dos corridas.
  - Aquí el desempate es estable (id_master / cuartel). Con 1 candidata
    (el caso típico) el resultado es idéntico al original. Con empate, esta
    versión siempre elige la misma fila; la base no.
"""

import sys
from datetime import datetime

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _paso(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Sesión (opcional). Si ya tienes `spark` creado afuera, no hace falta llamarla.
# ---------------------------------------------------------------------------
def crear_spark():
    """YARN/Hive: dynamic allocation ON. No uses executor.instances —
    si apagas dyn alloc el Stage 0 se queda en (0+1)/1 esperando containers.
    spark.stop() antes: getOrCreate() no aplica un driver nuevo.
    """
    return (
        SparkSession.builder
        .appName("create_nbco_base")
        .enableHiveSupport()
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.maxExecutors", "3")
        .config("spark.dynamicAllocation.initialExecutors", "2")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        .config("spark.executor.cores", "3")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "24")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def _sesion_spark(spark=None):
    if spark is not None:
        session = spark
    else:
        session = SparkSession.getActiveSession()
        if session is None:
            try:
                session = (
                    SparkSession.builder
                    .appName("create_nbco_base")
                    .enableHiveSupport()
                    .getOrCreate()
                )
            except Exception:
                session = SparkSession.builder.appName("create_nbco_base").getOrCreate()

    # AQE. Shuffle 24 = ~4 tasks por core con 2 exec × 3 cores.
    confs = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.localShuffleReader.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64m",
        "spark.sql.adaptive.coalescePartitions.minPartitionNum": "8",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum": "24",
        "spark.sql.autoBroadcastJoinThreshold": str(16 * 1024 * 1024),
        "spark.sql.adaptive.autoBroadcastJoinThreshold": str(16 * 1024 * 1024),
        "spark.sql.broadcastTimeout": "600",
        "spark.sql.files.maxPartitionBytes": "128m",
        "spark.sql.inMemoryColumnarStorage.compressed": "true",
        "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
    }
    for key, value in confs.items():
        session.conf.set(key, value)

    try:
        actuales = int(session.conf.get("spark.sql.shuffle.partitions"))
    except Exception:
        actuales = 200
    if actuales >= 100:
        session.conf.set("spark.sql.shuffle.partitions", "24")
    return session


def _dedup_estable(df, llave, *orden):
    """1 fila por llave con orden determinista (reproducible entre corridas)."""
    ventana = Window.partitionBy(llave).orderBy(*orden)
    return (
        df.withColumn("numero_fila", F.row_number().over(ventana))
        .where(F.col("numero_fila") == 1)
        .drop("numero_fila")
    )


def genera_layout(
    semana_cmp,
    semana_cltv,
    mes,
    semana_lae,
    modo,
    tabla_out=None,
    spark=None,
    escribir=True,
    refrescar=False,
):
    # ======================================================================
    # VARIABLES DE TABLAS (FROM) — edita aquí
    # ======================================================================
    semana = semana_cltv

    TBL_FECHAS = "cd_baz_bdclientes.cd_gen_fechas_cat"
    TBL_PIVOTE = "ws_celcobd_analitica.ta_338082_servilleta_total_cu_240826"
    TBL_CEREBRO = "ec_baz_bdclientes.ec_cre_comportamental_layout_cerebro_full"
    TBL_LAE = "ws_celcobd_analitica.lae_interno_mensual"
    TBL_CLTV_FUTURO_HOG = f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_{semana}_v2"
    TBL_CLTV_FUTURO_CON = f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2"
    TBL_CLTV_FUTURO_EFE = f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_{semana}_v2"
    # El original lee MOV desde la tabla CON. Cambia a _mov_ si corresponde.
    TBL_CLTV_FUTURO_MOV = f"ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2"
    TBL_CLTV_RENTA = "ws_aarent_analitica.cu_renta_credito_operaciones_cliente"
    TBL_CLTV_ACTIVO = f"ws_celcobd_analitica.`1034848_cltvpa_final_{semana}`"
    TBL_NBCO = "ma_bdbaz.ml_cre_recomendacion_nbco_existentes"
    TBL_DIGITAL = "cd_baz_bdclientes.cd_dig_clientes"
    TBL_TXN = "cd_baz_bdclientes.cd_dig_txn_financieras"
    TBL_CUARTELES = "ws_celcobd_analitica.tt_1117735_pedidoshistoricos_cuartel"
    TABLA_OUT = "ws_celcobd_analitica.nbco_base_layout"

    if tabla_out is None:
        tabla_out = TABLA_OUT

    spark = _sesion_spark(spark)
    try:
        _paso(
            f"Spark listo  app={spark.sparkContext.applicationId}  "
            f"ui={getattr(spark.sparkContext, 'uiWebUrl', None)}"
        )
    except Exception as exc:
        _paso(f"Spark listo (no pude leer applicationId: {exc})")

    spark.sparkContext.setJobDescription("00_heartbeat_select1")
    spark.sql("SELECT 1 AS ok").collect()
    _paso("Heartbeat SELECT 1 OK — si no viste 1 stage, estás en otra Spark UI")

    if refrescar:
        _paso(
            f"REFRESH {TBL_DIGITAL} y {TBL_TXN} — esto NO crea stages; "
            "si se queda aquí, mata la celda y llama con refrescar=False"
        )
        spark.sql(f"REFRESH TABLE {TBL_DIGITAL}")
        _paso(f"REFRESH OK {TBL_DIGITAL}")
        spark.sql(f"REFRESH TABLE {TBL_TXN}")
        _paso(f"REFRESH OK {TBL_TXN}")
    else:
        _paso("REFRESH omitido (refrescar=False)")

    # ----------------------------------------------------------------------
    # Fechas (tabla chica: collect de ~7 filas, igual que la base)
    # ----------------------------------------------------------------------
    spark.sparkContext.setJobDescription("01_fechas")
    _paso(f"Leyendo fechas semana={semana} de {TBL_FECHAS}")
    fechas = spark.sql(
        f"SELECT fec_num FROM {TBL_FECHAS} WHERE num_periodo_sem={semana}"
    )
    dias_semana_num = [row.fec_num for row in fechas.collect()]
    if not dias_semana_num:
        raise ValueError(f"Sin fechas para num_periodo_sem={semana} en {TBL_FECHAS}")

    fecha_num = sorted(dias_semana_num)[-1]
    fecha_str = f"{str(fecha_num)[:4]}-{str(fecha_num)[4:6]}-{str(fecha_num)[6:]}"
    semana_ini = int((int(semana / 100) - 1) * 100 + semana % 100)

    _paso(f"fechas OK  semana_ini={semana_ini}  fecha_num={fecha_num}  dias={dias_semana_num}")
    print(semana_ini)
    print(fecha_num, fecha_str)
    print(dias_semana_num)

    # ----------------------------------------------------------------------
    # Universo de clientes (se reutiliza para recortar TODAS las fact tables)
    # ----------------------------------------------------------------------
    N_PARTS = 24
    # El pivote llega en 1 archivo → 1 partición. Si AQE hace broadcast de
    # cerebro/lae/digital encima, TODO el COUNT corre en 1 task (~37 min).
    base_pivote = spark.sql(
        f"""
        SELECT cliente_unico, fecha_salida
        FROM {TBL_PIVOTE}
        WHERE fecha_salida={semana_cmp}
        """
    ).repartition(N_PARTS, "cliente_unico")

    pivot_keys = base_pivote.select("cliente_unico").distinct()
    _paso(f"Pivote reparticionado a {N_PARTS} (lazy)")

    # ----------------------------------------------------------------------
    # Cerebro: recorte por semana + CUs del pivote, dedup estable.
    # Tampoco se materializa aquí: el count() de masters leía cerebro_full
    # + window + broadcast y trababa el driver.
    # ----------------------------------------------------------------------
    cerebro = (
        spark.sql(
            f"""
            SELECT
                id_master,
                id_cte_unico AS cliente_unico,
                ind_inac_24meses,
                semanas_inactivo,
                est_ingresos_indirectos,
                tipo_sol_cte,
                marca_cliente_bueno,
                antig_tl,
                num_periodo_sem
            FROM {TBL_CEREBRO}
            WHERE num_periodo_sem={semana}
            """
        )
        .join(pivot_keys, on="cliente_unico", how="inner")
    )
    cerebro = _dedup_estable(
        cerebro, "cliente_unico",
        F.col("id_master").asc_nulls_last(),
        F.col("antig_tl").asc_nulls_last(),
    )

    master_keys = (
        cerebro.select("id_master")
        .where(F.col("id_master").isNotNull())
        .distinct()
    )
    _paso("Cerebro + master_keys definidos (lazy). Stages salen en el COUNT final.")

    # ----------------------------------------------------------------------
    # LAE recortado al pivote
    # ----------------------------------------------------------------------
    lae = (
        spark.sql(
            f"""
            SELECT cliente_unico, PD3, pd5, pd7, pd10, pd20, segmento
            FROM {TBL_LAE}
            WHERE num_periodo_sem={semana_lae}
            """
        )
        .join(pivot_keys, on="cliente_unico", how="inner")
    )

    # CLTV semanales: ya vienen filtrados por {semana}; no vale la pena
    # un left_semi extra. Solo se recorta la renta (histórico de un año).
    cltv_futuro_hog = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_hog FROM {TBL_CLTV_FUTURO_HOG}"
    )
    cltv_futuro_con = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_con FROM {TBL_CLTV_FUTURO_CON}"
    )
    cltv_futuro_efe = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_efe FROM {TBL_CLTV_FUTURO_EFE}"
    )
    cltv_futuro_mov = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_mov FROM {TBL_CLTV_FUTURO_MOV}"
    )
    cltv_activo = spark.sql(
        f"SELECT id_master, clvpa AS cltv_activo FROM {TBL_CLTV_ACTIVO}"
    )

    cltv_real_rbs = (
        spark.table(TBL_CLTV_RENTA)
        .where(
            (F.col("num_periodo_sem") > semana_ini)
            & (F.col("num_periodo_sem") <= semana)
        )
        .join(master_keys, on="id_master", how="left_semi")
        .groupBy("id_master")
        .agg(
            F.sum("rentabilidad_credito").alias("rentabilidad_credito"),
            F.sum("interes_pagado").alias("interes_pagado"),
            F.sum("intereses_cobrados").alias("intereses_cobrados"),
            F.sum("reserva").alias("reservas"),
            F.sum("margen_cliente").alias("margen_cliente"),
            F.sum("costo_gestion_calle").alias("costo_gestion_calle"),
            F.sum("costo_gestion_llamada").alias("costo_gestion_llamada"),
            F.sum("costo_gestion_sms").alias("costo_gestion_sms"),
            F.sum("papm").alias("papm"),
            F.max("atraso").alias("max_atraso"),
        )
    )

    nbco = (
        spark.table(TBL_NBCO)
        .where(F.col("num_periodo_mes") == mes)
        .drop("fcusuariocreacion", "fdfechacreacion", "num_periodo_mes")
        .repartition(N_PARTS, "id_master")
    )

    # ----------------------------------------------------------------------
    # Digital + uso (30 días): solo CUs del pivote e ICUs de esos clientes
    # ----------------------------------------------------------------------
    digital = spark.sql(
        f"""
        SELECT id_cliente_unico AS cliente_unico, id_icu, tms_alta, 1 AS ind_digital
        FROM {TBL_DIGITAL}
        WHERE tms_alta <= '{fecha_str}'
        """
    ).join(pivot_keys, on="cliente_unico", how="inner")

    icu_keys = digital.select("id_icu").distinct()
    # Primero recorta por ICU del pivote; el DISTINCT global de 30 días es carísimo.
    uso = (
        spark.sql(
            f"""
            SELECT id_icu
            FROM {TBL_TXN}
            WHERE tms_operacion > DATE_ADD('{fecha_str}', -30)
              AND tms_operacion <= '{fecha_str}'
            """
        )
        .join(icu_keys, on="id_icu", how="left_semi")
        .distinct()
        .withColumn("ind_digital_uso", F.lit(1))
    )

    digital_uso_agg = (
        digital.join(uso, on="id_icu", how="left")
        .groupBy("cliente_unico")
        .agg(
            F.max("tms_alta").alias("tms_alta"),
            F.max("ind_digital").alias("ind_digital"),
            F.max("ind_digital_uso").alias("ind_digital_uso"),
        )
    )

    # ----------------------------------------------------------------------
    # Cuarteles: UN solo scan filtrado (la base lee el histórico completo 2 veces)
    # Semántica vs. la base:
    #   max(fec) en (19010101, fecha_num] → distinct(CU, cuartel) → 1 fila.
    #   Un solo ORDER BY fec DESC, cuartel ASC equivale a eso y es estable
    #   (la base usaba rand() en el empate).
    # ----------------------------------------------------------------------
    w_cuartel = Window.partitionBy("cliente_unico").orderBy(
        F.col("fec_surtimiento").desc(),
        F.col("cuartel").asc_nulls_last(),
    )
    cuarteles_cliente = (
        spark.table(TBL_CUARTELES)
        .select(
            F.col("id_cliente").alias("cliente_unico"),
            F.col("cuartel"),
            F.col("fec_surtimiento"),
        )
        .where(
            (F.col("fec_surtimiento") > 19010101)
            & (F.col("fec_surtimiento") <= fecha_num)
        )
        .join(pivot_keys, on="cliente_unico", how="inner")
        .withColumn("numero_fila", F.row_number().over(w_cuartel))
        .where(F.col("numero_fila") == 1)
        .select("cliente_unico", "cuartel")
    )

    # ----------------------------------------------------------------------
    # Uniones — mismo orden que la base para conservar columnas
    # ----------------------------------------------------------------------
    base_pivote_s0 = base_pivote.join(cerebro, on=["cliente_unico"], how="left")
    base_pivote_s1 = base_pivote_s0.join(lae, on=["cliente_unico"], how="left")
    base_pivote_s2 = base_pivote_s1.join(digital_uso_agg, on="cliente_unico", how="left")
    base_pivote_s3 = (
        base_pivote_s2.join(cuarteles_cliente, on="cliente_unico", how="left")
        .repartition(N_PARTS, "id_master")
    )

    base_pivote_master = nbco.join(cltv_futuro_hog, on=["id_master"], how="left")
    base_pivote_master_s1 = base_pivote_master.join(cltv_futuro_con, on=["id_master"], how="left")
    base_pivote_master_s2 = base_pivote_master_s1.join(cltv_futuro_mov, on=["id_master"], how="left")
    base_pivote_master_s3 = base_pivote_master_s2.join(cltv_futuro_efe, on=["id_master"], how="left")
    base_pivote_master_s4 = base_pivote_master_s3.join(cltv_real_rbs, on=["id_master"], how="left")
    base_pivote_master_s5 = base_pivote_master_s4.join(cltv_activo, on="id_master", how="left")

    base_pivote_final = base_pivote_s3.join(
        base_pivote_master_s5, on=["id_master"], how="inner"
    )

    partes = F.split(F.col("cliente_unico"), "-", -1)
    base_pivote_final = (
        base_pivote_final
        .withColumn("id_pais", partes[0])
        .withColumn("id_canal", partes[1])
        .withColumn("id_sucursal", partes[2])
        .withColumn("id_num_folio", partes[3])
        .withColumn("cltv_f_con", F.col("cltv_f_con") * F.col("p1_conectividad"))
        .withColumn("cltv_f_hog", F.col("cltv_f_hog") * (F.lit(1) - F.col("p1_conectividad")))
        .drop("cliente_unico")
    )

    # Un solo action: WRITE. persist+count de 8.1M filas anchas en 8g
    # tira a disco y te come ~40 min; el write posterior ya era "gratis".
    if escribir:
        spark.sparkContext.setJobDescription("04_write")
        _paso(f"WRITE {tabla_out} modo={modo} — este es el job grande")
        (
            base_pivote_final.write
            .format("parquet")
            .mode(modo)
            .partitionBy("num_periodo_sem")
            .saveAsTable(tabla_out)
        )
        _paso(f"WRITE OK  {tabla_out}")
        n_final = (
            spark.table(tabla_out)
            .where(F.col("num_periodo_sem") == semana)
            .count()
        )
        _paso(f"COUNT post-write  filas={n_final}")
        print(n_final)
    else:
        n_final = base_pivote_final.count()
        print(n_final)

    return base_pivote_final


# Alias por si en el notebook la pegaste como genera_layout_optimized
genera_layout_optimized = genera_layout


# Recursos (YARN): NO pongas executor.instances ni dynamicAllocation=false.
#   spark.stop()
#   spark = crear_spark()
#   # UI → Executors: espera a ver 2 ALIVE, luego:
#   genera_layout(202630, 202626, 202605, 202627, "append", tabla_out, spark, refrescar=False)
#
# NO hagas count() de cerebro/masters a mano: el driver de 1g se traba
# colectando el broadcast y no aparecen stages. El job grande es el COUNT final.
