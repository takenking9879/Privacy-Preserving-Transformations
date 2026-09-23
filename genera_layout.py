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
  - El desempate con rand() en cerebro/cuarteles es no determinista en AMBAS
    versiones. Si un cliente tiene varias filas candidatas distintas, la fila
    ganadora puede cambiar entre corridas. Con una sola fila candidata el
    resultado es idéntico.
"""

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Sesión (opcional). Si ya tienes `spark` creado afuera, no hace falta llamarla.
# ---------------------------------------------------------------------------
def crear_spark():
    return (
        SparkSession.builder
        .appName("create_nbco_base")
        .enableHiveSupport()
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", "3")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        .config("spark.executor.cores", "3")
        .config("spark.executor.memory", "9g")
        .config("spark.driver.memory", "1g")
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

    # AQE + menos particiones de shuffle (9 cores efectivos: 3 exec * 3 cores).
    # El default de 200 genera muchos archivos/tareas chicas y satura CPU/RAM.
    confs = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.localShuffleReader.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64m",
        "spark.sql.adaptive.coalescePartitions.minPartitionNum": "4",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum": "36",
        "spark.sql.shuffle.partitions": "36",
        "spark.sql.autoBroadcastJoinThreshold": str(64 * 1024 * 1024),
        "spark.sql.adaptive.autoBroadcastJoinThreshold": str(64 * 1024 * 1024),
        "spark.sql.broadcastTimeout": "600",
        "spark.sql.files.maxPartitionBytes": "128m",
        "spark.sql.inMemoryColumnarStorage.compressed": "true",
        "spark.sql.optimizer.dynamicPartitionPruning.enabled": "true",
    }
    for key, value in confs.items():
        session.conf.set(key, value)
    return session


def _maybe_broadcast(df, n_rows, limite=1_500_000):
    """Broadcast solo si cabe cómodo en el driver de 1g."""
    if n_rows <= limite:
        return F.broadcast(df)
    return df


def _dedup_rand(df, llave="cliente_unico"):
    """Misma semántica que la base: 1 fila al azar por llave."""
    ventana = Window.partitionBy(llave).orderBy(F.rand())
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
    refrescar=True,
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

    if refrescar:
        spark.sql(f"REFRESH TABLE {TBL_DIGITAL}")
        spark.sql(f"REFRESH TABLE {TBL_TXN}")

    # ----------------------------------------------------------------------
    # Fechas (tabla chica: collect de ~7 filas, igual que la base)
    # ----------------------------------------------------------------------
    fechas = spark.sql(
        f"SELECT fec_num FROM {TBL_FECHAS} WHERE num_periodo_sem={semana}"
    )
    dias_semana_num = [row.fec_num for row in fechas.collect()]
    if not dias_semana_num:
        raise ValueError(f"Sin fechas para num_periodo_sem={semana} en {TBL_FECHAS}")

    fecha_num = sorted(dias_semana_num)[-1]
    fecha_str = f"{str(fecha_num)[:4]}-{str(fecha_num)[4:6]}-{str(fecha_num)[6:]}"
    semana_ini = int((int(semana / 100) - 1) * 100 + semana % 100)

    print(semana_ini)
    print(fecha_num, fecha_str)
    print(dias_semana_num)

    # ----------------------------------------------------------------------
    # Universo de clientes (se reutiliza para recortar TODAS las fact tables)
    # ----------------------------------------------------------------------
    base_pivote = spark.sql(
        f"""
        SELECT cliente_unico, fecha_salida
        FROM {TBL_PIVOTE}
        WHERE fecha_salida={semana_cmp}
        """
    )

    pivot_keys_cached = (
        base_pivote.select("cliente_unico")
        .distinct()
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    n_cu = pivot_keys_cached.count()
    pivot_keys = _maybe_broadcast(pivot_keys_cached, n_cu)

    # ----------------------------------------------------------------------
    # Cerebro: recorte por semana + CUs del pivote, luego 1 fila random
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
    cerebro = _dedup_rand(cerebro, "cliente_unico")

    master_keys_cached = (
        cerebro.select("id_master")
        .where(F.col("id_master").isNotNull())
        .distinct()
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    n_master = master_keys_cached.count()
    master_keys = _maybe_broadcast(master_keys_cached, n_master)

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

    # ----------------------------------------------------------------------
    # CLTV futuros / activo / renta: solo id_master que pueden entrar al inner
    # ----------------------------------------------------------------------
    cltv_futuro_hog = (
        spark.sql(f"SELECT id_master, cltv AS cltv_f_hog FROM {TBL_CLTV_FUTURO_HOG}")
        .join(master_keys, on="id_master", how="left_semi")
    )
    cltv_futuro_con = (
        spark.sql(f"SELECT id_master, cltv AS cltv_f_con FROM {TBL_CLTV_FUTURO_CON}")
        .join(master_keys, on="id_master", how="left_semi")
    )
    cltv_futuro_efe = (
        spark.sql(f"SELECT id_master, cltv AS cltv_f_efe FROM {TBL_CLTV_FUTURO_EFE}")
        .join(master_keys, on="id_master", how="left_semi")
    )
    cltv_futuro_mov = (
        spark.sql(f"SELECT id_master, cltv AS cltv_f_mov FROM {TBL_CLTV_FUTURO_MOV}")
        .join(master_keys, on="id_master", how="left_semi")
    )
    cltv_activo = (
        spark.sql(f"SELECT id_master, clvpa AS cltv_activo FROM {TBL_CLTV_ACTIVO}")
        .join(master_keys, on="id_master", how="left_semi")
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
        .join(master_keys, on="id_master", how="left_semi")
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
    uso = (
        spark.sql(
            f"""
            SELECT DISTINCT id_icu, 1 AS ind_digital_uso
            FROM {TBL_TXN}
            WHERE tms_operacion > DATE_ADD('{fecha_str}', -30)
              AND tms_operacion <= '{fecha_str}'
            """
        )
        .join(icu_keys, on="id_icu", how="left_semi")
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
    # Semántica idéntica:
    #   max(fec_surtimiento) en (19010101, fecha_num] → distinct(CU, cuartel)
    #   → 1 fila random por cliente
    # ----------------------------------------------------------------------
    cuarteles_filtrado = (
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
    )

    w_max_fec = Window.partitionBy("cliente_unico")
    cuarteles_cliente = (
        cuarteles_filtrado
        .withColumn("max_fec", F.max("fec_surtimiento").over(w_max_fec))
        .where(F.col("fec_surtimiento") == F.col("max_fec"))
        .select("cliente_unico", "cuartel")
        .distinct()
    )
    cuarteles_cliente = _dedup_rand(cuarteles_cliente, "cliente_unico")

    # ----------------------------------------------------------------------
    # Uniones — mismo orden que la base para conservar columnas
    # ----------------------------------------------------------------------
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

    # Un solo valor de num_periodo_sem: 9 archivos (~1 por core) evitan
    # cientos de part-files chicos sin crear archivos de decenas de GB.
    base_pivote_final = base_pivote_final.coalesce(9)
    base_pivote_final.persist(StorageLevel.MEMORY_AND_DISK)
    print(base_pivote_final.count())

    if escribir:
        (
            base_pivote_final.write
            .format("parquet")
            .mode(modo)
            .partitionBy("num_periodo_sem")
            .saveAsTable(tabla_out)
        )

    try:
        pivot_keys_cached.unpersist()
    except Exception:
        pass
    try:
        master_keys_cached.unpersist()
    except Exception:
        pass

    return base_pivote_final
