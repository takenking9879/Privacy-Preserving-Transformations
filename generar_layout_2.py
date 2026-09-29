# -*- coding: utf-8 -*-
"""
generar_layout_2.py

Copia este archivo al notebook.

    # 1) Qué semanas/mes poner (mira tablas reales, no solo el calendario)
    sug = sugerir_parametros(spark)
    # 2) Layout (mismas 5 args que la original)
    genera_layout(sug.semana_cmp, sug.semana_cltv, sug.mes_nbco, sug.semana_lae, "append", spark=spark)
    # 3) Lo que pidió tu jefe: 1 cuartel por cliente
    genera_cuartel_por_cliente(sug.semana_cmp, "overwrite", spark=spark)

Parámetros de genera_layout(semana_cmp, semana_cltv, mes_nbco, semana_lae, modo):
  1 semana_cmp   insumos / servilleta (la más actual)
  2 semana_cltv  último domingo de cada mes en fechas_cat (tope=hoy), <= insumo
  3 mes_nbco     YYYYMM. NBCO de mes M queda completa el día 5 de M+1
  4 semana_lae   última semana IMPAR previa al insumo (mismo lag que NBCO, en impar)
  5 modo         append | overwrite

Las 3 fechas de apoyo NUNCA pueden ser posteriores al insumo.
El calendario (fechas_cat) solo dice qué semana "debería" existir.
La fuente de verdad es MAX/SHOW TABLES de cada tabla (equipos desfasados).
"""

from __future__ import annotations

import builtins
import re
import sys
from datetime import date, datetime
from types import SimpleNamespace

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# ===========================================================================
# FROM — edita aquí los nombres de tabla
# ===========================================================================
def fuentes(semana_cltv=None):
    semana = semana_cltv
    return {
        "fechas": "cd_baz_bdclientes.cd_gen_fechas_cat",
        "pivote": "ws_celcobd_analitica.ta_338082_servilleta_total_cu_240826",
        "cerebro": "ec_baz_bdclientes.ec_cre_comportamental_layout_cerebro_full",
        "lae": "ws_celcobd_analitica.lae_interno_mensual",
        "cltv_hog_tpl": "ws_ektcomd_analitica.tt_1034848_cltv_futuros_hog_{semana}_v2",
        "cltv_con_tpl": "ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2",
        "cltv_efe_tpl": "ws_ektcomd_analitica.tt_1034848_cltv_futuros_efe_{semana}_v2",
        # El original lee MOV desde la tabla CON.
        "cltv_mov_tpl": "ws_ektcomd_analitica.tt_1034848_cltv_futuros_con_{semana}_v2",
        "cltv_renta": "ws_aarent_analitica.cu_renta_credito_operaciones_cliente",
        "cltv_activo_tpl": "ws_celcobd_analitica.`1034848_cltvpa_final_{semana}`",
        "nbco": "ma_bdbaz.ml_cre_recomendacion_nbco_existentes",
        "digital": "cd_baz_bdclientes.cd_dig_clientes",
        "txn": "cd_baz_bdclientes.cd_dig_txn_financieras",
        "cuarteles": "ws_celcobd_analitica.tt_1117735_pedidoshistoricos_cuartel",
        "out_layout": "ws_ektcomd_analitica.ma_cyc_variables_agente_campania_v4",
        "out_cuartel_cliente": "ws_ektcomd_analitica.ma_cyc_cuartel_por_cliente",
    }


def _tbl_semana(tpl, semana):
    return tpl.format(semana=semana)


# ===========================================================================
# Spark
# ===========================================================================
def _paso(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


def crear_spark():
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
    for key, value in {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.minPartitionNum": "8",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum": "24",
        "spark.sql.autoBroadcastJoinThreshold": str(16 * 1024 * 1024),
        "spark.sql.adaptive.autoBroadcastJoinThreshold": str(16 * 1024 * 1024),
        "spark.sql.files.maxPartitionBytes": "128m",
    }.items():
        session.conf.set(key, value)
    try:
        if int(session.conf.get("spark.sql.shuffle.partitions")) >= 100:
            session.conf.set("spark.sql.shuffle.partitions", "24")
    except Exception:
        session.conf.set("spark.sql.shuffle.partitions", "24")
    return session


def _dedup_estable(df, llave, *orden):
    w = Window.partitionBy(llave).orderBy(*orden)
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


# ===========================================================================
# Calendario / reglas (sin Spark donde se pueda)
# ===========================================================================
def fecha_hoy_num(hoy=None):
    """YYYYMMDD. Default: hoy. No hardcodear 20260902."""
    d = hoy or date.today()
    if isinstance(d, datetime):
        d = d.date()
    return int(d.strftime("%Y%m%d"))


def mes_nbco_por_regla(hoy=None):
    """
    NBCO del mes M queda completa el día 5 de M+1.
    5-may → ya está abril (202604). 4-may → último completo es marzo.
    """
    d = hoy or date.today()
    if isinstance(d, datetime):
        d = d.date()
    if d.day >= 5:
        y, m = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    else:
        # retrocede 2 meses
        m0 = d.month - 2
        y, m = (d.year, m0) if m0 > 0 else (d.year - 1, m0 + 12)
    return y * 100 + m


def mes_de_semana(semana):
    """Aprox YYYYMM a partir de YYYYWW (la semana no cruza de año en este código)."""
    return int(semana) // 100 * 100 + builtins.min(12, builtins.max(1, (int(semana) % 100 + 3) // 4))


def es_semana_impar(semana):
    """202635 → True (35 impar). LAE solo se publica en semanas impares."""
    return (int(semana) % 100) % 2 == 1


def _semana_previa(semana):
    year, ww = divmod(int(semana), 100)
    if ww <= 1:
        return (year - 1) * 100 + 53
    return year * 100 + (ww - 1)


def semana_lae_por_regla(semana_ref):
    """
    Igual que NBCO usa el mes *previo* completo: LAE usa la última
    semana IMPAR estrictamente anterior a la semana del insumo.
    202639 (impar, actual) → 202637.  202630 (par) → 202629.
    """
    w = _semana_previa(semana_ref)
    while not es_semana_impar(w):
        w = _semana_previa(w)
    return w


# ===========================================================================
# Lectura de metadatos (barato: metastore / SHOW PARTITIONS, NUNCA scan)
# ===========================================================================
def _dom_tabla(nombre):
    nombre = str(nombre).replace("`", "")
    if "." in nombre:
        return nombre.split(".", 1)[0]
    return None


def _nombre_tabla(nombre):
    return str(nombre).replace("`", "").split(".", 1)[-1]


def _existe_tabla(spark, nombre):
    """Catálogo primero. tableExists a menudo miente con tablas Hive; DESCRIBE es el fallback."""
    nombre = nombre.replace("`", "")
    try:
        if spark.catalog.tableExists(nombre):
            return True
    except Exception:
        pass
    try:
        spark.sql(f"DESCRIBE TABLE {nombre}").limit(1).collect()
        return True
    except Exception:
        return False


def _valores_particion(spark, tabla, col):
    try:
        spark.sparkContext.setJobDescription(f"sugerir_SHOW_PARTITIONS_{col}")
        rows = spark.sql(f"SHOW PARTITIONS {tabla}").collect()
    except Exception:
        return None
    vals = []
    for r in rows:
        texto = r[0] if r[0] is not None else ""
        for parte in str(texto).split("/"):
            if parte.startswith(f"{col}="):
                bruto = parte.split("=", 1)[1]
                try:
                    vals.append(int(bruto))
                except ValueError:
                    pass
    return vals


def _max_particion(spark, tabla, col, tope=None, solo_impares=False):
    _paso(f"  particiones {col} ← {tabla}")
    vals = _valores_particion(spark, tabla, col)
    if not vals:
        _paso(f"  sin particiones de {col}")
        return None
    if tope is not None:
        vals = [v for v in vals if v <= int(tope)]
    if solo_impares:
        vals = [v for v in vals if es_semana_impar(v)]
    return builtins.max(vals) if vals else None


def _max_col_sql(spark, tabla, col, tope=None):
    """MAX de una columna (para el pivote, que no está particionado)."""
    _paso(f"  MAX({col}) ← {tabla}  (un scan chico de 1 columna)")
    spark.sparkContext.setJobDescription(f"sugerir_MAX_{col}")
    q = f"SELECT MAX({col}) AS m FROM {tabla}"
    if tope is not None:
        q += f" WHERE {col} <= {int(tope)}"
    try:
        val = spark.sql(q).collect()[0]["m"]
        return int(val) if val is not None else None
    except Exception as exc:
        _paso(f"  WARN MAX({col}) falló: {exc}")
        return None


def _semanas_cltv_en_catalogo(spark, tpl_hog, tope, n_probar=16):
    """SHOW TABLES LIKE (metastore). tableExists a menudo no ve tablas Hive."""
    db = _dom_tabla(tpl_hog)
    encontradas = []
    pat = re.compile(r"hog_(\d{6})", re.I)
    name_glob = _nombre_tabla(tpl_hog).replace("{semana}", "*")
    likes = []
    if name_glob and name_glob != "*":
        likes.append(name_glob)
        likes.append(name_glob.replace("*", "%"))
    likes.extend(["*cltv_futuros_hog*", "%cltv_futuros_hog%"])

    if db:
        for like in likes:
            _paso(f"  SHOW TABLES IN {db} LIKE '{like}'")
            try:
                spark.sparkContext.setJobDescription("sugerir_SHOW_TABLES_cltv")
                rows = spark.sql(f"SHOW TABLES IN {db} LIKE '{like}'").collect()
            except Exception as exc:
                _paso(f"  WARN SHOW TABLES LIKE {like}: {exc}")
                continue
            for r in rows:
                d = r.asDict() if hasattr(r, "asDict") else {}
                name = d.get("tableName") or d.get("table") or (r[1] if len(r) > 1 else r[0])
                m = pat.search(str(name))
                if m:
                    w = int(m.group(1))
                    if w <= int(tope):
                        encontradas.append(w)
            if encontradas:
                break
    if encontradas:
        return sorted(set(encontradas), reverse=True)

    _paso("  fallback: DESCRIBE hog_{semana} (tableExists falla en Hive)")
    for w in _semanas_calendario_atras(spark, tope, n=n_probar):
        nombre = _tbl_semana(tpl_hog, w)
        if _existe_tabla(spark, nombre):
            encontradas.append(w)
            if len(encontradas) >= 6:
                break
    return encontradas


def _semanas_calendario_atras(spark, tope, n=25, src=None):
    src = src or fuentes()["fechas"]
    try:
        rows = spark.sql(
            f"""
            SELECT DISTINCT num_periodo_sem
            FROM {src}
            WHERE num_periodo_sem <= {int(tope)}
            ORDER BY num_periodo_sem DESC
            """
        ).take(n)
        return [int(r["num_periodo_sem"]) for r in rows]
    except Exception:
        # sin catálogo: decrementa YYYYWW de forma tosca
        out, w = [], int(tope)
        for _ in range(n):
            out.append(w)
            year, ww = divmod(w, 100)
            w = (year - 1) * 100 + 53 if ww <= 1 else year * 100 + (ww - 1)
        return out


def _domingos_calendario(spark, hoy_num, src=None):
    """
    Tu query de CLTV / contraste:

        SELECT MAX(num_periodo_sem) AS semana, MAX(fec_num) AS fecha
        FROM cd_gen_fechas_cat
        WHERE num_dia_sem='7'
          AND fec_num BETWEEN {anio}0101 AND {hoy}
        GROUP BY num_periodo_mes
        ORDER BY num_periodo_mes DESC

    El tope de fecha es hoy (no 20260902).
    """
    src = src or fuentes()["fechas"]
    anio = hoy_num // 10000
    return spark.sql(
        f"""
        SELECT
            num_periodo_mes,
            MAX(num_periodo_sem) AS semana,
            MAX(fec_num) AS fecha
        FROM {src}
        WHERE num_dia_sem = '7'
          AND fec_num BETWEEN {anio}0101 AND {hoy_num}
        GROUP BY num_periodo_mes
        ORDER BY num_periodo_mes DESC
        """
    )


def _domingos_por_mes(spark, hoy_num, tope_semana=None, src=None):
    """Último domingo de cada mes (YYYYWW), más reciente primero."""
    _paso(f"  fechas_cat  domingos/mes  tope_fec={hoy_num}  tope_sem={tope_semana}")
    try:
        spark.sparkContext.setJobDescription("sugerir_domingos_mes_cltv")
        rows = _domingos_calendario(spark, hoy_num, src).collect()
    except Exception as exc:
        _paso(f"  WARN fechas_cat domingos/mes: {exc}")
        return []
    out = []
    for r in rows:
        if r["semana"] is None:
            continue
        w = int(r["semana"])
        if tope_semana is not None and w > int(tope_semana):
            continue
        out.append(w)
    return out


def _mes_de_semana_cat(spark, semana, src=None):
    src = src or fuentes()["fechas"]
    try:
        row = spark.sql(
            f"""
            SELECT MAX(num_periodo_mes) AS mes
            FROM {src}
            WHERE num_periodo_sem = {int(semana)}
            """
        ).collect()[0]
        if row["mes"] is not None:
            return int(row["mes"])
    except Exception:
        pass
    return mes_de_semana(semana)


def _semana_domingo_max(spark, hoy_num, src=None):
    src = src or fuentes()["fechas"]
    row = spark.sql(
        f"""
        SELECT MAX(num_periodo_sem) AS semana
        FROM {src}
        WHERE num_dia_sem = '7' AND fec_num <= {hoy_num}
        """
    ).collect()[0]
    return int(row["semana"]) if row["semana"] is not None else None


# ===========================================================================
# Sugeridor
# ===========================================================================
def sugerir_parametros(spark, fecha_hoy=None, semana_cmp=None, imprimir=True):
    """
    Sugiere (semana_cmp, semana_cltv, mes_nbco, semana_lae).

    - Insumo: MAX(fecha_salida) del pivote (partición o scan de 1 col),
              sin pasarse del último domingo <= hoy.
    - CLTV:   último domingo de cada mes en fechas_cat (tope=hoy), <= insumo.
              Si la hog de ese domingo no existe, se recorta al domingo de mes
              más nuevo que sí tenga tabla.
    - NBCO:   min(regla día 5, mes del insumo, MAX real de la tabla).
    - LAE:    min(última impar previa al insumo, MAX impar real de la tabla).
              Mismo lag que NBCO (no uses la semana actual), solo impares.

    No ejecuta el layout. Tú decides si aceptas la sugerencia.
    """
    spark = _sesion_spark(spark)
    src = fuentes()
    hoy_num = fecha_hoy_num(fecha_hoy)
    alertas = []
    _paso(f"sugerir_parametros  hoy={hoy_num}  (metastore; MAX pivote solo si no hay partición)")

    _paso("calendario fechas_cat…")
    semana_cal = _semana_domingo_max(spark, hoy_num, src["fechas"])
    if semana_cal is None:
        raise ValueError(f"Sin semanas en {src['fechas']} con fec_num <= {hoy_num}")
    _paso(f"último domingo <= hoy: {semana_cal}")

    max_pivote = _max_particion(spark, src["pivote"], "fecha_salida", tope=semana_cal)
    if max_pivote is None:
        max_pivote = _max_col_sql(spark, src["pivote"], "fecha_salida", tope=semana_cal)
        if max_pivote is not None:
            alertas.append(
                "INSUMO: el pivote no está particionado por fecha_salida; "
                f"tomé MAX(fecha_salida)={max_pivote}."
            )
    if semana_cmp is None:
        semana_cmp = max_pivote or semana_cal
        if max_pivote is None:
            alertas.append(
                "INSUMO: no pude leer fecha_salida del pivote; "
                f"uso el domingo de calendario {semana_cal}. Pasa semana_cmp=… si ya la sabes."
            )
    semana_cmp = int(semana_cmp)
    if semana_cmp > semana_cal:
        alertas.append(
            f"INSUMO {semana_cmp} es posterior al último domingo de calendario "
            f"({semana_cal}, hoy={hoy_num}). Se recorta a {semana_cal}."
        )
        semana_cmp = semana_cal
    if max_pivote is not None and semana_cmp > max_pivote:
        alertas.append(
            f"INSUMO {semana_cmp} no está en el pivote (MAX={max_pivote}). "
            f"Usa {max_pivote}."
        )
        semana_cmp = max_pivote

    _paso("CLTV: último domingo de cada mes (fechas_cat)…")
    domingos_mes = _domingos_por_mes(
        spark, hoy_num, tope_semana=semana_cmp, src=src["fechas"]
    )
    regla_cltv = domingos_mes[0] if domingos_mes else None
    hog_ok = []
    for w in domingos_mes:
        if _existe_tabla(spark, _tbl_semana(src["cltv_hog_tpl"], w)):
            hog_ok.append(w)
    max_hog = hog_ok[0] if hog_ok else None
    if regla_cltv is None:
        _paso("  fechas_cat vacío; fallback SHOW TABLES hog_*")
        hog_ok = _semanas_cltv_en_catalogo(spark, src["cltv_hog_tpl"], semana_cmp)
        max_hog = hog_ok[0] if hog_ok else None
        semana_cltv = max_hog
        alertas.append("CLTV: fechas_cat no trajo domingos por mes <= insumo.")
    elif max_hog is None:
        semana_cltv = regla_cltv
        alertas.append(
            f"CLTV: fechas_cat sugiere {regla_cltv} (último domingo de mes) "
            f"pero no vi hog_{{semana}}_v2 en {domingos_mes[:8]}. "
            "Confirma si falta corrida."
        )
    else:
        semana_cltv = builtins.min(regla_cltv, max_hog)
        if regla_cltv > max_hog:
            alertas.append(
                f"CLTV: el último domingo de mes es {regla_cltv}, "
                f"pero la hog más nueva es {max_hog}."
            )
    if semana_cltv is not None and not _existe_tabla(
        spark, _tbl_semana(src["cltv_activo_tpl"], semana_cltv)
    ):
        alertas.append(
            f"CLTV activo no existe para {semana_cltv}: "
            f"{_tbl_semana(src['cltv_activo_tpl'], semana_cltv)}"
        )

    max_cerebro = _max_particion(spark, src["cerebro"], "num_periodo_sem", tope=semana_cmp)
    if semana_cltv and max_cerebro and semana_cltv > max_cerebro:
        alertas.append(
            f"CEREBRO solo llega a {max_cerebro} < CLTV {semana_cltv}. "
            f"El layout usa cerebro en la semana CLTV."
        )

    _paso("NBCO + LAE (particiones)…")
    mes_insumo = _mes_de_semana_cat(spark, semana_cmp, src["fechas"])
    mes_regla = mes_nbco_por_regla(fecha_hoy)
    max_nbco = _max_particion(
        spark, src["nbco"], "num_periodo_mes", tope=builtins.min(mes_insumo, mes_regla)
    )
    candidatos = [m for m in (mes_regla, mes_insumo, max_nbco) if m is not None]
    mes_nbco = builtins.min(candidatos) if candidatos else None
    if max_nbco is not None and mes_regla > max_nbco:
        alertas.append(
            f"NBCO: la regla del día 5 sugiere {mes_regla} pero la tabla "
            f"solo tiene hasta {max_nbco}."
        )
    if max_nbco is None:
        alertas.append(
            "NBCO: no pude leer particiones num_periodo_mes; uso min(regla, mes insumo)."
        )
    if mes_nbco is not None and mes_nbco > mes_insumo:
        alertas.append(f"NBCO {mes_nbco} no puede ser posterior al mes del insumo {mes_insumo}.")
        mes_nbco = mes_insumo

    regla_lae = semana_lae_por_regla(semana_cmp)
    max_lae = _max_particion(
        spark, src["lae"], "num_periodo_sem", tope=semana_cmp, solo_impares=True
    )
    if max_lae is None:
        semana_lae = regla_lae
        alertas.append(
            f"LAE: sin particiones impares de num_periodo_sem; "
            f"uso la última impar previa al insumo ({regla_lae}). Confirma en la tabla."
        )
    else:
        semana_lae = builtins.min(regla_lae, max_lae)
        if regla_lae > max_lae:
            alertas.append(
                f"LAE: la regla (última impar previa, como NBCO mes) sugiere "
                f"{regla_lae} pero la tabla solo tiene hasta {max_lae}."
            )

    sug = SimpleNamespace(
        semana_cmp=semana_cmp,
        semana_cltv=semana_cltv,
        mes_nbco=mes_nbco,
        semana_lae=semana_lae,
        modo="append",
        hoy_num=hoy_num,
        semana_calendario_max=semana_cal,
        disponible={
            "pivote_max": max_pivote,
            "cltv_regla": regla_cltv,
            "cltv_domingos_mes": domingos_mes[:12],
            "cltv_hog": max_hog,
            "cltv_tablas": hog_ok[:8],
            "cerebro_max": max_cerebro,
            "nbco_max": max_nbco,
            "nbco_regla": mes_regla,
            "nbco_mes_insumo": mes_insumo,
            "lae_max": max_lae,
            "lae_regla": regla_lae,
        },
        alertas=alertas,
        llamada=(
            f"genera_layout({semana_cmp}, {semana_cltv}, {mes_nbco}, {semana_lae}, 'append')"
            if None not in (semana_cmp, semana_cltv, mes_nbco, semana_lae)
            else None
        ),
    )

    if imprimir:
        _imprimir_sugerencia(sug, spark)
    return sug


def _imprimir_sugerencia(sug, spark):
    print("=" * 72)
    print(f"SUGERENCIA  hoy={sug.hoy_num}  domingo_cal={sug.semana_calendario_max}")
    print("=" * 72)
    print(f"  1 insumo / cmp   {sug.semana_cmp}   (MAX pivote={sug.disponible['pivote_max']})")
    print(
        f"  2 cltv           {sug.semana_cltv}   "
        f"(domingo_mes={sug.disponible['cltv_regla']}  "
        f"hog={sug.disponible['cltv_hog']}  "
        f"meses={sug.disponible['cltv_domingos_mes']})"
    )
    print(
        f"  3 nbco mes       {sug.mes_nbco}   "
        f"(regla_dia5={sug.disponible['nbco_regla']}  "
        f"mes_insumo={sug.disponible['nbco_mes_insumo']}  "
        f"MAX_tabla={sug.disponible['nbco_max']})"
    )
    print(
        f"  4 lae            {sug.semana_lae}   "
        f"(regla_impar={sug.disponible['lae_regla']}  "
        f"MAX_tabla_impar={sug.disponible['lae_max']})"
    )
    print()
    if sug.llamada:
        print(f"  {sug.llamada}")
    print()
    if sug.alertas:
        print("ALERTAS (el más cercano NO siempre es el correcto):")
        for a in sug.alertas:
            print(f"  - {a}")
    else:
        print("Sin alertas: calendario y tablas cuadran.")
    print()
    print("CLTV / contraste — último domingo de cada mes (fechas_cat, tope=hoy):")
    try:
        _domingos_calendario(spark, sug.hoy_num).show(12, truncate=False)
    except Exception as exc:
        print(f"  (no pude mostrar fechas_cat: {exc})")
    print("=" * 72)


def validar_parametros(spark, semana_cmp, semana_cltv, mes_nbco, semana_lae, fecha_hoy=None):
    """Compara lo que vas a pasar vs la sugerencia. No bloquea."""
    sug = sugerir_parametros(spark, fecha_hoy=fecha_hoy, semana_cmp=semana_cmp, imprimir=False)
    diffs = []
    pares = (
        ("semana_cmp", semana_cmp, sug.semana_cmp),
        ("semana_cltv", semana_cltv, sug.semana_cltv),
        ("mes_nbco", mes_nbco, sug.mes_nbco),
        ("semana_lae", semana_lae, sug.semana_lae),
    )
    for nombre, dado, reco in pares:
        if reco is not None and int(dado) != int(reco):
            diffs.append(f"{nombre}: tú={dado}  sugerido={reco}")
    _paso("Validación de parámetros")
    if diffs:
        print("DIFIERE de la sugerencia:")
        for d in diffs:
            print(f"  - {d}")
    else:
        print("Tus 4 fechas coinciden con la sugerencia.")
    for a in sug.alertas:
        print(f"  ALERTA: {a}")
    return sug


# ===========================================================================
# 1 cuartel por cliente (pedido del jefe)
# ===========================================================================
def df_cuartel_por_cliente(spark, fecha_num, pivot_keys=None, src=None):
    """
    1 fila por cliente_unico: el pedido con fec_surtimiento más reciente
    <= fecha_num (y > 19010101). Empate: cuartel ASC.

    Esto es el grano 'a nivel cliente' — no id_master, no NBCO.
    """
    src = src or fuentes()
    w = Window.partitionBy("cliente_unico").orderBy(
        F.col("fec_surtimiento").desc(),
        F.col("cuartel").asc_nulls_last(),
    )
    df = (
        spark.table(src["cuarteles"])
        .select(
            F.col("id_cliente").alias("cliente_unico"),
            F.col("cuartel"),
            F.col("fec_surtimiento"),
        )
        .where(
            (F.col("fec_surtimiento") > 19010101)
            & (F.col("fec_surtimiento") <= int(fecha_num))
        )
    )
    if pivot_keys is not None:
        df = df.join(pivot_keys, on="cliente_unico", how="inner")
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


def genera_cuartel_por_cliente(
    semana_ref,
    modo="overwrite",
    tabla_out=None,
    spark=None,
    solo_pivote=True,
    semana_cmp=None,
):
    """
    Proceso aparte: tabla cliente_unico × 1 cuartel.

    semana_ref  semana con la que se toma el domingo (fecha de corte).
    solo_pivote True = solo CUs del insumo (semana_cmp o semana_ref).
                False = todo cliente que tenga pedido histórico.
    """
    spark = _sesion_spark(spark)
    src = fuentes()
    if tabla_out is None:
        tabla_out = src["out_cuartel_cliente"]
    semana_cmp = int(semana_cmp or semana_ref)

    fechas = spark.sql(
        f"SELECT fec_num FROM {src['fechas']} WHERE num_periodo_sem={int(semana_ref)}"
    )
    dias = [r.fec_num for r in fechas.collect()]
    if not dias:
        raise ValueError(f"Sin fechas para semana {semana_ref}")
    fecha_num = builtins.max(dias)

    pivot_keys = None
    if solo_pivote:
        pivot_keys = (
            spark.sql(
                f"""
                SELECT DISTINCT cliente_unico
                FROM {src['pivote']}
                WHERE fecha_salida={semana_cmp}
                """
            )
        )

    _paso(f"cuartel_por_cliente  corte={fecha_num}  solo_pivote={solo_pivote}")
    out = (
        df_cuartel_por_cliente(spark, fecha_num, pivot_keys, src)
        .withColumn("num_periodo_sem", F.lit(int(semana_ref)))
        .repartition(24, "cliente_unico")
    )
    (
        out.write.format("parquet")
        .mode(modo)
        .partitionBy("num_periodo_sem")
        .saveAsTable(tabla_out)
    )
    n = spark.table(tabla_out).where(F.col("num_periodo_sem") == int(semana_ref)).count()
    _paso(f"WRITE OK {tabla_out}  filas={n}  (1 cuartel por cliente)")
    print(n)
    return out


# ===========================================================================
# Layout
# ===========================================================================
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
    grano="master",
    sugerir_si_difiere=False,
):
    """
    grano='master'  — igual que la original (tira cliente_unico, inner NBCO).
    grano='cliente' — deja cliente_unico (versión a nivel cliente + 1 cuartel).
    """
    semana = int(semana_cltv)
    src = fuentes(semana)
    if tabla_out is None:
        tabla_out = src["out_layout"]

    spark = _sesion_spark(spark)
    try:
        _paso(
            f"Spark listo  app={spark.sparkContext.applicationId}  "
            f"ui={getattr(spark.sparkContext, 'uiWebUrl', None)}"
        )
    except Exception as exc:
        _paso(f"Spark listo ({exc})")

    if sugerir_si_difiere:
        try:
            validar_parametros(spark, semana_cmp, semana_cltv, mes, semana_lae)
        except Exception as exc:
            _paso(f"Sugeridor omitido: {exc}")

    if refrescar:
        spark.sql(f"REFRESH TABLE {src['digital']}")
        spark.sql(f"REFRESH TABLE {src['txn']}")

    spark.sparkContext.setJobDescription("01_fechas")
    fechas = spark.sql(
        f"SELECT fec_num FROM {src['fechas']} WHERE num_periodo_sem={semana}"
    )
    dias_semana_num = [row.fec_num for row in fechas.collect()]
    if not dias_semana_num:
        raise ValueError(f"Sin fechas para num_periodo_sem={semana}")

    fecha_num = sorted(dias_semana_num)[-1]
    fecha_str = f"{str(fecha_num)[:4]}-{str(fecha_num)[4:6]}-{str(fecha_num)[6:]}"
    semana_ini = int((int(semana / 100) - 1) * 100 + semana % 100)
    _paso(f"fechas OK  semana_ini={semana_ini}  fecha_num={fecha_num}")
    print(semana_ini)
    print(fecha_num, fecha_str)
    print(dias_semana_num)

    N_PARTS = 24
    base_pivote = spark.sql(
        f"""
        SELECT cliente_unico, fecha_salida
        FROM {src['pivote']}
        WHERE fecha_salida={int(semana_cmp)}
        """
    ).repartition(N_PARTS, "cliente_unico")
    pivot_keys = base_pivote.select("cliente_unico").distinct()

    cerebro = spark.sql(
        f"""
        SELECT
            id_master, id_cte_unico AS cliente_unico,
            ind_inac_24meses, semanas_inactivo, est_ingresos_indirectos,
            tipo_sol_cte, marca_cliente_bueno, antig_tl, num_periodo_sem
        FROM {src['cerebro']}
        WHERE num_periodo_sem={semana}
        """
    ).join(pivot_keys, on="cliente_unico", how="inner")
    cerebro = _dedup_estable(
        cerebro, "cliente_unico",
        F.col("id_master").asc_nulls_last(),
        F.col("antig_tl").asc_nulls_last(),
    )
    master_keys = cerebro.select("id_master").where(F.col("id_master").isNotNull()).distinct()

    lae = spark.sql(
        f"""
        SELECT cliente_unico, PD3, pd5, pd7, pd10, pd20, segmento
        FROM {src['lae']}
        WHERE num_periodo_sem={int(semana_lae)}
        """
    ).join(pivot_keys, on="cliente_unico", how="inner")

    cltv_futuro_hog = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_hog FROM {_tbl_semana(src['cltv_hog_tpl'], semana)}"
    )
    cltv_futuro_con = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_con FROM {_tbl_semana(src['cltv_con_tpl'], semana)}"
    )
    cltv_futuro_efe = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_efe FROM {_tbl_semana(src['cltv_efe_tpl'], semana)}"
    )
    cltv_futuro_mov = spark.sql(
        f"SELECT id_master, cltv AS cltv_f_mov FROM {_tbl_semana(src['cltv_mov_tpl'], semana)}"
    )
    cltv_activo = spark.sql(
        f"SELECT id_master, clvpa AS cltv_activo FROM {_tbl_semana(src['cltv_activo_tpl'], semana)}"
    )

    cltv_real_rbs = (
        spark.table(src["cltv_renta"])
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
        spark.table(src["nbco"])
        .where(F.col("num_periodo_mes") == int(mes))
        .drop("fcusuariocreacion", "fdfechacreacion", "num_periodo_mes")
        .repartition(N_PARTS, "id_master")
    )

    digital = spark.sql(
        f"""
        SELECT id_cliente_unico AS cliente_unico, id_icu, tms_alta, 1 AS ind_digital
        FROM {src['digital']}
        WHERE tms_alta <= '{fecha_str}'
        """
    ).join(pivot_keys, on="cliente_unico", how="inner")
    icu_keys = digital.select("id_icu").distinct()
    uso = (
        spark.sql(
            f"""
            SELECT id_icu FROM {src['txn']}
            WHERE tms_operacion > DATE_ADD('{fecha_str}', -30)
              AND tms_operacion <= '{fecha_str}'
            """
        )
        .join(icu_keys, on="id_icu", how="left_semi")
        .distinct()
        .withColumn("ind_digital_uso", F.lit(1))
    )
    digital_uso_agg = digital.join(uso, on="id_icu", how="left").groupBy("cliente_unico").agg(
        F.max("tms_alta").alias("tms_alta"),
        F.max("ind_digital").alias("ind_digital"),
        F.max("ind_digital_uso").alias("ind_digital_uso"),
    )

    cuarteles_cliente = df_cuartel_por_cliente(spark, fecha_num, pivot_keys, src).select(
        "cliente_unico", "cuartel"
    )

    base_pivote_s3 = (
        base_pivote.join(cerebro, on=["cliente_unico"], how="left")
        .join(lae, on=["cliente_unico"], how="left")
        .join(digital_uso_agg, on="cliente_unico", how="left")
        .join(cuarteles_cliente, on="cliente_unico", how="left")
        .repartition(N_PARTS, "id_master")
    )

    base_pivote_master_s5 = (
        nbco.join(cltv_futuro_hog, on=["id_master"], how="left")
        .join(cltv_futuro_con, on=["id_master"], how="left")
        .join(cltv_futuro_mov, on=["id_master"], how="left")
        .join(cltv_futuro_efe, on=["id_master"], how="left")
        .join(cltv_real_rbs, on=["id_master"], how="left")
        .join(cltv_activo, on="id_master", how="left")
    )

    how_final = "left" if grano == "cliente" else "inner"
    base_pivote_final = base_pivote_s3.join(
        base_pivote_master_s5, on=["id_master"], how=how_final
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
    )
    if grano != "cliente":
        base_pivote_final = base_pivote_final.drop("cliente_unico")

    if escribir:
        spark.sparkContext.setJobDescription("04_write")
        _paso(f"WRITE {tabla_out} modo={modo} grano={grano}")
        (
            base_pivote_final.write.format("parquet")
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
        print(base_pivote_final.count())
    return base_pivote_final


# Uso:
#   spark = crear_spark()          # o tu sesión
#   sug = sugerir_parametros(spark)
#   genera_layout(sug.semana_cmp, sug.semana_cltv, sug.mes_nbco, sug.semana_lae, "append", spark=spark)
#   genera_cuartel_por_cliente(sug.semana_cmp, "overwrite", spark=spark)
#   genera_layout(..., grano="cliente")   # layout sin dropear cliente_unico
