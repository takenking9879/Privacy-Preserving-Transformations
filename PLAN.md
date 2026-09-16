# Plan: fidelidad de datos sintéticos (X → X')

**Repo:** `Privacy-Preserving-Transformations`  
**Audiencia:** Jorge (software engineer / data scientist)  
**Alcance:** utilidad y estructura estadística de una tabla sintética. **No** es el problema de las ramas hermanas (`Z = f(X)` con clave de dueño). Aquí el trainer ve filas nuevas `X'` en el **mismo** espacio de columnas que `X`.

**Apuesta primaria:** `SequentialCARTSynthesizer` (estilo synthpop) como generador por defecto.  
**Apuesta secundaria (si hay tiempo):** ensemble `CART ∪ ConditionalMixture` elegido por `fidelity_score` en hold-out.  
**No apostamos** a GAN/CTGAN/TVAE/diffusion ni a DP-Laplace global.

---

## 1. Objetivo (restate)

Tenemos una tabla original `(X, Y)` y queremos una tabla sintética `(X', Y')` del mismo esquema tal que:

1. **Estructura de `X`:** patrones entre variables (marginales, correlaciones, interacciones, segmentos) se conservan en `X'`.
2. **Relación `X → Y`:** la función condicional `P(Y | X)` se conserva en `P(Y' | X')`.
3. **TSTR ≈ TRTR:** un modelo `M'` entrenado solo en `(X', Y')` debe rendir casi igual al evaluar en datos **originales** que un modelo `M` entrenado en `(X, Y)` (mismo family, mismos hiperparámetros congelados).
4. **Predicciones de un modelo ya entrenado:** si `M` se entrenó en original, las predicciones `M(X)` y `M(X')` deben ser cercanas **en distribución** (`Y ≈ Y'` a nivel de scores, no fila a fila).

Implicación operativa: `X'` es un **sustituto de modelado**, no un cifrado ni un disfraz por fila. No existe pairing `i ↔ i'`. Comparar filas 1-a-1 es un error de evaluación.

```
(X, Y)  --synthesizer S-->  (X', Y')
M  = fit(X,  Y)     # TRTR / TRTS
M' = fit(X', Y')    # TSTR
queremos  score(M', X_holdout) ≈ score(M, X_holdout)
y         Law(M(X)) ≈ Law(M(X'))
```

---

## 2. Apuesta: qué sintetizador primario y por qué

### 2.1 Primario — Sequential CART / synthpop

Implementar `src/synthesizers/sequential_cart.py` → clase `SequentialCARTSynthesizer`.

**Algoritmo (`fit` / `sample`):**

1. Fijar un orden de columnas `π`. Default: `Y` al final; el resto por `mutual_info` descendente con `Y`, desempate por cardinalidad (baja primero). Exponer `--column-order {mi_y, corr_y, random, file}`.
2. Primera columna `X_{π1}`: resampling empírico (con bootstrap de filas).
3. Para cada columna siguiente `j`, ajustar un `DecisionTreeRegressor` / `DecisionTreeClassifier` (`sklearn`) de `X_{π<j}` → `X_{πj}` (o `Y` si es el último).
4. En cada hoja, guardar la **distribución empírica** de la respuesta (no solo la media). En `sample`, bajar el árbol con los padres ya sintéticos y **sortear un valor de la hoja** (con suavizado opcional en continuas: jitter gaussiano *dentro* de la hoja, `σ = 0.2 · std_hoja`).
5. Tipos mixtos: one-hot / ordinal encode de padres categóricos; `min_samples_leaf` por defecto `max(5, floor(0.01 n))`; `max_depth=None` pero `min_samples_leaf` es el regularizador.
6. Missing values: categoría explícita `__NA__` o indicador + mediana; no dropna silencioso.

**Por qué esta es la apuesta:**

- Captura **interacciones** y no-linealidades (splits) sin asumir copula gaussiana.
- Nativo a **tipos mixtos** (numérico + categórico + `Y` binario/continuo).
- CPU-only, `sklearn` + `pandas` + `numpy`. Cero `torch`. Encaja con el stack de las ramas `src/` ya existentes.
- Literatura de oficinas estadísticas (`synthpop` en R, Nowok / Raab / Dibben) muestra TSTR competitivo en tablas de `d ≲ 40`.
- Falla de forma *interpretable*: orden de columnas malo, hojas vacías, leakage si `Y` se usa como padre de `X` (prohibido).

**Hiperparámetros a grid-ear (pequeño):**

| knobs | valores |
| --- | --- |
| `min_samples_leaf` | `{5, 20, 50}` |
| `column_order` | `{mi_y, corr_y}` |
| `leaf_jitter` | `{0.0, 0.2}` |

No hacer random search grande. Tres knobs × 2 seeds bastan para decidir el default.

### 2.2 Copula gaussiana — control de estructura lineal

`src/synthesizers/gaussian_copula.py` → `GaussianCopulaSynthesizer`.

- Marginales: KDE gaussiana o `sklearn.quantile` empírica (continuas); empíricas discretas (categóricas).
- Dependencia: correlación de Spearman → copula gaussiana (`scipy.stats.norm.ppf` + `numpy.linalg.cholesky` de la matriz de rangos).
- `Y` entra en la copula (no se genera independiente).

**Por qué sí, pero no como primario:** recupera **marginales + correlación lineal** de forma estable y barata. Pierde interacciones y multimodalidad. Es el *upper bound* de “¿cuánto de la utilidad es solo corr + marginals?”. Si CART no gana a la copula en TSTR, el dataset es esencialmente elíptico y no hace falta el árbol.

### 2.3 Mixture condicional / estratificación — multimodalidad y efectos de segmento

`src/synthesizers/conditional_mixture.py` → `ConditionalMixtureSynthesizer`.

1. Elegir variables de estrato `S` (default: `Y` si es clase; si `Y` es continuo, quantiles de `Y` + 1–2 categóricas de alta MI).
2. Por estrato `s`: fit `BayesianGaussianMixture` (`sklearn`, `n_components ∈ {2,4,8}`, `weight_concentration_prior` Dirichlet) sobre las continuas; categóricas restantes por tablas condicionales `P(C | s)` o CART local.
3. Sample: sortear `s ~ P(S)`, luego `X | s`.

**Por qué:** CART global suaviza modos si las hojas mezclan segmentos. Un mixture **por clase / por decil de `Y`** preserva `P(X | Y=s)` que es exactamente lo que TSTR necesita. Riesgo: estratos pequeños → overfitting. Gate: no estratificar si `n_s < 80`.

### 2.4 Ensemble (solo si hay tiempo)

`src/synthesizers/ensemble.py` → `MixtureCartEnsemble`:

- Generar `X'_CART` y `X'_MIX`.
- Elegir el que maximice `fidelity_score` en un **hold-out de síntesis** (20% de `X` no usado para fit del synthesizer).
- No promediar filas (tipos mixtos). Sí se permite *stacking de columnas*: copula para el bloque casi-lineal + CART para el resto, si un diagnóstico `corr_frobenius` de la copula gana y `prediction_ks` del CART gana. Eso es extra; no bloquear el entregable.

### 2.5 SMOTE-like — baseline débil, no generador de `X`

`src/synthesizers/smote_baseline.py` → `SMOTEClassBaseline`.

- Solo si `Y` es clase. Interpola el **bloque numérico** de la minoritaria (`imbalanced-learn.SMOTE` o reimplementación mínima).
- Columnas categóricas: copia del vecino más cercano, no interpolación.
- **No** se usa para tablas de regresión ni como candidato a “mejor `X'`”. Sirve para demostrar que oversampling de clase **no** reconstruye joints ni TSTR de features.

---

## 3. Qué no intentar ahora (y por qué)

| Familia | Por qué se descarta *ahora* |
| --- | --- |
| GAN / CTGAN / TVAE / tabular diffusion from scratch | Pesados, inestables, grid de hiperparámetros enorme. Este entorno y el repo actual son `sklearn`-first; `torch` no está garantizado (la rama de ataques lo añadió ad-hoc). Un CTGAN mediocre pierde contra CART en `d < 30` y nos come el presupuesto de debug. |
| Ruido i.i.d. / perturbación de valores únicos | Mata joints. Marginales sobreviven, `corr_frobenius` y TSTR colapsan. Es el **negative control**, no un método. |
| Privacy-only que destruye utilidad (DP-Laplace / DP-Gaussian en *todas* las celdas) | El objetivo de *esta* rama es fidelidad `X' ≈ X` en estructura y `X→Y`. DP global con `ε` chico garantiza el fallo de los gates. Privacidad se mide *después* (DCR / nearest-neighbor) como diagnóstico, no como objetivo de transformación. |
| Foundation models tabulares enormes (GReaT, TabPFN fine-tune, LLMs que escriben filas) | Fuera de stack, no reproducible en CPU, no aporta un `S.fit/sample` auditables. |
| Reutilizar `src/transforms.py` de las ramas hermanas (`typed_keyed`, VIB, rotaciones) | Esos mapas producen `Z ≠ X` (espacio distinto, a menudo no invertible). Aquí el esquema de `X'` debe ser el de `X`. Mezclar los dos problemas contamina métricas. |

Si más adelante se pide *privacy + fidelity*, el orden es: primero un `X'` que pase los gates de este plan; después subsample / noise *calibrado* sobre `X'`, no sobre `X`.

---

## 4. Análogos comerciales (alto nivel, sin pretender igualarlos)

Estos productos optimizan un frente **privacidad–utilidad–fidelidad** con modelos pesados, constraints de negocio y dashboards. Nosotros no vamos a igualar su quality bar ni su superficie de producto. Sí copiamos *qué miden*, no *cómo generan*.

| Producto | Qué optimizan (resumen) | Qué tomamos |
| --- | --- | --- |
| **Mostly AI** | Generadores tabulares (GAN/VAE/transformer), fidelidad univariada/bivariada, TSTR, privacy (DCR, membership) | El trio fidelidad + TSTR + DCR como *reporte*, no su generador |
| **Gretel** | ACTGAN / LSTM / transformers + DP opcional configurable | Idea de “privacy report” aparte del “utility report”; no activamos DP |
| **Synthesized** | Constraints de dominio + fidelidad estadística en tablas empresariales | Validar dtypes, rangos y reglas simples (`age ≥ 0`) post-sample |
| **SDV (DataCebo)** | `GaussianCopula`, `CTGAN`, `TVAE`, `CopulaGAN`; quality report + ML efficacy | Misma *taxonomía* de métricas. Implementamos copula y CART *propios*; no dependemos de `sdv` en runtime (opcional en `requirements-optional.txt` si se quiere un oráculo de comparación) |

Frase para el `REPORT.md`: *“Medimos lo que Mostly AI / Gretel / SDV llaman fidelity + ML efficacy; el generador es Sequential CART, no su stack.”*

---

## 5. Protocolo de evaluación

Un solo pipeline: `python run.py` (mismo patrón que las ramas hermanas). Todo vive en `src/experiment.py`.

### 5.1 Splits (sin leakage del sintetizador)

```
X_all  →  X_syn_fit (60%) | X_eval (40%)
X_eval →  X_tr (50% de eval) | X_te (50% de eval)   # 20% / 20% del total
```

- `S.fit(X_syn_fit, Y_syn_fit)` **nunca** ve `X_te`.
- `n' = n_syn_fit` (misma cardinalidad que el fit set, no inflar TSTR con más filas).
- Seeds `{0, 1}` mínimo; `{0,1,2}` en `--full`.

### 5.2 Modelos `M` (congelados, no tuneados en sintético)

Reusar la idea de `src/models.py` de las otras ramas, recortada:

| Task | Modelos |
| --- | --- |
| Regresión | `Ridge(α=1)`, `HistGradientBoostingRegressor` (defaults sklearn) |
| Clasificación | `LogisticRegression`, `HistGradientBoostingClassifier` |

Hiperparámetros **fijados**. Tunear `M` en `X'` falsea TSTR.

### 5.3 Protocolos ML

| Sigla | Train | Test | Qué responde |
| --- | --- | --- | --- |
| **TRTR** | `(X_tr, Y_tr)` | `X_te` | techo de utilidad real |
| **TSTR** | `(X', Y')` | `X_te` | *el* gate de producto |
| **TRTS** | `(X_tr, Y_tr)` | `X'` | ¿`X'` es in-distribution para un modelo real? |

Métricas de score:

- Regresión: `R²`, `MAE`.
- Clasificación: `ROC-AUC` (binaria / OVR), `Brier`, `accuracy` solo como auxiliar.

### 5.4 Fidelidad estadística (estructura de `X` y de `Y`)

Implementar en `src/metrics.py`:

| Métrica | Definición concreta | Agregación |
| --- | --- | --- |
| `ks_1d` | `scipy.stats.ks_2samp` por columna continua; `TV distance` por categórica | `mean_1m_ks = mean(1 − KS_i)` y `mean_1m_tv = mean(1 − TV_j)` |
| `wasserstein_1d` | `scipy.stats.wasserstein_distance` en columnas **estandarizadas** (z-score con media/std de `X_syn_fit`) | media; reportar crudo y 1 / (1 + W) |
| `corr_frobenius` | `‖corr_S(X) − corr_S(X')‖_F / ‖corr_S(X)‖_F` con Spearman (robusto a monótonas) | un escalar; también `corr_cosine` = coseno de `vech(corr)` |
| `ols_coef_cosine` | `Ridge` (o `LogisticRegression`) de `X_tr→Y_tr` vs `X'→Y'`; coseno de vectores de coeficientes alineados por nombre de columna | un escalar en `[-1, 1]` |
| `prediction_ks` | KS (o TV si clase) entre `M_TRTR(X_te)` y `M_TRTR(X')` — modelo **ya entrenado en real**, aplicado a ambos | un escalar; esto es el requisito “aplicar un modelo ya entrenado → `Y ≈ Y'` en distribución” |
| `fidelity_score` | ver §6 | un escalar en `[0, 1]` |

`fidelity_score` (definición cerrada, no la cambiamos a mitad de experimento):

```
fidelity_score =
    0.30 * mean_1m_ks_or_tv      # marginals (KS en cont., 1-TV en cats)
  + 0.25 * (1 - clip(corr_frobenius, 0, 1))
  + 0.20 * clip(ols_coef_cosine, 0, 1)
  + 0.25 * (1 - clip(prediction_ks, 0, 1))
```

Pesos sesgados a joints + relación `X→Y`, no a un QQ-plot bonito.

### 5.5 Negative control (obligatorio)

`IndependenceShuffleSynthesizer`: resample **independiente por columna** (marginales exactas, joints rotos).  
`UniqueJitterSynthesizer`: `X' = X + ε` columna a columna (`ε ~ N(0, 0.15 σ)` en continuas; flip 10% en categóricas) — valores “únicos” / privacy-theater.

Ambos **deben fallar** los gates de §6. Si no fallan, los gates están mal calibrados (demasiado flojos) y se endurecen *antes* de declarar victoria de CART.

### 5.6 Privacidad (diagnóstico, no gate de éxito)

- `dcr_ratio = median NN_dist(X' → X) / median NN_dist(X_te → X_tr)` (features estandarizadas).
- Contar copias exactas de filas (`exact_match_rate`).

Reportar en tabla. **No** son criterios de go/no-go de esta rama. Un `exact_match_rate > 0` se documenta como fallo de privacidad del CART (hojas de 1), se sube `min_samples_leaf`, y se re-evalúa fidelidad.

### 5.7 Datasets (reusar lo que el repo ya tocó)

Prioridad, en este orden, para no inventar un zoo:

1. `sklearn` `fetch_california_housing` — regresión, numérico, ya usado en ramas hermanas.
2. Tabla mixta de clasificación (el `banking_mixed_clf` de `cursor/utility-privacy-transforms-ccb2` si el loader se puede copiar; si no, `adult`/`census` de `openml` id 1590, o `sklearn` `fetch_openml("adult")`).
3. Una regresión mixta pequeña (p.ej. `bike sharing` o el `banking_mixed_reg` hermano).

`--quick`: solo California, 1 seed, CART + copula + negative control, `Ridge` only.  
`--full`: 3 datasets × 2–3 seeds × {CART, copula, mixture, SMOTE-si-clf, 2 negative} × 2 modelos.

---

## 6. Success bar (gates numéricos)

Un synthesizer **pasa** en un dataset/seed/modelo si cumple **todos** los gates de utilidad **y** el de fidelidad. El **método se declara viable** si pasa en ≥ 2 de 3 datasets, en la **mediana** de seeds, para **HGB y Ridge/LogReg**.

### 6.1 Gates de utilidad (el producto)

| Gate | Regresión | Clasificación |
| --- | --- | --- |
| **TSTR vs TRTR** | `R²_TRTR − R²_TSTR ≤ 0.08` **y** `R²_TSTR ≥ 0.0` | `AUC_TRTR − AUC_TSTR ≤ 0.05` **y** `AUC_TSTR ≥ 0.55` |
| **TRTS vs TRTR** | `R²_TRTR − R²_TRTS ≤ 0.10` | `AUC_TRTR − AUC_TRTS ≤ 0.06` |
| **No colapso** | `R²_TSTR ≥ 0.70 · R²_TRTR` si `R²_TRTR > 0.2` | `AUC_TSTR ≥ 0.70 · (AUC_TRTR − 0.5) + 0.5` |

`0.08` de `R²` es deliberadamente holgado para HGB en tablas ruidosas, pero suficientemente duro para que el shuffle independiente falle. Si TRTR `R²` es 0.60, TSTR debe ser ≥ 0.52.

### 6.2 Gates de fidelidad

| Gate | Umbral |
| --- | --- |
| **`fidelity_score`** | `≥ 0.70` |
| **`mean_1m_ks_or_tv`** | `≥ 0.75` (ninguna columna continua con `KS > 0.35` salvo flag explícita) |
| **`corr_frobenius`** | `≤ 0.30` |
| **`ols_coef_cosine`** | `≥ 0.85` |
| **`prediction_ks`** | `≤ 0.20` |

### 6.3 Gates del negative control (deben *fallar*)

`IndependenceShuffleSynthesizer` y `UniqueJitterSynthesizer` son un **pass del protocolo** solo si:

- `fidelity_score < 0.70` **o** `corr_frobenius > 0.30` **o** `ols_coef_cosine < 0.85`, **y**
- el gate TSTR de §6.1 **falla** en al menos un modelo (típicamente HGB).

Si el shuffle **pasa** TSTR, el target es demasiado aditivo/univariado: se cambia de dataset o se añade una interacción sintética de control (`Y := X1 X2 + ε`) como smoke test interno (`tests/test_protocol.py`).

### 6.4 Criterio de selección del default

1. Filtrar métodos que pasan §6.1–6.2 en ≥ 2 datasets (mediana de seeds).
2. Entre los que pasan, elegir el de mayor `fidelity_score` medio.  
   Expectativa: **CART**. Si gana la copula, se documenta y se deja CART como default *igual* (mejor degradación fuera de lo elíptico) y se reporta el empate.
3. Si CART falla un dataset por multimodalidad visible (diagnóstico: `prediction_ks > 0.20` y GMM de 2 componentes en `X | Y` con BIC mejor que 1), se activa mixture / ensemble y se re-corre solo ese dataset.

---

## 7. Entregables (alineados a este repo)

El repo en `main` está vacío (README + LICENSE + `.gitignore` Python). Las ramas hermanas ya fijaron el *shape* del proyecto: `run.py`, `src/`, `tests/`, `results/`, `REPORT.md`, `requirements.txt`. Esta rama replica ese shape, **sin** importar `transforms.py` / attackers.

```
Privacy-Preserving-Transformations/
├── PLAN.md                 # este documento
├── README.md               # objetivo X→X', cómo correr, apuesta CART
├── REPORT.md               # resultados + gates pass/fail (escrito a mano / generado)
├── RESEARCH.md             # 1–2 págs: synthpop vs copula vs mixture; por qué no GAN
├── requirements.txt        # numpy pandas scipy scikit-learn matplotlib pytest
├── run.py                  # CLI --quick | --full
├── src/
│   ├── __init__.py
│   ├── config.py           # seeds, splits, knobs, gate thresholds
│   ├── datasets.py         # loaders (california, mixed clf, mixed reg)
│   ├── synthesizers/
│   │   ├── base.py         # interface fit(X,y) / sample(n) → DataFrame mismo schema
│   │   ├── sequential_cart.py
│   │   ├── gaussian_copula.py
│   │   ├── conditional_mixture.py
│   │   ├── smote_baseline.py
│   │   ├── negative.py     # IndependenceShuffle, UniqueJitter
│   │   └── ensemble.py     # opcional
│   ├── models.py           # Ridge, LogReg, HGB (frozen)
│   ├── metrics.py          # KS, W1, corr F, ols cosine, prediction KS, fidelity_score
│   ├── experiment.py       # splits, TRTR/TSTR/TRTS, gates, JSON por run
│   ├── plots.py            # 4–6 figuras fijas
│   └── report.py           # markdown tables → results/GENERATED_REPORT.md
├── tests/
│   ├── test_metrics.py     # fidelity_score bounds; cosine=1 on identical
│   ├── test_synthesizers.py
│   └── test_protocol.py    # negative control fails gates on toy interaction data
└── results/
    ├── figures/
    │   ├── 01_marginal_overlay.png
    │   ├── 02_corr_heatmaps.png
    │   ├── 03_tstr_vs_trtr.png
    │   ├── 04_prediction_ks.png
    │   ├── 05_fidelity_bars.png
    │   └── 06_ols_coefficients.png
    └── tables/
        ├── utility_trtr_tstr_trts.csv
        ├── fidelity_breakdown.csv
        └── gates_pass_fail.csv
```

**Contrato de `run.py`:**

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python run.py --quick    # smoke: california, 1 seed, CART+copula+shuffle
python run.py            # full grid
```

Escribe `results/run_meta.json`, CSVs y PNGs. `REPORT.md` de nivel científico (no pisarlo en cada run; el volcado automático va a `results/GENERATED_REPORT.md`, igual que la rama `utility-privacy-transforms`).

**Tests mínimos que bloquean merge:**

1. `SequentialCARTSynthesizer` en `Y = X0 + 1.5 X1 + ε` (`n=800, d=4`) → `ols_coef_cosine ≥ 0.90` y TSTR `R²` within `0.08` de TRTR.
2. `IndependenceShuffleSynthesizer` en `Y = X0 * X1 + ε` → falla `fidelity_score ≥ 0.70` **o** falla TSTR.
3. Schema: `X'.columns == X.columns`, dtypes compatibles, `n' == n` pedido.
4. `Y` no aparece como feature padre de columnas de `X` (assert en CART).

---

## 8. Orden de implementación (opinión)

1. `src/metrics.py` + `tests/test_metrics.py` + gates en `src/config.py`. Sin métricas no hay método.
2. `IndependenceShuffleSynthesizer` + `datasets.california` + TRTR/TSTR en `experiment.py`. Confirmar que el **negative control falla**. Si no, parar.
3. `GaussianCopulaSynthesizer`. Debe pasar California (casi elíptico) o quedar a ~0.05 de `R²`. Baseline serio.
4. `SequentialCARTSynthesizer` + orden `mi_y`. Este es el default que se documenta en README.
5. Dataset mixto de clasificación + `SMOTEClassBaseline` (para que se vea flojo).
6. `ConditionalMixtureSynthesizer` solo si CART pierde `prediction_ks` o hay modos visibles en residuales.
7. Figuras + `REPORT.md` con tabla única `gates_pass_fail.csv` y la frase de análogos comerciales.

No paralelizar 3–6 hasta que el paso 2 (negative control) esté en verde.

---

## 9. Riesgos concretos

- **Column order de CART:** un orden malo rompe cadenas de dependencia. Mitigación: dos órdenes en el grid; no más.
- **Hojas de 1 fila:** copia casi exacta → DCR/exact match feos. `min_samples_leaf ≥ 5` (mejor 20) es no negociable.
- **Categóricas de alta cardinalidad:** CART las fragmenta. Agrupar niveles `< 1%` en `__OTHER__` *antes* de fit.
- **Leakage `Y → X`:** si se permite `Y` como padre de features, TSTR se infla y el producto miente. Interfaz: `fit(X, y)` trata `y` solo como última columna a sintetizar.
- **Confundir esta rama con `Z = f(X)`:** el README debe decir en las primeras 10 líneas que `X'` vive en el mismo schema y que no hay clave de dueño.

---

## 10. Decisión (para no diluir)

| Rol | Método |
| --- | --- |
| **Default de producto** | `SequentialCARTSynthesizer` (`min_samples_leaf=20`, `column_order=mi_y`, `leaf_jitter=0.2`) |
| Baseline estructural | `GaussianCopulaSynthesizer` |
| Reserva por segmentos | `ConditionalMixtureSynthesizer` (activar si CART falla `prediction_ks`) |
| Ensemble | solo si ambos anteriores ganan en métricas *distintas* |
| Weak baseline de clase | `SMOTEClassBaseline` |
| Negative controls | `IndependenceShuffle`, `UniqueJitter` |

**Go:** CART (o CART+mixture) pasa §6.1–6.2 en ≥ 2 datasets y los negative controls fallan.  
**No-go:** solo la copula pasa, o CART pasa fidelidad 1-D pero TSTR se cae > `0.08` de `R²` — entonces el fallo es la relación `X→Y`, no las marginales, y se itera el orden / la hoja / el mixture, **no** se empieza un GAN.
