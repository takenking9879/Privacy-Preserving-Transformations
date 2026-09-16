# Plan GENERAL: un sintetizador que sobreviva a más de un DGP

**Repo:** `Privacy-Preserving-Transformations`  
**Audiencia:** Jorge (software engineer / data scientist)  
**Documento hermano:** `PLAN.md` (protocolo y apuesta sobre el DGP de crédito único). Este archivo **no** lo reemplaza: fija el listón para llamar al método *general*.

**Apuesta primaria:** `AutoSynthesizer` — bake-off de familias en un hold-out de síntesis, un ganador por dataset.  
**Default fuerte si el bake-off es lento:** `ForestSequential`.  
**No apostamos** a un `CTGAN` / GAN / TVAE / diffusion *from scratch*.

---

## 1. Por qué un solo DGP no basta

El DGP actual (`src.dgp.generate_original`, registrado como `credit`) ya es *difícil*: tipos mixtos, latentes `Z`/`H`, cluster ~4 %, interacciones `credit_util × is_premium` y `log(income) × usage`, umbral en `risk_score > 68`, heterocedasticidad, MAR, conteos Poisson/NegBin. `CARTSequentialSynthesizer` lo pasó (gap TSTR R² ≈ 0.054, `fidelity_score` ≈ 0.94). Eso **no** prueba un sintetizador general.

Razones concretas:

1. **Victoria local ≠ transferencia.** Un método puede memorizar los *motivos* de crédito (cluster `H`, hockey-stick, MAR en `income`/`engagement`) y fallar en un `Y` puramente interaccional, en colas de Pareto, o en un panel entidad-tiempo.
2. **Cada familia tiene un DGP “amigo”.** `GaussianCopulaSynthesizer` pierde TSTR en crédito (gap RF ≈ 0.36) pero es el techo correcto en tablas casi elípticas (`sparse_linear`). `ConditionalMixtureSynthesizer` pierde el RF TSTR en crédito y es el instrumento correcto en `multimodal`. Elegir el default mirando *un* leaderboard es sesgo de diseño.
3. **El gate actual no vetó interacciones.** Crédito *contiene* dos productos en `E[y | X]`, pero también muchos efectos aditivos y un `risk_score` casi-lineal. Un copula + residual débil puede “pasar cerca” sin recuperar `X_i X_j`. Sin un DGP cuyo `Y` *sea* la interacción, se puede declarar general un método que no lo es.
4. **Hard-coding de columnas.** `CARTSequentialSynthesizer` y `HybridSynthesizer` aún fuerzan `region` / `segment` / `credit_util`. Un único schema esconde ese acoplamiento. Diez schemas distintos lo hacen un bug, no una constante.
5. **Métricas univariadas mienten entre dominios.** KS/TV altos en crédito no predicen TSTR en `imbalanced` (AUC de minoritaria) ni en `heavytail` (cuantiles, no medias) ni en `panel` (correlación intra-entidad).
6. **Negative control calibrado a un target.** Si `IndependenceShuffleSynthesizer` falla TSTR en crédito, los gates están bien *ahí*. En un `Y` aditivo univariado el shuffle *pasaría*. La suite incluye `interactions` precisamente para que el protocolo no se ablande.

Implicación operativa: **no se llama “general” a un método que solo gana en `credit`.** El título se gana en la suite de §2 bajo los gates de §4.

```
credit  --(PLAN.md)-->  CART gana, copula/mixture/hybrid fallan TSTR
suite   --(este plan)-->  AutoSynthesizer elige familia por dataset
                          ForestSequential es el default si no hay tiempo
                          fail(interactions) ⇒ no es general
```

---

## 2. La suite (10 DGPs) y qué estresa cada uno

Contrato compartido (`src.datasets.base.DatasetSpec`): `generate(n, seed) → DataFrame` con `y` continuo y `y_class` opcional. El sintetizador **no** puede hard-codear nombres de crédito. Registro: `src.datasets.registry` (`credit`, `healthcare`, `retail`, `insurance`, `interactions`, `multimodal`, `imbalanced`, `heavytail`, `panel`, `sparse_linear`).

Un dataset **pasa** si el método cumple el gate TSTR de §4.1 en la mediana de seeds. La suite **pasa** solo si se cumplen **los tres** criterios de §4.2.

| `name` | Dominio (intención) | Qué debe romper |
| --- | --- | --- |
| `credit` | Tabla mixta de riesgo (DGP legado, `src.dgp`) | Latentes `Z`/`H`, cluster conjunto ~4 %, MAR, umbral, heterocedasticidad, conteos, tipos mixtos. Regresión de continuidad: si se pierde crédito, el “general” ni siquiera conserva el win de `PLAN.md`. |
| `healthcare` | Clínico mixto (labs acotados, códigos, comorbilidad) | Categóricas de **alta cardinalidad** (ICD-like), outcomes raros, labs bounded/skew, interacciones tratamiento × lab. Estresa hojas CART fragmentadas y one-hot explosivo en copula. |
| `retail` | Ticket / spend (ceros, categorías, power-law de SKU) | **Zero-inflation**, spend sesgado, conteos de ítems, muchos binarios sparse de cesta. Estresa copulas gaussianas (masa en cero) y CART que interpola ceros como continuo. |
| `insurance` | Frecuencia / severidad (rating factors) | `Y` tipo **Tweedie / two-part** (P(claim)=0 alta + cola de severidad), factores categóricos de tarifa. Estresa un solo modelo de `y` continuo y TSTR en la cola de siniestros. |
| `interactions` | DGP sintético: `Y` *es* producto / cruce | `Y := X1 X2 + X3 X4 + ε` (más un cruce categórico × continuo). Marginales y Spearman pueden verse bien; **TSTR de un RF real colapsa** si `X'` no trae el cruce. **Veto de generalidad** (§4.2). Copula y ridge-solo deben fallar. |
| `multimodal` | Mezcla de 3–4 componentes en `X \| segmento` | Modos bien separados. `GaussianCopulaSynthesizer` (elíptica, unimodal) debe fallar joints; `CARTSequentialSynthesizer` puede *suavizar* modos si las hojas mezclan clusters; `ConditionalMixtureSynthesizer` es el candidato natural. |
| `imbalanced` | Clase rara (~2–5 % en `y_class`) | TSTR **AUC / Brier de minoritaria**, no accuracy. SMOTE-like reconstruye el margen de clase y **rompe** joints de `X`. Un método general no puede “ganar” oversampleando `Y` e ignorando `P(X)`. |
| `heavytail` | Pareto / Student-t en `X` y en `y` | Momentos y copula gaussiana se rompen; Wasserstein y cuantiles 95/99 importan más que KS central. TSTR en el cuerpo puede pasar y **fallar en la cola** (el caso insurance/retail extremo, sin ceros). |
| `panel` | Entidad × tiempo (2–8 olas) | Correlación **intra-id**, efectos fijos, auto-correlación. Un sintetizador i.i.d. por fila destruye el panel: TRTR de un modelo con `entity`/`lag` se mantiene, TSTR se cae. Prohibido romper `id` al barajar filas en el split. |
| `sparse_linear` | `d` alto, `s ≪ d` señales, resto ruido | `Y = X_S β + ε`. Estresa sequential trees que **gastan splits en ruido** y un `column_order` que antepone basura. Copula / ridge deberían ser competitivos; si `ForestSequential` pierde *solo* aquí, no es veto (sí lo es perder `interactions`). |

### 2.1 Por qué estas diez (y no otras)

- Cuatro **dominios** (`credit`, `healthcare`, `retail`, `insurance`): tipos mixtos reales, no un zoo de UCI.
- Cuatro **geometrías** (`interactions`, `multimodal`, `heavytail`, `sparse_linear`): fallos que un único schema de crédito no aísla.
- Dos **protocolos** (`imbalanced`, `panel`): el target o la unidad muestral cambian; las métricas 1-D de `PLAN.md` no bastan.

`--quick`: `credit` + `interactions` + `sparse_linear` (1 seed). Si `interactions` ya falla, no se gasta el grid de 10.  
`--full`: 10 × seeds `{0,1}` ( `{0,1,2}` si hay presupuesto).

---

## 3. Apuesta de método GENERAL

### 3.1 Primario — `AutoSynthesizer` (bake-off en hold-out)

Implementar `src/synthesizers/auto.py` → clase `AutoSynthesizer`.

No es un generador nuevo: es un **selector**. Por dataset:

```
X_syn_fit  →  X_bake_fit (80%) | X_bake_ho (20%)
para cada candidato C en CANDIDATES:
    C.fit(X_bake_fit)
    X'_C = C.sample(n = |X_bake_fit|)
    score(C) = bakeoff_score(X'_C, X_bake_ho)   # ver abajo
C* = argmax score(C)
C*.fit(X_syn_fit)          # re-fit en todo el syn-fit
X' = C*.sample(n = |X_syn_fit|)
```

`CANDIDATES` (nombres estables, los que ya existen o el default nuevo):

| Candidato | Rol en el bake-off |
| --- | --- |
| `ForestSequential` | Default fuerte; interacciones + tipos mixtos, más estable que un árbol |
| `CARTSequentialSynthesizer` | Control synthpop; barato; a veces gana en `d` chico |
| `GaussianCopulaSynthesizer` | Techo lineal/elíptico (`sparse_linear`) |
| `ConditionalMixtureSynthesizer` | Techo de modos (`multimodal`) |
| `HybridSynthesizer` | Copula en `X` + RF residual en `Y`; útil si joints de `X` son casi gaussianos y el mapa `X→Y` no lo es |

**`bakeoff_score` (definición cerrada):**

```
bakeoff_score =
    0.60 * (1 - clip(tstr_gap_holdout, 0, 1))   # TSTR vs TRTR en X_bake_ho
  + 0.40 * fidelity_score(X_bake_fit, X'_C)     # misma fórmula que PLAN.md §5.4
```

El hold-out de bake-off **no** es `X_te` del protocolo TSTR de producto (`PLAN.md` §5.1). El sintetizador ganador nunca ve `X_te`.

**Por qué esta es la apuesta GENERAL:** ningún candidato gana las diez geometrías. El producto *general* es el que **elige** la familia correcta con un presupuesto fijo, no el que finge que CART es universal. El bake-off es auditable: se serializa `winner_name` por dataset en `results/auto_choice.json`.

**Presupuesto de bake-off:** cada candidato un fit+sample; sin grid interno. Si un candidato lanza o tarda más de un tope (`--bakeoff-seconds`, default 90 s por familia), se descarta *ese* candidato, no se aborta el Auto.

### 3.2 Default fuerte — `ForestSequential`

Implementar `src/synthesizers/forest_sequential.py` → clase `ForestSequential`.

Mismo esqueleto que `CARTSequentialSynthesizer` (orden de visita, primera columna empírica, `Y` / `y_class` al final, sample de hoja / residual, nunca la media sola):

1. Orden: categóricas por cardinalidad creciente; numéricas por `|Spearman|` medio descendente; targets últimos. **Sin** `FORCE_CATEGORICAL` con nombres de crédito — solo dtypes + cardinalidad.
2. Condicional `j`: `RandomForestClassifier` / `RandomForestRegressor` (`sklearn`) de `X_{π<j}` → `X_{πj}`.
3. Sample: bajar cada árbol, sortear un valor de la **unión de hojas** (o residual empírico del árbol sorteado). `n_estimators=40`, `max_depth=8`, `min_samples_leaf=max(10, floor(0.01 n))`, `max_features="sqrt"`.
4. Si `d > 25` (caso `sparse_linear` / `panel` wide): `max_features=min(8, d)` y `n_estimators=25` para no explotar CPU.

**Por qué, si Auto es lento, este es el default:**

- Un solo `DecisionTree` (CART) tiene varianza alta de hoja; el bosque promedia splits y conserva interacciones de profundidad media.
- Nativo a tipos mixtos, CPU-only, mismo stack que el híbrido (que ya usa RF para `Y`).
- Literatura synthpop + “sequential forest” (oficinas estadísticas / variantes de CART-bagging) es el upgrade natural cuando CART gana crédito pero se pone nervioso en `healthcare` / `retail`.
- Falla de forma interpretable: igual que CART (orden, hojas chicas, leakage `Y→X`).

**Regla de fallback (obligatoria en código):**

```
if wall_clock_auto > AUTO_BUDGET or n_candidates_ok < 2:
    use ForestSequential on full X_syn_fit
    record reason = "auto_timeout" | "too_few_candidates"
```

`AUTO_BUDGET` default: 8 min por dataset en `--full`, 90 s en `--quick`. Documentar el fallback en `REPORT.md`; no venderlo como bake-off.

### 3.3 Qué sigue en el banco (no son la apuesta)

- `CARTSequentialSynthesizer` — baseline synthpop; debe seguir en el bake-off.
- `GaussianCopulaSynthesizer` — control lineal.
- `ConditionalMixtureSynthesizer` — reserva de modos.
- `HybridSynthesizer` — copula `X` + RF `Y`.
- `IndependenceShuffleSynthesizer` — negative control; **debe fallar** `interactions` y al menos 6/10 datasets.

No se añade `SMOTEClassBaseline` como candidato de `AutoSynthesizer` (solo diagnóstico en `imbalanced`).

---

## 4. Gates GENERAL

Los umbrales de `PLAN.md` §6 (`gap ≤ 0.08`, `fidelity_score ≥ 0.70`) siguen valiendo para el **reporte por dataset** de `credit`. El sello **GENERAL** usa umbrales *de suite*, más holgados por dataset (la geometría es más dura) y más estrictos en cobertura.

### 4.1 Gate por dataset

Un método **pasa** un dataset (mediana de seeds) si:

| Gate | Regresión | Clasificación (`y_class` presente) |
| --- | --- | --- |
| **TSTR vs TRTR** | `R²_TRTR − R²_TSTR ≤ 0.10` **y** `R²_TSTR ≥ 0.0` | `AUC_TRTR − AUC_TSTR ≤ 0.06` **y** `AUC_TSTR ≥ 0.55` |
| **No colapso** | `R²_TSTR ≥ 0.65 · R²_TRTR` si `R²_TRTR > 0.2` | `AUC_TSTR ≥ 0.65 · (AUC_TRTR − 0.5) + 0.5` |
| **Fidelidad** | `fidelity_score ≥ 0.65` (0.70 sigue siendo la meta blanda en `credit`) | igual |

Se reportan RF y lineal (mismos families congelados que `PLAN.md` §5.2). El gap que entra al gate es el **mejor** de los dos (`min(gap_RF, gap_linear)`), igual que `reports/RESULTS.md`.

`0.10` (no `0.08`) es deliberado: `heavytail`, `panel` e `imbalanced` son más ruidosos que crédito. Sigue siendo suficiente para que el shuffle falle en `interactions`.

### 4.2 Gates de suite (el sello GENERAL)

Los tres son **conjuntivos**. Fallar uno ⇒ no se llama general.

1. **Mediana de gaps TSTR ≤ 0.10** a través de los 10 datasets (un número por dataset: el best-gap de §4.1; la mediana de esos diez ≤ 0.10).
2. **Pasar al menos 7/10** datasets bajo §4.1.
3. **Veto `interactions`:** el método **debe pasar** `interactions`. Si falla `interactions` y gana 9/10, **no es general**. Frase de `REPORT.md`: *“sin interactions no hay claim de generalidad — solo un ensemble de tablas casi aditivas.”*

Corolarios:

- Perder solo `sparse_linear` o solo `panel` es aceptable (se documenta el fallo de modo).
- Perder `credit` **y** `interactions` es no-go inmediato (ni sequimos el legado ni el veto).
- El negative control debe fallar `interactions` **y** tener mediana de gaps **> 0.15**. Si el shuffle pasa `interactions`, el DGP está mal especificado (se añade otro cruce o se baja el ruido), no se relaja el veto.

### 4.3 Criterio de selección del default de producto

1. Si `AutoSynthesizer` cumple §4.2 → **default de producto = `AutoSynthesizer`**, con `ForestSequential` como fallback de timeout.
2. Si Auto no cumple pero `ForestSequential` solo sí cumple §4.2 → default = `ForestSequential`, Auto se queda como oráculo de diagnóstico.
3. Si ninguno cumple §4.2 pero `ForestSequential` pasa `interactions` y ≥ 6/10 → se itera knobs de bosque / orden (`min_samples_leaf`, `n_estimators`), **no** se abre un GAN.
4. Si `interactions` no lo pasa nadie del banco → se revisa el DGP de `interactions` (¿ruido excesivo? ¿`n` chico?) *antes* de añadir familias.

---

## 5. Qué no intentar (todavía)

| Familia | Por qué se descarta *en GENERAL* |
| --- | --- |
| **`CTGAN` / GAN / TVAE / tabular diffusion from scratch** | Sigue siendo el no-go de `PLAN.md` §3, *más caro* ahora: 10 DGPs × 2–3 seeds × inestabilidad de adversarial training. Este repo es `sklearn`-first; `torch` no está garantizado. Un CTGAN mediocre pierde contra `ForestSequential` en `d < 30` y multiplica el presupuesto de debug por la suite. **No** se reimplementa CTGAN “para ser general”. Si más adelante se pide un oráculo, es un binario opcional (`requirements-optional.txt` + SDV), nunca el default. |
| Grid grande *dentro* de cada candidato | El bake-off de Auto *es* el presupuesto. Tunear `max_depth` × `n_estimators` × 10 datasets es el mismo anti-patrón que el grid de GAN. |
| Foundation models tabulares (GReaT, LLM rows, TabPFN fine-tune) | Fuera de stack, no CPU-reproducible, no `fit`/`sample` auditable. |
| DP-Laplace / ruido i.i.d. en todas las celdas | Objetivo de esta rama = fidelidad + TSTR. El ruido es negative control, no método. |
| Reusar `src/transforms.py` de ramas `Z = f(X)` | Schema distinto. Contamina TSTR. |
| Declarar generalidad con `--quick` | Quick es smoke (`credit`+`interactions`+`sparse_linear`). El sello GENERAL exige `--full` y el veto. |

Si más adelante se pide *privacy + generality*, el orden no cambia: primero un `X'` que pase §4.2; después subsample / ruido *calibrado* sobre `X'`.

---

## 6. Protocolo (delta respecto a `PLAN.md`)

Se reutiliza `src/evaluate.py` / splits 60/20/20, modelos congelados, `fidelity_score`, TRTR/TSTR/TRTS, DCR como diagnóstico. Cambios:

1. **Loop de datasets** vía `src.datasets.registry.get(name)`. Nada de `from src.dgp import FEATURE_COLS` dentro de sintetizadores.
2. **`AutoSynthesizer.fit`** consume solo `X_syn_fit`; el bake-off interno parte ese bloque, nunca `X_te`.
3. Artefacto extra: `results/auto_choice.json` (`dataset → winner_name, scores, fallback_reason`).
4. Tabla de suite en `reports/RESULTS_GENERAL.md`: 10 filas × métodos, columna `pass`, mediana de gaps, flag `interactions_veto`.
5. `--quick` aborta con no-go si `interactions` falla; no se finge cobertura 7/10.

Tests que bloquean el claim GENERAL (además de los de `PLAN.md` §7):

1. `ForestSequential` en `Y = X0 * X1 + ε` (`n=800, d=4`) → TSTR R² within `0.10` de TRTR; `IndependenceShuffleSynthesizer` falla.
2. `AutoSynthesizer` en un toy bimodal elige `ConditionalMixtureSynthesizer` **o** `ForestSequential`, nunca el shuffle.
3. Ningún sintetizador de producto importa nombres `region`/`segment`/`credit_util` como constantes (grep de test).
4. Split de `panel` agrupa por `id` (no hay el mismo `id` en `X_syn_fit` y `X_te`).

---

## 7. Orden de implementación (opinión)

1. Contratos `DatasetSpec` + 10 módulos en `src/datasets/` (aunque sea `n` chico). Sin suite no hay gate.
2. DGP `interactions` + test del shuffle que **falla**. Si no falla, parar.
3. `ForestSequential` (sin forzar columnas de crédito) + smoke `--quick`.
4. `AutoSynthesizer` con candidatos ya registrados (`cart`, `copula`, `mixture`, `hybrid`, `forest`).
5. `--full`, tabla §4.2, veto.
6. Solo entonces: knobs de bosque en los datasets que fallen *sin* ser `interactions`.

No paralelizar 4–5 hasta que el paso 2 (veto) esté en verde.

---

## 8. Riesgos concretos

- **Bake-off overfit al hold-out de síntesis.** Mitigación: 20 % bake-ho, un solo score, re-fit en todo `X_syn_fit`; `X_te` intacto.
- **Timeout → siempre Forest.** Si Auto cae a fallback en > 4/10, el claim es “`ForestSequential` general”, no “Auto elige”. Se baja `n_estimators` / se saca `hybrid` del banco antes de mentir.
- **`panel` mal partido.** Leakage de entidad infla TSTR. Split por `id` es no negociable.
- **`imbalanced` con AUC inestable.** Reportar Brier y gap en la minoritaria; no declarar pass con accuracy.
- **Confundir mediana de gaps con media.** Un desastre en `heavytail` (gap 0.40) y nueve pases holgados puede tener mediana ≤ 0.10. Por eso existen el 7/10 **y** el veto, no solo la mediana.

---

## 9. Decisión (para no diluir)

| Rol | Método |
| --- | --- |
| **Default GENERAL** | `AutoSynthesizer` (bake-off en hold-out) |
| **Default si Auto es lento** | `ForestSequential` (`n_estimators=40`, `max_depth=8`, `min_samples_leaf≈0.01 n`) |
| Baseline estructural | `GaussianCopulaSynthesizer` |
| Reserva de modos | `ConditionalMixtureSynthesizer` |
| Control synthpop | `CARTSequentialSynthesizer` |
| Copula `X` + RF `Y` | `HybridSynthesizer` |
| Negative control | `IndependenceShuffleSynthesizer` |

**Go GENERAL:** mediana de TSTR gaps ≤ 0.10, ≥ 7/10 datasets en §4.1, **y** pass en `interactions`.  
**No-go:** fallar `interactions` (aunque el resto brille); o solo copula/CART-crédito pasan; o se “arregla” la suite empezando un `CTGAN` from scratch.

Frase para `REPORT.md`: *“Medimos generality como TSTR transferido a diez geometrías, no como un win en crédito. El selector es `AutoSynthesizer`; el generador que corre si no hay tiempo es `ForestSequential`. Sin `interactions`, no usamos la palabra general.”*
