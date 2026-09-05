# Guía del disfraz (cápsula de inferencia)

## Primero: qué se disfraza

| Pieza | ¿Se disfraza? | ¿Se puede deshacer? | Para qué |
| --- | --- | --- | --- |
| Las columnas de entrada (`X`) | Sí, se **comprimen** a un resumen | **No**. El resumen no está hecho para volver atrás | Entrenar e inferir |
| El objetivo (`y`: impago, precio, …) | Sí, se **estira/desplaza** (o se permutan las etiquetas) | **Sí**, solo en tu lado | Que el entrenador no vea unidades ni el significado del objetivo, y tú recuperes la predicción normal |

El entrenador recibe solo:

- un resumen anónimo `Z` (columnas `c000`, `c001`, …)
- un objetivo disfrazado `ỹ`

Tú, con la clave de dueño, conviertes la predicción otra vez a la escala de siempre:

`predicción_real = g⁻¹( modelo(Z) )`

---

## Fórmula

Clave privada del dueño (no se publica):

```
K_dueño = { resumidor E, rotación R, cuantizador Q, mapa del objetivo g, clave HMAC }
```

Disfraz de una fila:

```
U = tipeado_con_clave(X)          # números reescalados, categorías recodificadas, textos a cubos
H = E(U)                           # resumen que solo sirve para predecir y
Z = R · Q(H)                       # se redondea y se mezcla
ỹ = g(y)                           # a·y + b   (o permutación de clases)
```

Inferencia de una fila nueva:

```
ŷ = g⁻¹(  M(  R · Q( E( tipeado_con_clave(X_nueva) ) )  )  )
```

Sobre el paquete que se envía, opcional:

```
paquete = Cifrar_con_clave_pública_del_entrenador( Z, ỹ )
```

Solo quien tenga la **clave privada del entrenador** abre el archivo. Eso no para al entrenador: él sí ve `Z`.

No existe una “clave pública para disfrazar”. Si cualquiera pudiera crear `Z` a partir de `X`, fabricaría pares a voluntad y rompería el método como el atacante A.

---

## Cómo disfrazar el dataset (pasos)

1. **En tu máquina**, con las etiquetas reales, quita identificadores (id de cliente, correo único).
2. Ajusta el resumidor `E` para que `H` sirva para predecir `y` y olvide el resto.
3. Redondea ese resumen (`Q`) y mézclalo con una rotación secreta (`R`).
4. Disfraza el objetivo con `g` (guardar `g⁻¹` solo tú).
5. Genera un par de claves del **entrenador**. Cifra `(Z, ỹ)` con su clave pública.
6. Envíale el paquete. Él entrena `M: Z → ỹ`.
7. Para usar el modelo: tú (o un servicio que tenga `K_dueño`) resume la fila nueva, el modelo predice, tú aplicas `g⁻¹`.

Código: transform `capsule` en `src/transforms.py`. Cifrado: `src/keys.py`.

Valores por defecto que mantuvieron la efectividad del modelo (≥ 90 % respecto a entrenar en crudo):

- tamaño del resumen ≈ 38 % de las columnas de entrada
- redondeo a 12 niveles
- el resumidor además se entrena para que reconstruir `X` desde `H` sea difícil

Si quieres apretar más al atacante A y puedes ceder ~2–3 % de acierto, usa `capsule_tight`.

---

## Qué lo rompe más, qué lo rompe menos

| Situación | ¿Lo rompe? | Qué consigue |
| --- | --- | --- |
| Solo ve `Z` y sabe que “es un banco” (atacante B) | **Casi no** | No nombra columnas, no reconstruye valores. Adivinar “esta fila se usó para entrenar” es como lanzar una moneda |
| Roba el archivo cifrado **sin** la clave privada del entrenador | **No** | Ve basura |
| Es el entrenador (o robó su clave) y **no** tiene filas originales | **Casi no** | Tiene el resumen, no la tabla real ni los nombres |
| Es el entrenador **y** tiene **2 filas** originales emparejadas con su resumen | **Sí, en lo importante** | Recupera bastante bien las columnas que sirven para predecir (sueldo, riesgo, …). No reconstruye la tabla entera |
| Tiene **25–200** pares | **La tabla entera sigue resistiendo** | El acierto global al reconstruir todas las columnas se queda bajo (~0.08). Las columnas predictivas ya las tenía desde el par 2 |
| Roba `K_dueño` (el resumidor) | **Sí, del todo** | Puede crear resúmenes de quien quiera y deshacer `g` |
| Publicamos el resumidor como “clave pública” | **Sí, nos suicidamos** | El atacante fabrica pares ilimitados |

### Cuántas filas tiene que saber

Medido en las tablas de esta repo (cápsula vs el VIB anterior):

| Meta del atacante | Filas emparejadas que necesita | Cápsula | VIB suelto |
| --- | --- | --- | --- |
| Reconstruir **toda** la tabla con calidad alta | más de 200; **no lo logró** | resumen global ~0.08 en banco | ~0.30–0.36 |
| Recuperar **una columna que predice** | **2** ya bastan | ~0.75–0.88 | ~0.78–0.89 |
| Poner nombres a las columnas sin pares | 0 (solo `Z`) | no puede | no puede |

“Romper” aquí no significa desencriptar. Significa: “ya puedo adivinar los datos que importan”.

### Supuestos mínimos para romper lo importante

Hace falta **las dos** a la vez:

1. Ver `Z` en claro (ser el entrenador, o tener su clave privada del sobre).
2. Tener al menos **un par de filas** originales ↔ resumen. Con **2** ya es práctico.

### Qué no se puede romper con solo eso

Con esos dos supuestos **no** logra:

- volver atrás el resumen como si fuera una foto (está redondeado y comprimido a propósito)
- reconstruir correos, identificadores ni textos (se fueron a cubos irreversibles)
- aprender `g⁻¹` del objetivo si no ve `y` original (el objetivo también va disfrazado; si además le das pares de `y`, ahí sí)
- disfrazar filas nuevas sin `K_dueño`

Con **solo** el supuesto “tengo `Z` y el contexto” (atacante B) **no** logra ni nombres ni valores.

---

## Claves pública / privada: qué sí y qué no

Sí implementamos un sobre: el entrenador tiene un par RSA. El dueño cifra el envío con la pública. Un ladrón del USB no lee la tabla.

No usamos clave pública para el disfraz de `X`. Eso convertiría el método en “cualquiera fabrica pares” y le haría la vida fácil al atacante A.

La clave que de verdad protege el disfraz es **simétrica y tuya**: el resumidor, la rotación, `g` y el HMAC. Trátala como una llave de caja fuerte, no como un certificado público.

---

## Efectividad del modelo (cápsula)

Respecto a entrenar en los datos crudos, el modelo de árboles retuvo:

- vivienda: **91 %**
- banco (cantidad): **98 %**
- banco (sí/no): **96 %**

El modelo lineal se mantiene o mejora un poco, porque el resumen ya viene preparado para predecir.

---

## Receta corta

```
tú:     X,y  --K_dueño-->  (Z, ỹ)  --clave pública entrenador-->  paquete
él:     abre con su clave privada, entrena M(Z)=ỹ
tú:     ŷ_real = g⁻¹( M( resumen(X_nueva) ) )
```

Asume que el entrenador es curioso. No asumas que es honesto si puede conseguir dos filas reales de tus clientes.
