# Guia de Ajuste: PRE_CLEAN, FIX_TECHO e INTERPOL

## Objetivo
Este documento explica, con enfoque practico, como ajustar los parametros de postprocesamiento en la app de inferencia:

- `PRE_CLEAN_SURFACE`: limpieza previa de ruido y micro-huecos.
- `FIX_TECHO`: corrige falsos negativos en maquinaria "hueca" o fragmentada.
- `INTERPOL`: genera DTM aplanando maquinaria hacia cota de suelo.

Referencia de implementacion:
- `app_inference/core/postprocess.py`
- `app_inference/ui/app.py`

---

## Mapa Rapido del Pipeline
1. Inferencia clasifica puntos en:
- Clase `1`: maquinaria
- Clase `2`: suelo

2. `PRE_CLEAN_SURFACE` (opcional):
- Limpia ruido de clase 1 (islas pequeñas / baja coherencia local).
- Rellena micro-huecos de clase 2 rodeados por clase 1.

3. `FIX_TECHO` (opcional):
- Encuentra clusters de maquinaria (DBSCAN).
- Busca puntos suelo "dentro" del volumen esperable de cada objeto.
- Convierte esos puntos a maquinaria para cerrar huecos.

4. `INTERPOL` (opcional):
- Toma puntos maquinaria (clase 1).
- Busca vecinos de suelo alrededor.
- Interpola una nueva Z (IDW) y cambia clase a suelo.
- Resultado: superficie mas "limpia" para DTM.

---

## PRE_CLEAN_SURFACE: Que hace cada parametro

### `clean_eps` y `clean_min_samples`
- Funcion: DBSCAN para detectar ruido dentro de clase 1.
- Efecto:
- `clean_eps` bajo / `clean_min_samples` alto: mas conservador.
- `clean_eps` alto / `clean_min_samples` bajo: mas agresivo.
- Rango inicial recomendado: `clean_eps=1.0-1.5`, `clean_min_samples=5-10`.

### `clean_small_cluster_max_points`
### `clean_small_cluster_max_height`
### `clean_small_cluster_max_area`
- Funcion: regla combinada de "cluster chico".
- Si un cluster cumple los 3 umbrales, se mueve a suelo.
- Uso: eliminar islas pequenas de ruido.

### `clean_protect_tall` + `clean_protected_min_height`
- Funcion: evita borrar estructuras delgadas pero altas (ej. mastiles/perforadoras).
- Recomendado: mantener activo para faena minera.

### `clean_local_support_k` + `clean_local_support_ratio`
- Funcion: coherencia local de clase.
- Si un punto clase 1 tiene poco soporte de vecinos clase 1, se limpia.
- Uso: quitar puntos sueltos en aire o bordes muy ruidosos.

### `clean_hole_fill_enabled`
### `clean_hole_fill_radius`
### `clean_hole_fill_k`
### `clean_hole_fill_ratio`
- Funcion: relleno de micro-huecos previos a FIX_TECHO.
- Convierte suelo a maquinaria cuando hay suficiente vecindad clase 1.
- Uso: cerrar huecos internos antes del relleno volumetrico.

---

## FIX_TECHO: Que hace cada parametro

### `eps` (DBSCAN radius, metros)
- Funcion: radio para agrupar puntos de maquinaria en un mismo objeto.
- Efecto:
- `eps` bajo: separa objetos cercanos, puede partir una maquina en varios clusters.
- `eps` alto: une objetos distintos, sobre-rellena.
- Rango inicial recomendado: `2.0 - 3.0` (default `2.5`).

### `min_samples`
- Funcion: minimo de puntos para que DBSCAN considere un cluster valido.
- Efecto:
- bajo: detecta objetos pequenos pero mete ruido.
- alto: limpia ruido pero puede perder maquinaria pequena.
- Rango inicial recomendado: `20 - 50` (default `30`).

### `z_buffer` (metros)
- Funcion: protege suelo bajo maquinaria; no rellena por debajo de `min_z + z_buffer`.
- Efecto:
- bajo: riesgo de "comer" suelo real.
- alto: deja huecos sin rellenar en la base de maquinaria.
- Rango inicial recomendado: `1.0 - 2.0` (default `1.5`).

### `max_height` (metros)
- Funcion: techo maximo de busqueda, desde base del cluster (`min_z + max_height`).
- Efecto:
- bajo: recorta maquinaria alta.
- alto: incluye puntos que no deberian rellenarse.
- Rango inicial recomendado: `6 - 10` (default `8.0`).

### `padding` (metros)
- Funcion: expande en XY la caja de busqueda alrededor del cluster.
- Efecto:
- bajo: no alcanza bordes reales de maquinaria.
- alto: captura suelo lateral no deseado.
- Rango inicial recomendado: `1.0 - 2.0` (default `1.5`).

### `proximity_radius` (metros)
- Funcion: radio 2D de validacion final para convertir suelo -> maquinaria.
- Efecto:
- bajo: relleno mas estricto.
- alto: relleno mas agresivo.
- Rango inicial recomendado: `1.2 - 2.0` (default `1.5`).

### `smart_merge` (boolean)
- Funcion: pre-paso para unir fragmentos cercanos de maquinaria usando vecindad espacial.
- Cuando activarlo:
- maquinaria sale "rota" en islas pequenas.
- cuando hay gaps en laterales o transiciones.
- Riesgo:
- puede sumar falsos positivos si radio/criterio son agresivos.

### `merge_radius` (metros, Smart Merge)
- Funcion: radio de busqueda de vecinos de maquinaria para candidatos suelo.
- Efecto:
- bajo: no rellena gaps.
- alto: puede invadir suelo.
- Rango inicial recomendado: `2.0 - 3.0` (default `2.5`).

### `merge_neighbors` (entero, Smart Merge)
- Funcion: minimo de vecinos maquinaria para promover suelo -> maquinaria.
- Efecto:
- bajo: mas agresivo (mas union, mas riesgo de ruido).
- alto: mas conservador (menos union, puede dejar huecos).
- Rango inicial recomendado: `3 - 6` (default `4`).

---

## INTERPOL: Que hace cada parametro

### `k_neighbors`
- Funcion: cantidad de vecinos de suelo usados por IDW.
- Efecto:
- bajo: sigue detalle local, puede ser ruidoso.
- alto: suaviza mas, puede perder micro-relieve.
- Rango inicial recomendado: `8 - 16` (default `12`).

### `max_dist` (metros)
- Funcion: distancia maxima para aceptar vecinos de suelo en la interpolacion.
- Efecto:
- bajo: algunos puntos no encuentran vecinos y no se aplanan.
- alto: usa suelo lejano, puede deformar zonas complejas.
- Rango inicial recomendado: `30 - 60` (default `50`).

Nota tecnica:
- El codigo usa `valid_mask = np.isfinite(dists).all(axis=1)`.
- Si no hay `k` vecinos validos dentro de `max_dist`, ese punto maquinaria no se interpola.

---

## Recetas de Configuracion (listas para usar)

## 1) Conservadora (minimo riesgo de sobrecorreccion)
Para: cuando ya tienes buena inferencia y solo quieres limpiar un poco.

- PRE_CLEAN:
- `enabled=True`
- `clean_eps=1.0`
- `clean_min_samples=8`
- `clean_small_cluster_max_points=20`
- `clean_small_cluster_max_height=0.6`
- `clean_small_cluster_max_area=0.8`
- `clean_local_support_k=16`
- `clean_local_support_ratio=0.20`
- `clean_hole_fill_enabled=True`
- `clean_hole_fill_radius=1.0`
- `clean_hole_fill_k=10`
- `clean_hole_fill_ratio=0.75`

- FIX_TECHO:
- `eps=2.0`
- `min_samples=40`
- `z_buffer=1.8`
- `max_height=7.0`
- `padding=1.0`
- `proximity_radius=1.2`
- `smart_merge=False`

- INTERPOL:
- `k_neighbors=10`
- `max_dist=35`

Esperado:
- pocos cambios, alta precision, menor recall de huecos.

## 2) Balanceada (recomendada para empezar)
Para: uso general en cantera/planta sin casos extremos.

- PRE_CLEAN:
- `enabled=True`
- `clean_eps=1.2`
- `clean_min_samples=6`
- `clean_small_cluster_max_points=35`
- `clean_small_cluster_max_height=0.9`
- `clean_small_cluster_max_area=1.2`
- `clean_local_support_k=16`
- `clean_local_support_ratio=0.25`
- `clean_hole_fill_enabled=True`
- `clean_hole_fill_radius=1.2`
- `clean_hole_fill_k=12`
- `clean_hole_fill_ratio=0.70`

- FIX_TECHO:
- `eps=2.5`
- `min_samples=30`
- `z_buffer=1.5`
- `max_height=8.0`
- `padding=1.5`
- `proximity_radius=1.5`
- `smart_merge=True`
- `merge_radius=2.5`
- `merge_neighbors=4`

- INTERPOL:
- `k_neighbors=12`
- `max_dist=50`

Esperado:
- buen equilibrio precision/recall y DTM estable.

## 3) Agresiva (cerrar huecos y fragmentacion fuerte)
Para: maquinaria muy fragmentada o sombras duras que rompen la clase 1.

- PRE_CLEAN:
- `enabled=True`
- `clean_eps=1.6`
- `clean_min_samples=4`
- `clean_small_cluster_max_points=60`
- `clean_small_cluster_max_height=1.2`
- `clean_small_cluster_max_area=2.0`
- `clean_local_support_k=14`
- `clean_local_support_ratio=0.20`
- `clean_hole_fill_enabled=True`
- `clean_hole_fill_radius=1.8`
- `clean_hole_fill_k=14`
- `clean_hole_fill_ratio=0.60`

- FIX_TECHO:
- `eps=3.0`
- `min_samples=20`
- `z_buffer=1.0`
- `max_height=10.0`
- `padding=2.0`
- `proximity_radius=2.0`
- `smart_merge=True`
- `merge_radius=3.0`
- `merge_neighbors=3`

- INTERPOL:
- `k_neighbors=16`
- `max_dist=60`

Esperado:
- sube recall de maquinaria, pero vigilar falsos positivos.

---

## Estrategia de Tuneo Paso a Paso
1. Ajusta `PRE_CLEAN_SURFACE` primero.
- Objetivo: quitar islas de ruido y cerrar micro-huecos internos.

2. Ajusta `FIX_TECHO` despues.
- Objetivo: maquinaria compacta sin tragarte suelo.

3. Luego ajusta `INTERPOL`.
- Objetivo: DTM continuo sin artefactos ni "pozos" raros.

4. Cambia pocos parametros por iteracion.
- Recomendado: 1-2 parametros por corrida.

5. Evalua visualmente 3 zonas:
- bordes de maquinaria
- base de maquinaria (evitar comer suelo)
- zonas abiertas de suelo (evitar deformacion)

6. Guarda presets por escenario.
- Ejemplo: `preset_seco`, `preset_sombra_dura`, `preset_material_fino`.

---

## Sintomas Tipicos -> Causa -> Correccion

### Sintoma: "Se unen maquinas separadas"
- Causa probable: `eps` alto o `padding` alto.
- Correccion:
- baja `eps` (p.ej. `3.0 -> 2.3`)
- baja `padding` (p.ej. `2.0 -> 1.2`)

### Sintoma: "La maquinaria sigue hueca por dentro"
- Causa probable: `z_buffer` alto o `merge_neighbors` alto.
- Correccion:
- baja `z_buffer` (`1.8 -> 1.2`)
- baja `merge_neighbors` (`5 -> 3`)
- sube levemente `merge_radius` (`2.0 -> 2.5`)

### Sintoma: "Me esta comiendo suelo bueno"
- Causa probable: FIX_TECHO agresivo.
- Correccion:
- sube `z_buffer`
- sube `min_samples`
- desactiva `smart_merge` temporalmente

### Sintoma: "El DTM queda ondulado/ruidoso"
- Causa probable: `k_neighbors` bajo.
- Correccion:
- sube `k_neighbors` (`8 -> 12` o `16`)

### Sintoma: "Hay puntos de maquinaria que no se aplanan"
- Causa probable: `max_dist` bajo (no encuentra k vecinos).
- Correccion:
- sube `max_dist` (`30 -> 50` o `60`)

---

## Ejemplo Practico de Iteracion
Escenario: maquinaria fragmentada + DTM con huecos.

Iteracion 0 (base):
- `eps=2.5, min_samples=30, z_buffer=1.5, padding=1.5`
- `smart_merge=True, merge_radius=2.5, merge_neighbors=4`
- `k_neighbors=12, max_dist=50`

Observacion:
- mejora parcial, quedan huecos laterales.

Iteracion 1:
- `merge_radius: 2.5 -> 3.0`
- `merge_neighbors: 4 -> 3`

Observacion:
- cierra huecos, aparece algo de sobre-etiquetado en borde.

Iteracion 2 (correccion):
- `padding: 1.5 -> 1.2`
- `z_buffer: 1.5 -> 1.8`

Resultado:
- bordes mas limpios, relleno interno mantenido.

---

## Parametros en la UI
Todos los parametros anteriores estan disponibles en:
- App Gradio -> seccion `Parametros` dentro de `app_inference/ui/app.py`.

Defaults actuales en codigo:
- PRE_CLEAN: `disabled`, `clean_eps=1.2`, `clean_min_samples=6`, `hole_fill_radius=1.2`
- FIX_TECHO: `eps=2.5`, `min_samples=30`, `z_buffer=1.5`, `max_height=8.0`, `padding=1.5`, `proximity_radius=1.5`
- SMART MERGE: `enabled`, `merge_radius=2.5`, `merge_neighbors=4`
- INTERPOL: `k_neighbors=12`, `max_dist=50`

---

## Recomendacion Final
Si tu objetivo principal es calidad de DTM:
1. Usa preset balanceado.
2. Ajusta `z_buffer` y `padding` hasta no comer suelo.
3. Ajusta `k_neighbors` para suavidad.
4. Ajusta `max_dist` solo si quedan puntos sin aplanar.
