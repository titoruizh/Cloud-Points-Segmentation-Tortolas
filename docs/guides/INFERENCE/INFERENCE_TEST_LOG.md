# Bitacora de Tests de Inferencia (V2/V3/V4)

## Objetivo
Registrar configuraciones usadas, resultado observado y decisiones para calibrar inferencia en dos dominios:

- `Tranque` (dominio de entrenamiento)
- `Rajo minero` (dominio nuevo, no visto en entrenamiento)

---

## Confirmacion de defaults al reiniciar
La app **no persiste estado** entre sesiones. Al reiniciar, vuelve a los `value=` definidos en UI.

Referencia:
- `app_inference/ui/app.py:438`
- `app_inference/ui/app.py:440`
- `app_inference/ui/app.py:469`
- `app_inference/ui/app.py:476`
- `app_inference/ui/app.py:480`

Defaults actuales (relevantes):
- `batch_size=64`
- `torch.compile=True`
- `confidence=0.5`
- `pre_clean_enabled=False`
- `eps=2.5`
- `min_samples=30`
- `z_buffer=1.5`
- `max_height=8.0`
- `padding=1.5`
- `proximity_radius=1.5`
- `smart_merge=True`
- `merge_radius=2.5`
- `merge_neighbors=4`
- `k_neighbors=12`
- `max_dist=50`

Nota:
- En parámetros previos ya existentes (los que no agregamos), se mantienen los mismos defaults que estaban en la UI.
- Se agregaron nuevos controles (`PRE_CLEAN`, `min_samples`, `proximity_radius`) con defaults conservadores.

---

## V2 (referencia previa, no satisfactoria)
Estado:
- Reportado como "no me gusto".
- Sin detalle completo de todos los sliders en este registro.

Uso:
- Mantener solo como referencia histórica.

---

## V3 - Agresivo (base que funciono bien en rajo)

### Parametros
- `batch_size=256`
- `torch.compile=True`
- `confidence=0.20`

- `PRE_CLEAN`: `OFF` (no existia en esa corrida original)

- `FIX_TECHO`:
- `eps=4.0`
- `min_samples=20`
- `z_buffer=0.5`
- `max_height=12.0`
- `padding=2.5`
- `proximity_radius=2.2` (equivalente agresivo actual)

- `SMART_MERGE`:
- `enabled=True`
- `merge_radius=4.0`
- `merge_neighbors=2`

- `INTERPOL`:
- `k_neighbors=6`
- `max_dist=100`

### Resultado observado
- `Rajo minero`: mejora considerable, DTM casi solo suelo, mucho mejor que antes.
- `Tranque`: tiende a introducir halo/ruido si se aplica sin adaptación.

### Conclusión
- Config muy buena para recall y DTM en dominio nuevo.
- Demasiado agresiva para dominio entrenado (tranque).

---

## V4 - Agresivo + PRE_CLEAN (estado actual)

### Parametros usados
- `batch_size=256`
- `torch.compile=True`
- `confidence=0.20`

- `PRE_CLEAN`:
- `enabled=True`
- `clean_eps=1.6`
- `clean_min_samples=4`
- `clean_small_cluster_max_points=60`
- `clean_small_cluster_max_height=1.2`
- `clean_small_cluster_max_area=2.0`
- `clean_protect_tall=True`
- `clean_protected_min_height=1.8`
- `clean_local_support_k=14`
- `clean_local_support_ratio=0.20`
- `clean_hole_fill_enabled=True`
- `clean_hole_fill_radius=1.8`
- `clean_hole_fill_k=14`
- `clean_hole_fill_ratio=0.60`

- `FIX_TECHO`:
- `eps=4.0`
- `min_samples=20`
- `z_buffer=0.5`
- `max_height=12.0`
- `padding=2.5`
- `proximity_radius=2.2`

- `SMART_MERGE`:
- `enabled=True`
- `merge_radius=4.0`
- `merge_neighbors=2`

- `INTERPOL`:
- `k_neighbors=6`
- `max_dist=100`

### Resultado observado
- `Rajo minero`: mejora fuerte de DTM; aun con ruido, resultado util para produccion.
- `Tranque`: peor que baseline; aparecen clusters grandes de error y mayor halo.

### Diagnostico
- Hay `domain shift` fuerte entre ambos escenarios.
- Una sola config agresiva no generaliza bien a ambos dominios.
- En tranque, esta config sobre-expande maquinaria (FPR alto).

---

## Recomendacion operativa inmediata

1. Mantener dos presets separados:
- `Preset_Rajo_Agresivo` = V4 actual.
- `Preset_Tranque_Conservador` = defaults + ajustes leves.

2. Preset sugerido para tranque (inicio):
- `confidence=0.35`
- `pre_clean_enabled=False`
- `eps=2.5`
- `min_samples=35`
- `z_buffer=1.5`
- `max_height=8.0`
- `padding=1.2`
- `proximity_radius=1.2`
- `smart_merge=True`
- `merge_radius=2.0`
- `merge_neighbors=4`
- `k_neighbors=10`
- `max_dist=50`

3. Criterio de selección:
- Si objetivo es DTM en rajo: usar preset agresivo.
- Si objetivo es precisión en tranque: usar preset conservador.

---

## Proximo registro recomendado (V5)
Probar `Tranque_Conservador` y registrar:
- nivel de halo
- tamaño de clusters falsos
- continuidad de DTM
- tiempo por archivo

Este archivo se debe actualizar en cada iteracion de tuning.
