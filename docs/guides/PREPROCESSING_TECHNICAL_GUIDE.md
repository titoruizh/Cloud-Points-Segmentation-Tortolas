# Pipeline de Preprocesamiento - Documentación Técnica Completa

## 📋 Resumen Ejecutivo

Este documento explica **CADA DECISIÓN TÉCNICA** del pipeline de preprocesamiento de nubes de puntos para detección de maquinaria en minería.

**Objetivo**: Convertir archivos .laz (nubes de puntos fotogramétricas) en bloques .npy optimizados para entrenamiento de redes neuronales (PointNet2, RandLANet, MiniPointNet).

**Formato de salida**: `(N, 8)` → `[x, y, z, nx, ny, nz, verticalidad, label]`

---

## 🔄 Pipeline Completo - Paso a Paso

### **PASO 1: Carga de Datos (.laz → numpy)**

```python
las = laspy.read(filepath)
xyz = np.vstack((las.x, las.y, las.z)).transpose()  # Shape: (N, 3)
labels = np.array(las.classification)  # Shape: (N,)
```

**Razones técnicas:**
- `.laz` = formato comprimido LAS (LiDAR/Fotogrametría)
- Extraemos coordenadas XYZ como array NumPy para procesamiento vectorizado (100x más rápido que loops)
- Labels vienen de clasificación manual: 1=Maquinaria, 2=Suelo

---

### **PASO 2: Remapeo de Labels (2→0, 1→1)**

```python
labels_remapped = np.zeros_like(labels)
labels_remapped[labels == 1] = 1  # Maquinaria
labels_remapped[labels == 2] = 0  # Suelo
```

**Razones técnicas:**
- **PyTorch CrossEntropyLoss** espera clases en rango `[0, num_classes-1]`
- Estándar LAS usa 1=Maquinaria, 2=Suelo
- Remapeamos a **0=Suelo (clase mayoritaria), 1=Maquinaria (clase de interés)**
- Esto es convención en clasificación binaria (clase positiva = 1)

---

### **PASO 3: Cálculo de Normales (Open3D)**

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2.0,  # 2 metros
        max_nn=50    # Máximo 50 vecinos
    )
)
```

#### **¿Por qué normales?**
- Las normales capturan la **orientación local de la superficie**
- Diferencian:
  - Superficies planas (suelo) → normal apunta hacia arriba [0, 0, 1]
  - Superficies verticales (ruedas, cabinas) → normal horizontal [1, 0, 0]
  - Superficies inclinadas (taludes) → normal intermedia [0.7, 0, 0.7]
- Son **invariantes a traslación**: no importa dónde esté el objeto, su geometría es la misma

#### **¿Por qué radius=2.0m?**
- **Problema**: Datos fotogramétricos tienen ruido de "olas" en el suelo (variaciones de ±10cm)
- **Radio pequeño (0.5m)**: Normales ruidosas, capturan cada ola → suelo parece rugoso
- **Radio grande (5m)**: Normales sobre-suavizadas, pierden detalles de maquinaria → ruedas se difuminan
- **Radio 2.0m (sweet spot)**: 
  - Suaviza ruido del suelo (promedia sobre 12m² ≈ 100-200 puntos)
  - Mantiene detalles de maquinaria (rueda de 1m sigue siendo detectable)

#### **¿Por qué max_nn=50?**
- **Balance entre calidad y velocidad**:
  - Más vecinos = normales más robustas (menos afectadas por outliers)
  - Más vecinos = más lento (complejidad O(N log N) del KDTree)
- **50 vecinos** en 2m de radio es suficiente para:
  - Superficies estables (varianza < 5°)
  - Tiempo razonable (~30 seg por 7M puntos en CPU)

#### **¿Por qué KDTreeSearchParamHybrid?**
- Combina **búsqueda por radio** Y **límite de vecinos**
- **Ventaja**: Evita que zonas densas (millones de puntos) se vuelvan extremadamente lentas
- **Comportamiento**:
  - Zona densa (200 puntos en 2m): limita a 50 vecinos → rápido
  - Zona dispersa (30 puntos en 2m): usa todos los 30 → robusto
- **Alternativa rechazada**: `KDTreeSearchParamRadius(2.0)` sin límite → puede usar 500+ vecinos en zonas densas → 10x más lento

---

### **PASO 4: Orientación de Normales (+Z)**

```python
pcd.orient_normals_to_align_with_direction(
    orientation_reference=np.array([0., 0., 1.])
)
normals = np.asarray(pcd.normals)
normals[normals[:, 2] < 0] *= -1  # Forzar hacia arriba
```

#### **¿Por qué orientar?**
- **Problema**: Normales tienen **ambigüedad de 180°**
  - Mismo plano puede tener normal [0,0,1] o [0,0,-1]
  - Sin orientación: red neuronal ve features diferentes para misma geometría
- **Solución**: Forzar consistencia → todas las normales apuntan hacia +Z (arriba)

#### **¿Por qué hacia +Z (arriba)?**
- **Asunción**: Minería a cielo abierto (vista desde arriba con dron/fotogrametría)
- **Lógica**: Todas las superficies "visibles" apuntan hacia el sensor (arriba)
- **Beneficio**: Consistencia geométrica
  - Suelo siempre tiene Nz ≈ 1.0
  - Paredes verticales siempre tienen Nz ≈ 0.0
  - Taludes 45° siempre tienen Nz ≈ 0.7

#### **¿Por qué el fix adicional `normals[:, 2] < 0`?**
- `orient_normals_to_align_with_direction` usa heurística (no es 100% confiable)
- **Casos problemáticos**: Superficies con oclusiones o ruido pueden quedar invertidas
- **Fix**: Forzamos manualmente que **TODAS** las normales tengan componente Z positiva
- **Justificación**: En minería a cielo abierto, no hay superficies mirando hacia abajo

---

### **PASO 5: Cálculo de Verticalidad**

```python
verticality = 1.0 - np.abs(normals[:, 2])
```

#### **¿Qué es verticalidad?**
- Mide qué tan **vertical** es una superficie
- **Fórmula**: `vert = 1 - |Nz|`
- **Rango**: [0, 1]
  - **0.0** = Superficie horizontal (suelo, techo de camión)
  - **1.0** = Superficie vertical (pared, rueda, cabina)
  - **0.3** = Talud 45°

#### **¿Por qué `1.0 - abs(Nz)`?**
- **Ejemplos**:
  - Normal de suelo plano: [0, 0, 1] → Nz=1 → vert=0 ✅
  - Normal de pared vertical: [1, 0, 0] → Nz=0 → vert=1 ✅
  - Normal de talud 45°: [0.7, 0, 0.7] → Nz=0.7 → vert=0.3 ✅

#### **¿Por qué es útil?**
- **Discrimina geometría compleja**:
  - **Maquinaria**: Muchas superficies verticales (ruedas, cabinas, brazos) → vert alta (0.5-1.0)
  - **Suelo plano**: Solo superficies horizontales → vert baja (0.0-0.1)
  - **Taludes/rocas**: Superficies inclinadas → vert media (0.2-0.5)
- **Permite filtrado inteligente**:
  - HARD_NEGATIVE: bloques con vert > 0.20 (geometría compleja sin maquinaria)
  - EASY_NEGATIVE: bloques con vert < 0.10 (suelo plano simple)

#### **¿Por qué no usar directamente Nz?**
- **Verticalidad es más interpretable**:
  - "Dame bloques con vert > 0.2" es más claro que "Nz < 0.8"
  - Facilita ajuste de umbrales sin confusión de signos

---

### **PASO 6: Clustering DBSCAN (Solo para MACHINERY)**

```python
clustering = DBSCAN(eps=3.0, min_samples=20).fit(mach_xyz)
```

#### **¿Por qué DBSCAN?**
- **Ventajas sobre K-means**:
  - Encuentra clusters de **forma arbitraria** (camiones no son círculos)
  - **No requiere** saber el número de clusters de antemano
  - Maneja **ruido** (puntos aislados marcados como -1)
- **Ideal para maquinaria**: Camiones, excavadoras tienen formas irregulares

#### **¿Por qué eps=3.0m?**
- `eps` = radio máximo para considerar puntos "conectados"
- **Geometría de camión típico**:
  - Largo: ~5m
  - Ancho: ~2.5m
  - Altura: ~3m
- **eps=3.0m**: Conecta todos los puntos de un mismo vehículo
- **Alternativas rechazadas**:
  - eps=1.0m → Fragmenta un camión en múltiples clusters (ruedas separadas de cabina)
  - eps=10.0m → Une múltiples camiones cercanos en un solo cluster

#### **¿Por qué min_samples=20?**
- Mínimo de puntos para formar un cluster válido
- **Filtra ruido**: Puntos aislados de maquinaria mal clasificados (falsos positivos)
- **20 puntos** ≈ mínimo para representar un objeto pequeño:
  - Balde de excavadora: ~30-50 puntos
  - Cono de tráfico: ~10-20 puntos (filtrado como ruido)
- **Evita bloques inútiles**: No creamos bloques para "falsos positivos" de 2-3 puntos

#### **¿Por qué calcular el centroide?**
```python
center = np.mean(cluster_points, axis=0)
```
- **Centroide** = centro geométrico del cluster
- **Garantiza** que el bloque 10x10m esté **centrado en la maquinaria**
- **Maximiza probabilidad** de capturar el objeto completo
- **Evita bloques mal centrados**: Maquinaria en una esquina → contexto incompleto

---

### **PASO 7: Corte de Bloques (crop_block)**

```python
half = block_size / 2.0
mask = (
    (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
    (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
)
xyz_crop = xyz[mask].copy()
```

#### **¿Por qué cuadrados en XY?**
- Bloques de **10×10m en planta** (vista aérea)
- **NO cortamos en Z**: tomamos toda la altura
- **Razón**: Maquinaria puede estar en diferentes elevaciones (taludes, rampas)

#### **¿Por qué `< cx + half` (sin =)?**
- **Evita solapamiento de bordes** entre bloques adyacentes
- **Garantiza** que cada punto pertenezca a **un solo bloque**
- **Importante**: Evita duplicados en el dataset (mismo punto en múltiples bloques)

---

### **PASO 8: Normalización de Coordenadas (CRÍTICO)**

```python
xyz_crop[:, 0] -= cx  # Centrado en X
xyz_crop[:, 1] -= cy  # Centrado en Y
xyz_crop[:, 2] -= np.min(xyz_crop[:, 2])  # Z relativo al suelo
```

#### **¿Por qué normalizar X, Y?**
- **Invarianza a posición absoluta**: El modelo debe aprender **geometría**, no coordenadas GPS
- **Sin normalización**:
  - Bloque en (100, 200) vs (500, 600) → features diferentes
  - Red aprende "camiones están en (100, 200)" → NO generaliza
- **Con normalización**:
  - Ambos bloques tienen X,Y ∈ [-5, 5] → features iguales
  - Red aprende "forma de camión" → generaliza a cualquier ubicación

#### **¿Por qué Z relativo al mínimo?**
- **Invarianza a elevación absoluta**:
  - Camión a 100m de altura vs 500m → mismo objeto
  - Sin normalización: red aprende "camiones están a 100m" → NO generaliza
- **Z=0 siempre es el "suelo" del bloque**
- **Preserva altura relativa**:
  - Rueda a 2m del suelo → sigue siendo 2m
  - Cabina a 4m del suelo → sigue siendo 4m
- **Ayuda al modelo** a aprender "altura sobre el suelo" en lugar de "altura absoluta"

#### **¿Por qué NO normalizar las normales?**
- Normales ya están **normalizadas** (magnitud = 1)
- Su dirección es **invariante a traslación**
- Rotar/trasladar el objeto **NO cambia** las normales

#### **¿Por qué NO normalizar la verticalidad?**
- Ya está en rango [0, 1]
- Es una **propiedad geométrica intrínseca**
- Independiente de posición o escala

---

### **PASO 9: Formato de Guardado**

```python
save_array = np.hstack((data, lbl.reshape(-1, 1)))
# Shape: (N, 8) = [x, y, z, nx, ny, nz, vert, label]
np.save(output_path, save_array.astype(np.float32))
```

#### **¿Por qué .npy?**
- Formato binario de NumPy: **extremadamente rápido** de cargar
- **vs .txt**: 10-100x más rápido
- **vs .las**: No necesitamos metadata LiDAR, solo geometría
- **Carga**: `np.load()` → 0.1 seg para 10,000 puntos

#### **¿Por qué float32 en lugar de float64?**
- **Reduce tamaño a la mitad**: 4 bytes vs 8 bytes por número
- **PyTorch usa float32** por defecto en GPU
- **Precisión suficiente**: 7 dígitos decimales
  - Coordenadas en metros: 0.0001m = 0.1mm (más que suficiente)
  - Normales: 0.001 de precisión (ángulo de 0.06°)

#### **¿Por qué guardar label junto con features?**
- **Un solo archivo** por bloque → más fácil de manejar
- **Evita desincronización** entre archivos de features y labels
- **Carga atómica**: O se carga todo o nada (no hay corrupción parcial)

---

## 🎯 Mejoras Implementadas (Versión Final)

### **1. Filtro de Ratio de Maquinaria (min 3%)**

```python
machinery_ratio = np.sum(crop_labels == 1) / len(crop_labels)
if machinery_ratio >= 0.03:  # Mínimo 3%
    crops.append((crop, crop_labels, "MACHINERY", machinery_ratio))
```

**Razón**: Eliminar bloques MACHINERY con muy poca maquinaria (< 3%)
- **Antes**: Bloques con 0.5% maquinaria (50 puntos de 10,000)
- **Ahora**: Bloques con mínimo 3% maquinaria (300 puntos de 10,000)
- **Resultado**: **23% más maquinaria** por bloque (4.13% vs 3.35%)

### **2. Eliminación de EASY_NEGATIVE**

```python
EASY_NEGATIVE_RATIO = 0.0  # Eliminado
```

**Razón**: Ya hay suficiente suelo en bloques MACHINERY y HARD_NEGATIVE
- **Antes**: 13 MACH + 7 HARD + 3 EASY = 23 bloques
- **Ahora**: 11 MACH + 7 HARD + 0 EASY = 18 bloques
- **Beneficio**: Más enfoque en geometría compleja

### **3. Umbral de Verticalidad Más Estricto (0.20)**

```python
HARD_VERTICALITY_THRESHOLD = 0.20  # Antes: 0.15
```

**Razón**: Filtrar solo geometría realmente compleja
- **Antes**: vert > 0.15 (incluye taludes suaves)
- **Ahora**: vert > 0.20 (solo taludes pronunciados, rocas)
- **Beneficio**: HARD_NEGATIVE más desafiantes para el modelo

### **4. Ratio HARD_NEGATIVE Aumentado (0.8)**

```python
HARD_NEGATIVE_RATIO = 0.8  # Antes: 0.5
```

**Razón**: Más ejemplos de geometría compleja sin maquinaria
- **Antes**: 0.5 × MACHINERY bloques
- **Ahora**: 0.8 × MACHINERY bloques
- **Beneficio**: Modelo aprende mejor a distinguir taludes de maquinaria

---

## 📊 Resultados Finales

**Mejora en balance de clases**:
- Versión anterior: 3.35% maquinaria
- Versión mejorada: **4.13% maquinaria** (+23%)

**Distribución de bloques**:
- 61% MACHINERY (centrados en objetos)
- 39% HARD_NEGATIVE (geometría compleja)
- 0% EASY_NEGATIVE (eliminado)

**Calidad de features**:
- Coordenadas normalizadas: X,Y ∈ [-5, 5], Z ∈ [0, altura_bloque]
- Normales válidas: Nx,Ny,Nz ∈ [-1, 1], siempre apuntando hacia +Z
- Verticalidad: [0, 1], media 0.135 (indica mezcla de suelo y geometría compleja)
