root@8075d3e39eaa:/workspaces/Cloud-Point-Research V2 Docker C# python main_inference_app.py 


                                                                  ║
   🚀 Point Cloud Inference App V5                               ║
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
                                                                  ║
   PointNet++ "Geometric Purification"                           ║
   Optimizado para RTX 5090 | FP16 + torch.compile               ║
                                                                  ║
   Pipeline: Inferencia → FIX_TECHO → INTERPOL (DTM)             ║
                                                                  ║


============================================================
📊 INFORMACIÓN DEL SISTEMA
============================================================
Sistema Operativo: Linux 6.6.87.2-microsoft-standard-WSL2
Python: 3.12.3

🔥 GPU:
   GPU Detectada: NVIDIA GeForce RTX 5090
   VRAM Total: 31.84 GB
   VRAM Reservada: 0.00 GB
   VRAM Usada: 0.00 GB
   VRAM Libre: 31.84 GB

💻 CPU:
   Núcleos: 32
   Frecuencia: 1997 MHz
   RAM Total: 62.71 GB
   RAM Disponible: 59.70 GB
   RAM Usada: 4.8%
============================================================

🔍 Verificando dependencias...
✅ Dependencias OK

📁 Directorio de salida: /workspaces/Cloud-Point-Research V2 Docker C/data/predictions/app_output

🌐 Iniciando servidor en puerto 7860...
   Abre en tu navegador: http://localhost:7860

============================================================

* Running on local URL:  http://0.0.0.0:7860
* To create a public link, set `share=True` in `launch()`.
   🔍 [Antes de cargar modelo] GPU Memory: Usada=0.00GB, Reservada=0.00GB, Libre=31.84GB, Total=31.84GB
   🔍 [Después de cargar modelo] GPU Memory: Usada=0.01GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB

======================================================================
🎯 INICIANDO INFERENCIA: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m.laz
======================================================================
   🔍 [Inicio de inferencia] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   📦 Tamaño del archivo: 244.36 MB
   🔍 [Antes de extraer features] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   🧮 Calculando normales en chunks espaciales (r=3.5m, ~50m x 50m por chunk)...
   🔥 Normales: usando GPU (Open3D Tensor CUDA)
   📐 Nube: 76,536,995 puntos → ~30 chunks (10×3) de 500m
   ⚡ Chunk 1/30 | core=3,071,381 pts | 1s elapsed  ETA 19s
   ⚡ Chunk 2/30 | core=3,779,447 pts | 1s elapsed  ETA 16s
   ⚡ Chunk 3/30 | core=1,412,694 pts | 1s elapsed  ETA 12s
   ⚡ Chunk 4/30 | core=3,255,564 pts | 2s elapsed  ETA 11s
   ⚡ Chunk 5/30 | core=3,988,372 pts | 2s elapsed  ETA 11s
   ⚡ Chunk 6/30 | core=1,901,925 pts | 3s elapsed  ETA 10s
   ⚡ Chunk 7/30 | core=3,161,304 pts | 3s elapsed  ETA 10s
   ⚡ Chunk 8/30 | core=3,979,412 pts | 3s elapsed  ETA 9s
   ⚡ Chunk 9/30 | core=1,757,545 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 10/30 | core=2,682,968 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 11/30 | core=2,955,260 pts | 4s elapsed  ETA 7s
   ⚡ Chunk 12/30 | core=1,426,306 pts | 5s elapsed  ETA 7s
   ⚡ Chunk 13/30 | core=3,265,826 pts | 5s elapsed  ETA 6s
   ⚡ Chunk 14/30 | core=3,995,523 pts | 5s elapsed  ETA 6s
   ⚡ Chunk 15/30 | core=1,611,844 pts | 6s elapsed  ETA 6s
   ⚡ Chunk 16/30 | core=3,216,534 pts | 6s elapsed  ETA 5s
   ⚡ Chunk 17/30 | core=3,991,073 pts | 6s elapsed  ETA 5s
   ⚡ Chunk 18/30 | core=1,543,374 pts | 7s elapsed  ETA 4s
   ⚡ Chunk 19/30 | core=3,202,135 pts | 7s elapsed  ETA 4s
   ⚡ Chunk 20/30 | core=3,987,316 pts | 7s elapsed  ETA 4s
   ⚡ Chunk 21/30 | core=1,624,251 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 22/30 | core=3,180,516 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 23/30 | core=3,989,658 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 24/30 | core=1,594,611 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 25/30 | core=3,008,817 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 9s elapsed  ETA 1s
   ⚡ Chunk 27/30 | core=816,914 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 10s elapsed  ETA 0s
   ✅ Normales completadas: 10.2s  (7,484,100 pts/s)
   💾 Array de features: 2627.69 MB en RAM
   🔍 [Después de extraer features] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   → 48903 bloques activos
⚙️ Configurando DataLoader (batch_size=256, workers=12)...
   🔍 [Antes de inferencia GPU] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
🧠 Ejecutando inferencia en GPU...
   Total de batches: 192
   → Batch 1/192 (0.5%)
   → Batch 10/192 (5.2%)
   → Batch 20/192 (10.4%)
   → Batch 30/192 (15.6%)
   → Batch 40/192 (20.8%)
   → Batch 50/192 (26.0%)
   🔍 [Batch 50] GPU Memory: Usada=0.04GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
   → Batch 60/192 (31.2%)
   → Batch 70/192 (36.5%)
   → Batch 80/192 (41.7%)
   → Batch 90/192 (46.9%)
   → Batch 100/192 (52.1%)
   🔍 [Batch 100] GPU Memory: Usada=0.04GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
   → Batch 110/192 (57.3%)
   → Batch 120/192 (62.5%)
   → Batch 130/192 (67.7%)
   → Batch 140/192 (72.9%)
   → Batch 150/192 (78.1%)
   🔍 [Batch 150] GPU Memory: Usada=0.04GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
   → Batch 160/192 (83.3%)
   → Batch 170/192 (88.5%)
   → Batch 180/192 (93.8%)
   → Batch 190/192 (99.0%)
   🔍 [Después de inferencia GPU] GPU Memory: Usada=0.01GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
   🔍 [Final de inferencia] GPU Memory: Usada=0.01GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
✅ Inferencia completada en 158.8s - Maquinaria: 165,936 puntos (0.2%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15602 MB
   🚜 Maquinaria: 165,936 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   ⚡ Smart Merge GPU: NVIDIA GeForce RTX 5090
   🔍 Smart Merge [GPU+CPU fallback]: 76,256,408 candidatos en 153 bloques
   🔍 Smart Merge: 76,256,408 candidatos
   ⚠️ Smart Merge abortado: 28,894,261 pts exceden umbral (829,680 = 5× maq original). Usando clasificación original sin merge.
   🔢 Objetos encontrados: 781
   ⚡ Procesando en paralelo 781 objetos...
   ✅ Rellenados 210,973 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   📊 RSS al iniciar INTERPOL: 15602 MB
   📉 Maquinaria: 376,845 pts | Suelo: 76,160,150 pts | RAM arrays: 949 MB
   📉 Maquinaria: 376,845 | Suelo: 76,160,150 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1607.66m, mediana_suelo=1604.10m, gap=3.56m
   📐 Altura: mediana_maq=1607.66m, mediana_suelo=1604.10m, gap=3.56m
   ⚡ INTERPOL modo: GPU (HIGH, 31.8GB)
   ⚡ INTERPOL modo: GPU (HIGH, 31.8GB)
   📊 Voxel downsample global: 76,160,150 → 4,104,294 (voxel=1.08m) en 3.8s
   📊 Voxel downsample global: 76,160,150 → 4,104,294 (voxel=1.08m) en 3.8s
   🔍 GPU knn: 4,104,294 suelo × 376,845 maq, k=6...
   🔍 GPU knn: 4,104,294 suelo × 376,845 maq, k=6...
   ⚡ GPU knn+IDW completado en 4.8s
   ⚡ GPU knn+IDW completado en 4.8s
   📊 Z diagnostico: 342,446/376,845 puntos con dZ>1cm | dZ medio=0.457m | dZ max=9.789m
   📊 Z diagnostico: 342,446/376,845 puntos con dZ>1cm | dZ medio=0.457m | dZ max=9.789m
   ✅ Aplanados 376,845 puntos
   ✅ INTERPOL: 376,845 puntos aplanados
💾 DTM guardado en 24.7s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 24.7s
