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
   RAM Disponible: 59.84 GB
   RAM Usada: 4.6%
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
   ⚡ Chunk 2/30 | core=3,779,447 pts | 1s elapsed  ETA 15s
   ⚡ Chunk 3/30 | core=1,412,694 pts | 1s elapsed  ETA 12s
   ⚡ Chunk 4/30 | core=3,255,564 pts | 2s elapsed  ETA 11s
   ⚡ Chunk 5/30 | core=3,988,372 pts | 2s elapsed  ETA 11s
   ⚡ Chunk 6/30 | core=1,901,925 pts | 2s elapsed  ETA 10s
   ⚡ Chunk 7/30 | core=3,161,304 pts | 3s elapsed  ETA 9s
   ⚡ Chunk 8/30 | core=3,979,412 pts | 3s elapsed  ETA 9s
   ⚡ Chunk 9/30 | core=1,757,545 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 10/30 | core=2,682,968 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 11/30 | core=2,955,260 pts | 4s elapsed  ETA 7s
   ⚡ Chunk 12/30 | core=1,426,306 pts | 4s elapsed  ETA 7s
   ⚡ Chunk 13/30 | core=3,265,826 pts | 5s elapsed  ETA 6s
   ⚡ Chunk 14/30 | core=3,995,523 pts | 5s elapsed  ETA 6s
   ⚡ Chunk 15/30 | core=1,611,844 pts | 5s elapsed  ETA 5s
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
   ✅ Normales completadas: 10.1s  (7,600,717 pts/s)
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
✅ Inferencia completada en 156.8s - Maquinaria: 210,727 puntos (0.3%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15648 MB
   🚜 Maquinaria: 210,727 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   ⚡ Smart Merge GPU: NVIDIA GeForce RTX 5090
   🔍 Smart Merge [GPU+CPU fallback]: 76,215,927 candidatos en 153 bloques
   🔍 Smart Merge: 76,215,927 candidatos
   ⚠️ Smart Merge abortado: 32,234,813 pts exceden umbral (1,053,635 = 5× maq original). Usando clasificación original sin merge.
   🔢 Objetos encontrados: 954
   ⚡ Procesando en paralelo 954 objetos...
   ✅ Rellenados 246,276 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   📊 RSS al iniciar INTERPOL: 15648 MB
   📉 Maquinaria: 456,931 pts | Suelo: 76,080,064 pts | RAM arrays: 949 MB
   📉 Maquinaria: 456,931 | Suelo: 76,080,064 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1606.88m, mediana_suelo=1604.10m, gap=2.78m
   📐 Altura: mediana_maq=1606.88m, mediana_suelo=1604.10m, gap=2.78m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL modo: GPU (HIGH, 31.8GB libre)
   ⚡ INTERPOL modo: GPU (HIGH, 31.8GB libre)
   ⚡ INTERPOL chunk 1/95 [GPU] | maq=381 | suelo_local=1,395,319 [↓22M→1395k] | 7s  ETA 634s
   ⚡ INTERPOL chunk 1/95 [GPU] | maq=381 | suelo_local=1,395,319 [↓22M→1395k] | 7s  ETA 634s
   ⚡ INTERPOL chunk 2/95 [GPU] | maq=128 | suelo_local=1,395,319 [↓22M→1395k] | 8s  ETA 391s
   ⚡ INTERPOL chunk 2/95 [GPU] | maq=128 | suelo_local=1,395,319 [↓22M→1395k] | 8s  ETA 391s
   ⚡ INTERPOL chunk 3/95 [GPU] | maq=1,154 | suelo_local=1,395,319 [↓22M→1395k] | 10s  ETA 307s
   ⚡ INTERPOL chunk 3/95 [GPU] | maq=1,154 | suelo_local=1,395,319 [↓22M→1395k] | 10s  ETA 307s
   ⚡ INTERPOL chunk 4/95 [GPU] | maq=609 | suelo_local=1,395,319 [↓22M→1395k] | 12s  ETA 262s
   ⚡ INTERPOL chunk 4/95 [GPU] | maq=609 | suelo_local=1,395,319 [↓22M→1395k] | 12s  ETA 262s
   ⚡ INTERPOL chunk 6/95 [GPU] | maq=235 | suelo_local=1,664,872 [↓26M→1664k] | 14s  ETA 202s
   ⚡ INTERPOL chunk 6/95 [GPU] | maq=235 | suelo_local=1,664,872 [↓26M→1664k] | 14s  ETA 202s
   ⚡ INTERPOL chunk 7/95 [GPU] | maq=414 | suelo_local=1,664,872 [↓26M→1664k] | 16s  ETA 198s
   ⚡ INTERPOL chunk 7/95 [GPU] | maq=414 | suelo_local=1,664,872 [↓26M→1664k] | 16s  ETA 198s
   ⚡ INTERPOL chunk 8/95 [GPU] | maq=429 | suelo_local=1,664,872 [↓26M→1664k] | 18s  ETA 194s
   ⚡ INTERPOL chunk 8/95 [GPU] | maq=429 | suelo_local=1,664,872 [↓26M→1664k] | 18s  ETA 194s
   ⚡ INTERPOL chunk 9/95 [GPU] | maq=5,071 | suelo_local=1,664,872 [↓26M→1664k] | 20s  ETA 191s
   ⚡ INTERPOL chunk 9/95 [GPU] | maq=5,071 | suelo_local=1,664,872 [↓26M→1664k] | 20s  ETA 191s
   ⚡ INTERPOL chunk 10/95 [GPU] | maq=965 | suelo_local=1,664,652 [↓26M→1664k] | 22s  ETA 188s
   ⚡ INTERPOL chunk 10/95 [GPU] | maq=965 | suelo_local=1,664,652 [↓26M→1664k] | 22s  ETA 188s
   ⚡ INTERPOL chunk 11/95 [GPU] | maq=443 | suelo_local=1,851,479 [↓29M→1851k] | 27s  ETA 209s
   ⚡ INTERPOL chunk 11/95 [GPU] | maq=443 | suelo_local=1,851,479 [↓29M→1851k] | 27s  ETA 209s
   ⚡ INTERPOL chunk 12/95 [GPU] | maq=7,021 | suelo_local=1,851,479 [↓29M→1851k] | 30s  ETA 206s
   ⚡ INTERPOL chunk 12/95 [GPU] | maq=7,021 | suelo_local=1,851,479 [↓29M→1851k] | 30s  ETA 206s
   ⚡ INTERPOL chunk 13/95 [GPU] | maq=854 | suelo_local=1,851,479 [↓29M→1851k] | 32s  ETA 203s
   ⚡ INTERPOL chunk 13/95 [GPU] | maq=854 | suelo_local=1,851,479 [↓29M→1851k] | 32s  ETA 203s
   ⚡ INTERPOL chunk 14/95 [GPU] | maq=2,036 | suelo_local=1,851,479 [↓29M→1851k] | 34s  ETA 198s
   ⚡ INTERPOL chunk 14/95 [GPU] | maq=2,036 | suelo_local=1,851,479 [↓29M→1851k] | 34s  ETA 198s
   ⚡ INTERPOL chunk 15/95 [GPU] | maq=3,004 | suelo_local=1,851,257 [↓29M→1851k] | 36s  ETA 194s
   ⚡ INTERPOL chunk 15/95 [GPU] | maq=3,004 | suelo_local=1,851,257 [↓29M→1851k] | 36s  ETA 194s
   ⚡ INTERPOL chunk 16/95 [GPU] | maq=105 | suelo_local=2,110,794 [↓33M→2110k] | 39s  ETA 191s
   ⚡ INTERPOL chunk 16/95 [GPU] | maq=105 | suelo_local=2,110,794 [↓33M→2110k] | 39s  ETA 191s
   ⚡ INTERPOL chunk 17/95 [GPU] | maq=1,008 | suelo_local=2,110,794 [↓33M→2110k] | 41s  ETA 188s
   ⚡ INTERPOL chunk 17/95 [GPU] | maq=1,008 | suelo_local=2,110,794 [↓33M→2110k] | 41s  ETA 188s
   ⚡ INTERPOL chunk 18/95 [GPU] | maq=2,680 | suelo_local=2,110,794 [↓33M→2110k] | 44s  ETA 186s
   ⚡ INTERPOL chunk 18/95 [GPU] | maq=2,680 | suelo_local=2,110,794 [↓33M→2110k] | 44s  ETA 186s
   ⚡ INTERPOL chunk 19/95 [GPU] | maq=6,371 | suelo_local=2,110,794 [↓33M→2110k] | 46s  ETA 185s
   ⚡ INTERPOL chunk 19/95 [GPU] | maq=6,371 | suelo_local=2,110,794 [↓33M→2110k] | 46s  ETA 185s
   ⚡ INTERPOL chunk 20/95 [GPU] | maq=8,208 | suelo_local=2,110,569 [↓33M→2110k] | 49s  ETA 184s
   ⚡ INTERPOL chunk 20/95 [GPU] | maq=8,208 | suelo_local=2,110,569 [↓33M→2110k] | 49s  ETA 184s
   ⚡ INTERPOL chunk 21/95 [GPU] | maq=976 | suelo_local=2,379,219 [↓38M→2379k] | 52s  ETA 183s
   ⚡ INTERPOL chunk 21/95 [GPU] | maq=976 | suelo_local=2,379,219 [↓38M→2379k] | 52s  ETA 183s
   ⚡ INTERPOL chunk 22/95 [GPU] | maq=1,660 | suelo_local=2,379,219 [↓38M→2379k] | 55s  ETA 182s
   ⚡ INTERPOL chunk 22/95 [GPU] | maq=1,660 | suelo_local=2,379,219 [↓38M→2379k] | 55s  ETA 182s
   ⚡ INTERPOL chunk 23/95 [GPU] | maq=6,293 | suelo_local=2,379,219 [↓38M→2379k] | 61s  ETA 190s
   ⚡ INTERPOL chunk 23/95 [GPU] | maq=6,293 | suelo_local=2,379,219 [↓38M→2379k] | 61s  ETA 190s
   ⚡ INTERPOL chunk 24/95 [GPU] | maq=14,766 | suelo_local=2,379,219 [↓38M→2379k] | 64s  ETA 188s
   ⚡ INTERPOL chunk 24/95 [GPU] | maq=14,766 | suelo_local=2,379,219 [↓38M→2379k] | 64s  ETA 188s
   ⚡ INTERPOL chunk 25/95 [GPU] | maq=5,766 | suelo_local=2,378,967 [↓38M→2378k] | 67s  ETA 187s
   ⚡ INTERPOL chunk 25/95 [GPU] | maq=5,766 | suelo_local=2,378,967 [↓38M→2378k] | 67s  ETA 187s
   ⚡ INTERPOL chunk 26/95 [GPU] | maq=9,555 | suelo_local=2,397,956 [↓38M→2397k] | 70s  ETA 185s
   ⚡ INTERPOL chunk 26/95 [GPU] | maq=9,555 | suelo_local=2,397,956 [↓38M→2397k] | 70s  ETA 185s
   ⚡ INTERPOL chunk 27/95 [GPU] | maq=8,575 | suelo_local=2,397,956 [↓38M→2397k] | 73s  ETA 183s
   ⚡ INTERPOL chunk 27/95 [GPU] | maq=8,575 | suelo_local=2,397,956 [↓38M→2397k] | 73s  ETA 183s
   ⚡ INTERPOL chunk 28/95 [GPU] | maq=9,013 | suelo_local=2,397,956 [↓38M→2397k] | 76s  ETA 181s
   ⚡ INTERPOL chunk 28/95 [GPU] | maq=9,013 | suelo_local=2,397,956 [↓38M→2397k] | 76s  ETA 181s
   ⚡ INTERPOL chunk 29/95 [GPU] | maq=5,953 | suelo_local=2,397,956 [↓38M→2397k] | 78s  ETA 178s
   ⚡ INTERPOL chunk 29/95 [GPU] | maq=5,953 | suelo_local=2,397,956 [↓38M→2397k] | 78s  ETA 178s
   ⚡ INTERPOL chunk 30/95 [GPU] | maq=4,418 | suelo_local=2,397,956 [↓38M→2397k] | 81s  ETA 176s
   ⚡ INTERPOL chunk 30/95 [GPU] | maq=4,418 | suelo_local=2,397,956 [↓38M→2397k] | 81s  ETA 176s
   ⚡ INTERPOL chunk 31/95 [GPU] | maq=2,229 | suelo_local=2,380,066 [↓38M→2380k] | 84s  ETA 174s
   ⚡ INTERPOL chunk 31/95 [GPU] | maq=2,229 | suelo_local=2,380,066 [↓38M→2380k] | 84s  ETA 174s
   ⚡ INTERPOL chunk 32/95 [GPU] | maq=17,242 | suelo_local=2,380,066 [↓38M→2380k] | 87s  ETA 171s
   ⚡ INTERPOL chunk 32/95 [GPU] | maq=17,242 | suelo_local=2,380,066 [↓38M→2380k] | 87s  ETA 171s
   ⚡ INTERPOL chunk 33/95 [GPU] | maq=24,408 | suelo_local=2,380,066 [↓38M→2380k] | 93s  ETA 174s
   ⚡ INTERPOL chunk 33/95 [GPU] | maq=24,408 | suelo_local=2,380,066 [↓38M→2380k] | 93s  ETA 174s
   ⚡ INTERPOL chunk 34/95 [GPU] | maq=25,398 | suelo_local=2,380,066 [↓38M→2380k] | 96s  ETA 172s
   ⚡ INTERPOL chunk 34/95 [GPU] | maq=25,398 | suelo_local=2,380,066 [↓38M→2380k] | 96s  ETA 172s
   ⚡ INTERPOL chunk 35/95 [GPU] | maq=24,035 | suelo_local=2,380,066 [↓38M→2380k] | 99s  ETA 169s
   ⚡ INTERPOL chunk 35/95 [GPU] | maq=24,035 | suelo_local=2,380,066 [↓38M→2380k] | 99s  ETA 169s
   ⚡ INTERPOL chunk 36/95 [GPU] | maq=165 | suelo_local=2,364,999 [↓37M→2364k] | 102s  ETA 167s
   ⚡ INTERPOL chunk 36/95 [GPU] | maq=165 | suelo_local=2,364,999 [↓37M→2364k] | 102s  ETA 167s
   ⚡ INTERPOL chunk 37/95 [GPU] | maq=9,483 | suelo_local=2,364,999 [↓37M→2364k] | 105s  ETA 164s
   ⚡ INTERPOL chunk 37/95 [GPU] | maq=9,483 | suelo_local=2,364,999 [↓37M→2364k] | 105s  ETA 164s
   ⚡ INTERPOL chunk 38/95 [GPU] | maq=7,438 | suelo_local=2,364,999 [↓37M→2364k] | 108s  ETA 161s
   ⚡ INTERPOL chunk 38/95 [GPU] | maq=7,438 | suelo_local=2,364,999 [↓37M→2364k] | 108s  ETA 161s
   ⚡ INTERPOL chunk 39/95 [GPU] | maq=1,732 | suelo_local=2,364,999 [↓37M→2364k] | 111s  ETA 159s
   ⚡ INTERPOL chunk 39/95 [GPU] | maq=1,732 | suelo_local=2,364,999 [↓37M→2364k] | 111s  ETA 159s
   ⚡ INTERPOL chunk 40/95 [GPU] | maq=1,712 | suelo_local=2,364,999 [↓37M→2364k] | 113s  ETA 156s
   ⚡ INTERPOL chunk 40/95 [GPU] | maq=1,712 | suelo_local=2,364,999 [↓37M→2364k] | 113s  ETA 156s
   ⚡ INTERPOL chunk 42/95 [GPU] | maq=830 | suelo_local=2,355,390 [↓37M→2355k] | 116s  ETA 147s
   ⚡ INTERPOL chunk 42/95 [GPU] | maq=830 | suelo_local=2,355,390 [↓37M→2355k] | 116s  ETA 147s
   ⚡ INTERPOL chunk 43/95 [GPU] | maq=28 | suelo_local=2,355,390 [↓37M→2355k] | 119s  ETA 144s
   ⚡ INTERPOL chunk 43/95 [GPU] | maq=28 | suelo_local=2,355,390 [↓37M→2355k] | 119s  ETA 144s
   ⚡ INTERPOL chunk 44/95 [GPU] | maq=1,191 | suelo_local=2,355,390 [↓37M→2355k] | 125s  ETA 145s
   ⚡ INTERPOL chunk 44/95 [GPU] | maq=1,191 | suelo_local=2,355,390 [↓37M→2355k] | 125s  ETA 145s
   ⚡ INTERPOL chunk 45/95 [GPU] | maq=6,293 | suelo_local=2,355,390 [↓37M→2355k] | 128s  ETA 142s
   ⚡ INTERPOL chunk 45/95 [GPU] | maq=6,293 | suelo_local=2,355,390 [↓37M→2355k] | 128s  ETA 142s
   ⚡ INTERPOL chunk 46/95 [GPU] | maq=41 | suelo_local=2,352,551 [↓37M→2352k] | 131s  ETA 140s
   ⚡ INTERPOL chunk 46/95 [GPU] | maq=41 | suelo_local=2,352,551 [↓37M→2352k] | 131s  ETA 140s
   ⚡ INTERPOL chunk 47/95 [GPU] | maq=1,602 | suelo_local=2,352,551 [↓37M→2352k] | 134s  ETA 137s
   ⚡ INTERPOL chunk 47/95 [GPU] | maq=1,602 | suelo_local=2,352,551 [↓37M→2352k] | 134s  ETA 137s
   ⚡ INTERPOL chunk 49/95 [GPU] | maq=1,054 | suelo_local=2,352,551 [↓37M→2352k] | 137s  ETA 128s
   ⚡ INTERPOL chunk 49/95 [GPU] | maq=1,054 | suelo_local=2,352,551 [↓37M→2352k] | 137s  ETA 128s
   ⚡ INTERPOL chunk 50/95 [GPU] | maq=6,225 | suelo_local=2,352,551 [↓37M→2352k] | 140s  ETA 126s
   ⚡ INTERPOL chunk 50/95 [GPU] | maq=6,225 | suelo_local=2,352,551 [↓37M→2352k] | 140s  ETA 126s
   ⚡ INTERPOL chunk 51/95 [GPU] | maq=146 | suelo_local=2,356,036 [↓37M→2356k] | 143s  ETA 123s
   ⚡ INTERPOL chunk 51/95 [GPU] | maq=146 | suelo_local=2,356,036 [↓37M→2356k] | 143s  ETA 123s
   ⚡ INTERPOL chunk 52/95 [GPU] | maq=769 | suelo_local=2,356,036 [↓37M→2356k] | 146s  ETA 120s
   ⚡ INTERPOL chunk 52/95 [GPU] | maq=769 | suelo_local=2,356,036 [↓37M→2356k] | 146s  ETA 120s
   ⚡ INTERPOL chunk 54/95 [GPU] | maq=1,310 | suelo_local=2,356,036 [↓37M→2356k] | 148s  ETA 113s
   ⚡ INTERPOL chunk 54/95 [GPU] | maq=1,310 | suelo_local=2,356,036 [↓37M→2356k] | 148s  ETA 113s
   ⚡ INTERPOL chunk 55/95 [GPU] | maq=53 | suelo_local=2,356,036 [↓37M→2356k] | 151s  ETA 110s
   ⚡ INTERPOL chunk 55/95 [GPU] | maq=53 | suelo_local=2,356,036 [↓37M→2356k] | 151s  ETA 110s
   ⚡ INTERPOL chunk 56/95 [GPU] | maq=1,212 | suelo_local=2,436,873 [↓39M→2436k] | 157s  ETA 110s
   ⚡ INTERPOL chunk 56/95 [GPU] | maq=1,212 | suelo_local=2,436,873 [↓39M→2436k] | 157s  ETA 110s
   ⚡ INTERPOL chunk 57/95 [GPU] | maq=9,534 | suelo_local=2,436,873 [↓39M→2436k] | 160s  ETA 107s
   ⚡ INTERPOL chunk 57/95 [GPU] | maq=9,534 | suelo_local=2,436,873 [↓39M→2436k] | 160s  ETA 107s
   ⚡ INTERPOL chunk 59/95 [GPU] | maq=2,383 | suelo_local=2,436,873 [↓39M→2436k] | 163s  ETA 100s
   ⚡ INTERPOL chunk 59/95 [GPU] | maq=2,383 | suelo_local=2,436,873 [↓39M→2436k] | 163s  ETA 100s
   ⚡ INTERPOL chunk 61/95 [GPU] | maq=2,410 | suelo_local=2,435,586 [↓39M→2435k] | 166s  ETA 93s
   ⚡ INTERPOL chunk 61/95 [GPU] | maq=2,410 | suelo_local=2,435,586 [↓39M→2435k] | 166s  ETA 93s
   ⚡ INTERPOL chunk 62/95 [GPU] | maq=11,942 | suelo_local=2,435,586 [↓39M→2435k] | 170s  ETA 90s
   ⚡ INTERPOL chunk 62/95 [GPU] | maq=11,942 | suelo_local=2,435,586 [↓39M→2435k] | 170s  ETA 90s
   ⚡ INTERPOL chunk 63/95 [GPU] | maq=8,159 | suelo_local=2,435,586 [↓39M→2435k] | 173s  ETA 88s
   ⚡ INTERPOL chunk 63/95 [GPU] | maq=8,159 | suelo_local=2,435,586 [↓39M→2435k] | 173s  ETA 88s
   ⚡ INTERPOL chunk 64/95 [GPU] | maq=8,609 | suelo_local=2,435,586 [↓39M→2435k] | 175s  ETA 85s
   ⚡ INTERPOL chunk 64/95 [GPU] | maq=8,609 | suelo_local=2,435,586 [↓39M→2435k] | 175s  ETA 85s
   ⚡ INTERPOL chunk 65/95 [GPU] | maq=304 | suelo_local=2,435,586 [↓39M→2435k] | 179s  ETA 82s
   ⚡ INTERPOL chunk 65/95 [GPU] | maq=304 | suelo_local=2,435,586 [↓39M→2435k] | 179s  ETA 82s
   ⚡ INTERPOL chunk 66/95 [GPU] | maq=3,529 | suelo_local=2,333,327 [↓37M→2333k] | 181s  ETA 80s
   ⚡ INTERPOL chunk 66/95 [GPU] | maq=3,529 | suelo_local=2,333,327 [↓37M→2333k] | 181s  ETA 80s
   ⚡ INTERPOL chunk 67/95 [GPU] | maq=5,081 | suelo_local=2,333,327 [↓37M→2333k] | 184s  ETA 77s
   ⚡ INTERPOL chunk 67/95 [GPU] | maq=5,081 | suelo_local=2,333,327 [↓37M→2333k] | 184s  ETA 77s
   ⚡ INTERPOL chunk 68/95 [GPU] | maq=14,964 | suelo_local=2,333,327 [↓37M→2333k] | 190s  ETA 76s
   ⚡ INTERPOL chunk 68/95 [GPU] | maq=14,964 | suelo_local=2,333,327 [↓37M→2333k] | 190s  ETA 76s
   ⚡ INTERPOL chunk 69/95 [GPU] | maq=6,317 | suelo_local=2,333,327 [↓37M→2333k] | 193s  ETA 73s
   ⚡ INTERPOL chunk 69/95 [GPU] | maq=6,317 | suelo_local=2,333,327 [↓37M→2333k] | 193s  ETA 73s
   ⚡ INTERPOL chunk 70/95 [GPU] | maq=749 | suelo_local=2,333,327 [↓37M→2333k] | 196s  ETA 70s
   ⚡ INTERPOL chunk 70/95 [GPU] | maq=749 | suelo_local=2,333,327 [↓37M→2333k] | 196s  ETA 70s
   ⚡ INTERPOL chunk 71/95 [GPU] | maq=943 | suelo_local=2,084,090 [↓33M→2084k] | 199s  ETA 67s
   ⚡ INTERPOL chunk 71/95 [GPU] | maq=943 | suelo_local=2,084,090 [↓33M→2084k] | 199s  ETA 67s
   ⚡ INTERPOL chunk 72/95 [GPU] | maq=2,525 | suelo_local=2,084,090 [↓33M→2084k] | 201s  ETA 64s
   ⚡ INTERPOL chunk 72/95 [GPU] | maq=2,525 | suelo_local=2,084,090 [↓33M→2084k] | 201s  ETA 64s
   ⚡ INTERPOL chunk 73/95 [GPU] | maq=6,786 | suelo_local=2,084,090 [↓33M→2084k] | 204s  ETA 61s
   ⚡ INTERPOL chunk 73/95 [GPU] | maq=6,786 | suelo_local=2,084,090 [↓33M→2084k] | 204s  ETA 61s
   ⚡ INTERPOL chunk 74/95 [GPU] | maq=907 | suelo_local=2,084,090 [↓33M→2084k] | 207s  ETA 59s
   ⚡ INTERPOL chunk 74/95 [GPU] | maq=907 | suelo_local=2,084,090 [↓33M→2084k] | 207s  ETA 59s
   ⚡ INTERPOL chunk 75/95 [GPU] | maq=6,269 | suelo_local=2,084,090 [↓33M→2084k] | 209s  ETA 56s
   ⚡ INTERPOL chunk 75/95 [GPU] | maq=6,269 | suelo_local=2,084,090 [↓33M→2084k] | 209s  ETA 56s
   ⚡ INTERPOL chunk 76/95 [GPU] | maq=295 | suelo_local=1,812,792 [↓29M→1812k] | 211s  ETA 53s
   ⚡ INTERPOL chunk 76/95 [GPU] | maq=295 | suelo_local=1,812,792 [↓29M→1812k] | 211s  ETA 53s
   ⚡ INTERPOL chunk 77/95 [GPU] | maq=1,965 | suelo_local=1,812,792 [↓29M→1812k] | 213s  ETA 50s
   ⚡ INTERPOL chunk 77/95 [GPU] | maq=1,965 | suelo_local=1,812,792 [↓29M→1812k] | 213s  ETA 50s
   ⚡ INTERPOL chunk 78/95 [GPU] | maq=2,643 | suelo_local=1,812,792 [↓29M→1812k] | 216s  ETA 47s
   ⚡ INTERPOL chunk 78/95 [GPU] | maq=2,643 | suelo_local=1,812,792 [↓29M→1812k] | 216s  ETA 47s
   ⚡ INTERPOL chunk 79/95 [GPU] | maq=9,172 | suelo_local=1,812,792 [↓29M→1812k] | 218s  ETA 44s
   ⚡ INTERPOL chunk 79/95 [GPU] | maq=9,172 | suelo_local=1,812,792 [↓29M→1812k] | 218s  ETA 44s
   ⚡ INTERPOL chunk 80/95 [GPU] | maq=1,709 | suelo_local=1,812,792 [↓29M→1812k] | 223s  ETA 42s
   ⚡ INTERPOL chunk 80/95 [GPU] | maq=1,709 | suelo_local=1,812,792 [↓29M→1812k] | 223s  ETA 42s
   ⚡ INTERPOL chunk 81/95 [GPU] | maq=709 | suelo_local=1,542,221 [↓24M→1542k] | 225s  ETA 39s
   ⚡ INTERPOL chunk 81/95 [GPU] | maq=709 | suelo_local=1,542,221 [↓24M→1542k] | 225s  ETA 39s
   ⚡ INTERPOL chunk 82/95 [GPU] | maq=8,021 | suelo_local=1,542,221 [↓24M→1542k] | 226s  ETA 36s
   ⚡ INTERPOL chunk 82/95 [GPU] | maq=8,021 | suelo_local=1,542,221 [↓24M→1542k] | 226s  ETA 36s
   ⚡ INTERPOL chunk 83/95 [GPU] | maq=19,915 | suelo_local=1,542,221 [↓24M→1542k] | 228s  ETA 33s
   ⚡ INTERPOL chunk 83/95 [GPU] | maq=19,915 | suelo_local=1,542,221 [↓24M→1542k] | 228s  ETA 33s
   ⚡ INTERPOL chunk 84/95 [GPU] | maq=19,357 | suelo_local=1,542,221 [↓24M→1542k] | 230s  ETA 30s
   ⚡ INTERPOL chunk 84/95 [GPU] | maq=19,357 | suelo_local=1,542,221 [↓24M→1542k] | 230s  ETA 30s
   ⚡ INTERPOL chunk 85/95 [GPU] | maq=10,564 | suelo_local=1,542,221 [↓24M→1542k] | 232s  ETA 27s
   ⚡ INTERPOL chunk 85/95 [GPU] | maq=10,564 | suelo_local=1,542,221 [↓24M→1542k] | 232s  ETA 27s
   ⚡ INTERPOL chunk 86/95 [GPU] | maq=13,465 | suelo_local=1,271,047 [↓20M→1271k] | 233s  ETA 24s
   ⚡ INTERPOL chunk 86/95 [GPU] | maq=13,465 | suelo_local=1,271,047 [↓20M→1271k] | 233s  ETA 24s
   ⚡ INTERPOL chunk 87/95 [GPU] | maq=30,016 | suelo_local=1,271,047 [↓20M→1271k] | 235s  ETA 22s
   ⚡ INTERPOL chunk 87/95 [GPU] | maq=30,016 | suelo_local=1,271,047 [↓20M→1271k] | 235s  ETA 22s
   ⚡ INTERPOL chunk 88/95 [GPU] | maq=3,723 | suelo_local=1,271,047 [↓20M→1271k] | 236s  ETA 19s
   ⚡ INTERPOL chunk 88/95 [GPU] | maq=3,723 | suelo_local=1,271,047 [↓20M→1271k] | 236s  ETA 19s
   ⚡ INTERPOL chunk 89/95 [GPU] | maq=1 | suelo_local=1,271,047 [↓20M→1271k] | 238s  ETA 16s
   ⚡ INTERPOL chunk 89/95 [GPU] | maq=1 | suelo_local=1,271,047 [↓20M→1271k] | 238s  ETA 16s
   ⚡ INTERPOL chunk 91/95 [GPU] | maq=1,278 | suelo_local=996,874 [↓16M→996k] | 239s  ETA 11s
   ⚡ INTERPOL chunk 91/95 [GPU] | maq=1,278 | suelo_local=996,874 [↓16M→996k] | 239s  ETA 11s
   📊 INTERPOL backend: 84 tiles GPU, 0 tiles CPU fallback
   📊 INTERPOL backend: 84 tiles GPU, 0 tiles CPU fallback
   📊 Z diagnostico: 412,485/456,931 puntos con dZ>1cm | dZ medio=0.431m | dZ max=9.848m
   📊 Z diagnostico: 412,485/456,931 puntos con dZ>1cm | dZ medio=0.431m | dZ max=9.848m
   ✅ Aplanados 456,931 puntos
   ✅ INTERPOL: 456,931 puntos aplanados
💾 DTM guardado en 247.8s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 247.8s
