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
   RAM Disponible: 60.14 GB
   RAM Usada: 4.1%
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
🎯 INICIANDO INFERENCIA: LINK_260226_LOG176_NDP_PTL_edit_RGB.laz
======================================================================
   🔍 [Inicio de inferencia] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   📦 Tamaño del archivo: 683.65 MB
   🔍 [Antes de extraer features] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   🧮 Calculando normales en chunks espaciales (r=3.5m, ~50m x 50m por chunk)...
   🔥 Normales: usando GPU (Open3D Tensor CUDA)
   📐 Nube: 213,816,052 puntos → ~30 chunks (10×3) de 500m
   ⚡ Chunk 1/30 | core=8,410,003 pts | 2s elapsed  ETA 47s
   ⚡ Chunk 2/30 | core=10,457,324 pts | 3s elapsed  ETA 42s
   ⚡ Chunk 3/30 | core=3,797,463 pts | 4s elapsed  ETA 34s
   ⚡ Chunk 4/30 | core=9,087,355 pts | 6s elapsed  ETA 41s
   ⚡ Chunk 5/30 | core=11,144,821 pts | 8s elapsed  ETA 38s
   ⚡ Chunk 6/30 | core=5,221,854 pts | 9s elapsed  ETA 34s
   ⚡ Chunk 7/30 | core=8,859,779 pts | 10s elapsed  ETA 32s
   ⚡ Chunk 8/30 | core=11,216,628 pts | 11s elapsed  ETA 30s
   ⚡ Chunk 9/30 | core=4,889,617 pts | 12s elapsed  ETA 28s
   ⚡ Chunk 10/30 | core=7,515,096 pts | 13s elapsed  ETA 26s
   ⚡ Chunk 11/30 | core=8,368,051 pts | 14s elapsed  ETA 24s
   ⚡ Chunk 12/30 | core=4,010,019 pts | 15s elapsed  ETA 22s
   ⚡ Chunk 13/30 | core=9,115,662 pts | 16s elapsed  ETA 21s
   ⚡ Chunk 14/30 | core=11,190,812 pts | 17s elapsed  ETA 20s
   ⚡ Chunk 15/30 | core=4,458,511 pts | 18s elapsed  ETA 18s
   ⚡ Chunk 16/30 | core=8,988,879 pts | 19s elapsed  ETA 17s
   ⚡ Chunk 17/30 | core=11,156,099 pts | 21s elapsed  ETA 16s
   ⚡ Chunk 18/30 | core=4,301,676 pts | 21s elapsed  ETA 14s
   ⚡ Chunk 19/30 | core=8,955,064 pts | 23s elapsed  ETA 13s
   ⚡ Chunk 20/30 | core=11,202,396 pts | 24s elapsed  ETA 12s
   ⚡ Chunk 21/30 | core=4,556,490 pts | 25s elapsed  ETA 11s
   ⚡ Chunk 22/30 | core=8,925,799 pts | 26s elapsed  ETA 9s
   ⚡ Chunk 23/30 | core=11,207,877 pts | 27s elapsed  ETA 8s
   ⚡ Chunk 24/30 | core=4,492,313 pts | 28s elapsed  ETA 7s
   ⚡ Chunk 25/30 | core=8,422,511 pts | 29s elapsed  ETA 6s
   ⚡ Chunk 26/30 | core=9,811,102 pts | 31s elapsed  ETA 5s
   ⚡ Chunk 27/30 | core=2,282,438 pts | 31s elapsed  ETA 3s
   ⚡ Chunk 28/30 | core=1,710,944 pts | 32s elapsed  ETA 2s
   ⚡ Chunk 29/30 | core=59,469 pts | 33s elapsed  ETA 1s
   ✅ Normales completadas: 33.0s  (6,486,802 pts/s)
   💾 Array de features: 7340.79 MB en RAM
   🔍 [Después de extraer features] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   → 48963 bloques activos
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
   🔍 [Después de inferencia GPU] GPU Memory: Usada=0.02GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
   🔍 [Final de inferencia] GPU Memory: Usada=0.02GB, Reservada=10.07GB, Libre=21.78GB, Total=31.84GB
✅ Inferencia completada en 224.0s - Maquinaria: 843,366 puntos (0.4%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 40990 MB
   🚜 Maquinaria: 843,366 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   🔍 Smart Merge: 212,952,761 candidatos en 426 bloques
   🔍 Smart Merge: 212,952,761 candidatos
   ✨ Smart Merge: 3,812,790 puntos unidos
   🔢 Objetos encontrados: 1979
   ⚡ Procesando en paralelo 1979 objetos...
   ✅ Rellenados 2,028,857 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6_techos.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6_techos.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6_techos.laz
   📊 RSS al iniciar INTERPOL: 40990 MB
   📉 Maquinaria: 6,685,013 pts | Suelo: 207,131,039 pts | RAM arrays: 2651 MB
   📉 Maquinaria: 6,685,013 | Suelo: 207,131,039 | Total: 213,816,052 | RAM: 2651 MB
   📐 Altura: mediana_maq=1604.45m, mediana_suelo=1604.18m, gap=0.27m
   📐 Altura: mediana_maq=1604.45m, mediana_suelo=1604.18m, gap=0.27m
   ⚠️ Gap maquinaria-suelo < 0.5m — IDW producirá cambios mínimos en Z
   ⚠️ Gap maquinaria-suelo < 0.5m — IDW producirá cambios mínimos en Z
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL chunk 1/95 | maq=25,753 | suelo_local=3,864,757 | 12s  ETA 1156s
   ⚡ INTERPOL chunk 1/95 | maq=25,753 | suelo_local=3,864,757 | 12s  ETA 1156s
   ⚡ INTERPOL chunk 2/95 | maq=10,861 | suelo_local=6,413,421 | 14s  ETA 641s
   ⚡ INTERPOL chunk 2/95 | maq=10,861 | suelo_local=6,413,421 | 14s  ETA 641s
   ⚡ INTERPOL chunk 3/95 | maq=19,486 | suelo_local=6,738,994 | 15s  ETA 472s
   ⚡ INTERPOL chunk 3/95 | maq=19,486 | suelo_local=6,738,994 | 15s  ETA 472s
   ⚡ INTERPOL chunk 4/95 | maq=16,779 | suelo_local=6,427,248 | 17s  ETA 384s
   ⚡ INTERPOL chunk 4/95 | maq=16,779 | suelo_local=6,427,248 | 17s  ETA 384s
   ⚡ INTERPOL chunk 5/95 | maq=664 | suelo_local=3,765,790 | 18s  ETA 321s
   ⚡ INTERPOL chunk 5/95 | maq=664 | suelo_local=3,765,790 | 18s  ETA 321s
   ⚡ INTERPOL chunk 6/95 | maq=11,093 | suelo_local=5,337,737 | 19s  ETA 284s
   ⚡ INTERPOL chunk 6/95 | maq=11,093 | suelo_local=5,337,737 | 19s  ETA 284s
   ⚡ INTERPOL chunk 7/95 | maq=3,976 | suelo_local=8,848,410 | 21s  ETA 266s
   ⚡ INTERPOL chunk 7/95 | maq=3,976 | suelo_local=8,848,410 | 21s  ETA 266s
   ⚡ INTERPOL chunk 8/95 | maq=841 | suelo_local=8,952,025 | 25s  ETA 268s
   ⚡ INTERPOL chunk 8/95 | maq=841 | suelo_local=8,952,025 | 25s  ETA 268s
   ⚡ INTERPOL chunk 9/95 | maq=54,071 | suelo_local=8,878,316 | 27s  ETA 254s
   ⚡ INTERPOL chunk 9/95 | maq=54,071 | suelo_local=8,878,316 | 27s  ETA 254s
   ⚡ INTERPOL chunk 10/95 | maq=21,636 | suelo_local=6,531,560 | 28s  ETA 238s
   ⚡ INTERPOL chunk 10/95 | maq=21,636 | suelo_local=6,531,560 | 28s  ETA 238s
   ⚡ INTERPOL chunk 11/95 | maq=25,563 | suelo_local=5,333,242 | 29s  ETA 224s
   ⚡ INTERPOL chunk 11/95 | maq=25,563 | suelo_local=5,333,242 | 29s  ETA 224s
   ⚡ INTERPOL chunk 12/95 | maq=155,647 | suelo_local=8,834,052 | 31s  ETA 216s
   ⚡ INTERPOL chunk 12/95 | maq=155,647 | suelo_local=8,834,052 | 31s  ETA 216s
   ⚡ INTERPOL chunk 13/95 | maq=28,955 | suelo_local=8,938,646 | 33s  ETA 210s
   ⚡ INTERPOL chunk 13/95 | maq=28,955 | suelo_local=8,938,646 | 33s  ETA 210s
   ⚡ INTERPOL chunk 14/95 | maq=59,423 | suelo_local=8,874,065 | 35s  ETA 204s
   ⚡ INTERPOL chunk 14/95 | maq=59,423 | suelo_local=8,874,065 | 35s  ETA 204s
   ⚡ INTERPOL chunk 15/95 | maq=81,662 | suelo_local=6,385,279 | 37s  ETA 196s
   ⚡ INTERPOL chunk 15/95 | maq=81,662 | suelo_local=6,385,279 | 37s  ETA 196s
   ⚡ INTERPOL chunk 16/95 | maq=4,026 | suelo_local=5,037,669 | 38s  ETA 187s
   ⚡ INTERPOL chunk 16/95 | maq=4,026 | suelo_local=5,037,669 | 38s  ETA 187s
   ⚡ INTERPOL chunk 17/95 | maq=18,390 | suelo_local=8,935,650 | 40s  ETA 183s
   ⚡ INTERPOL chunk 17/95 | maq=18,390 | suelo_local=8,935,650 | 40s  ETA 183s
   ⚡ INTERPOL chunk 18/95 | maq=93,517 | suelo_local=8,803,133 | 42s  ETA 179s
   ⚡ INTERPOL chunk 18/95 | maq=93,517 | suelo_local=8,803,133 | 42s  ETA 179s
   ⚡ INTERPOL chunk 19/95 | maq=73,689 | suelo_local=8,688,651 | 44s  ETA 175s
   ⚡ INTERPOL chunk 19/95 | maq=73,689 | suelo_local=8,688,651 | 44s  ETA 175s
   ⚡ INTERPOL chunk 20/95 | maq=185,708 | suelo_local=6,203,709 | 45s  ETA 170s
   ⚡ INTERPOL chunk 20/95 | maq=185,708 | suelo_local=6,203,709 | 45s  ETA 170s
   ⚡ INTERPOL chunk 21/95 | maq=49,128 | suelo_local=4,870,338 | 47s  ETA 164s
   ⚡ INTERPOL chunk 21/95 | maq=49,128 | suelo_local=4,870,338 | 47s  ETA 164s
   ⚡ INTERPOL chunk 22/95 | maq=36,644 | suelo_local=8,936,624 | 48s  ETA 161s
   ⚡ INTERPOL chunk 22/95 | maq=36,644 | suelo_local=8,936,624 | 48s  ETA 161s
   ⚡ INTERPOL chunk 23/95 | maq=130,450 | suelo_local=8,637,853 | 50s  ETA 158s
   ⚡ INTERPOL chunk 23/95 | maq=130,450 | suelo_local=8,637,853 | 50s  ETA 158s
   ⚡ INTERPOL chunk 24/95 | maq=214,936 | suelo_local=8,554,584 | 52s  ETA 155s
   ⚡ INTERPOL chunk 24/95 | maq=214,936 | suelo_local=8,554,584 | 52s  ETA 155s
   ⚡ INTERPOL chunk 25/95 | maq=100,167 | suelo_local=6,111,727 | 54s  ETA 151s
   ⚡ INTERPOL chunk 25/95 | maq=100,167 | suelo_local=6,111,727 | 54s  ETA 151s
   ⚡ INTERPOL chunk 26/95 | maq=160,629 | suelo_local=4,424,925 | 56s  ETA 150s
   ⚡ INTERPOL chunk 26/95 | maq=160,629 | suelo_local=4,424,925 | 56s  ETA 150s
   ⚡ INTERPOL chunk 27/95 | maq=176,518 | suelo_local=7,773,558 | 58s  ETA 147s
   ⚡ INTERPOL chunk 27/95 | maq=176,518 | suelo_local=7,773,558 | 58s  ETA 147s
   ⚡ INTERPOL chunk 28/95 | maq=137,315 | suelo_local=7,420,388 | 60s  ETA 143s
   ⚡ INTERPOL chunk 28/95 | maq=137,315 | suelo_local=7,420,388 | 60s  ETA 143s
   ⚡ INTERPOL chunk 29/95 | maq=110,181 | suelo_local=8,438,071 | 62s  ETA 141s
   ⚡ INTERPOL chunk 29/95 | maq=110,181 | suelo_local=8,438,071 | 62s  ETA 141s
   ⚡ INTERPOL chunk 30/95 | maq=178,354 | suelo_local=5,608,246 | 63s  ETA 137s
   ⚡ INTERPOL chunk 30/95 | maq=178,354 | suelo_local=5,608,246 | 63s  ETA 137s
   ⚡ INTERPOL chunk 31/95 | maq=36,940 | suelo_local=4,241,854 | 64s  ETA 133s
   ⚡ INTERPOL chunk 31/95 | maq=36,940 | suelo_local=4,241,854 | 64s  ETA 133s
   ⚡ INTERPOL chunk 32/95 | maq=173,773 | suelo_local=5,715,825 | 66s  ETA 129s
   ⚡ INTERPOL chunk 32/95 | maq=173,773 | suelo_local=5,715,825 | 66s  ETA 129s
   ⚡ INTERPOL chunk 33/95 | maq=274,873 | suelo_local=4,564,085 | 67s  ETA 126s
   ⚡ INTERPOL chunk 33/95 | maq=274,873 | suelo_local=4,564,085 | 67s  ETA 126s
   ⚡ INTERPOL chunk 34/95 | maq=196,922 | suelo_local=7,791,606 | 69s  ETA 123s
   ⚡ INTERPOL chunk 34/95 | maq=196,922 | suelo_local=7,791,606 | 69s  ETA 123s
   ⚡ INTERPOL chunk 35/95 | maq=244,595 | suelo_local=5,097,671 | 70s  ETA 120s
   ⚡ INTERPOL chunk 35/95 | maq=244,595 | suelo_local=5,097,671 | 70s  ETA 120s
   ⚡ INTERPOL chunk 36/95 | maq=22,763 | suelo_local=4,993,868 | 71s  ETA 117s
   ⚡ INTERPOL chunk 36/95 | maq=22,763 | suelo_local=4,993,868 | 71s  ETA 117s
   ⚡ INTERPOL chunk 37/95 | maq=116,040 | suelo_local=7,294,899 | 73s  ETA 114s
   ⚡ INTERPOL chunk 37/95 | maq=116,040 | suelo_local=7,294,899 | 73s  ETA 114s
   ⚡ INTERPOL chunk 38/95 | maq=240,338 | suelo_local=6,611,342 | 74s  ETA 111s
   ⚡ INTERPOL chunk 38/95 | maq=240,338 | suelo_local=6,611,342 | 74s  ETA 111s
   ⚡ INTERPOL chunk 39/95 | maq=123,065 | suelo_local=8,389,292 | 76s  ETA 109s
   ⚡ INTERPOL chunk 39/95 | maq=123,065 | suelo_local=8,389,292 | 76s  ETA 109s
   ⚡ INTERPOL chunk 40/95 | maq=77,148 | suelo_local=5,372,810 | 77s  ETA 106s
   ⚡ INTERPOL chunk 40/95 | maq=77,148 | suelo_local=5,372,810 | 77s  ETA 106s
   ⚡ INTERPOL chunk 41/95 | maq=731 | suelo_local=5,285,737 | 79s  ETA 104s
   ⚡ INTERPOL chunk 41/95 | maq=731 | suelo_local=5,285,737 | 79s  ETA 104s
   ⚡ INTERPOL chunk 42/95 | maq=31,735 | suelo_local=8,969,166 | 81s  ETA 102s
   ⚡ INTERPOL chunk 42/95 | maq=31,735 | suelo_local=8,969,166 | 81s  ETA 102s
   ⚡ INTERPOL chunk 43/95 | maq=22,122 | suelo_local=8,914,306 | 83s  ETA 100s
   ⚡ INTERPOL chunk 43/95 | maq=22,122 | suelo_local=8,914,306 | 83s  ETA 100s
   ⚡ INTERPOL chunk 44/95 | maq=33,708 | suelo_local=8,930,193 | 85s  ETA 98s
   ⚡ INTERPOL chunk 44/95 | maq=33,708 | suelo_local=8,930,193 | 85s  ETA 98s
   ⚡ INTERPOL chunk 45/95 | maq=89,470 | suelo_local=5,583,303 | 87s  ETA 97s
   ⚡ INTERPOL chunk 45/95 | maq=89,470 | suelo_local=5,583,303 | 87s  ETA 97s
   ⚡ INTERPOL chunk 46/95 | maq=1,120 | suelo_local=5,220,297 | 89s  ETA 94s
   ⚡ INTERPOL chunk 46/95 | maq=1,120 | suelo_local=5,220,297 | 89s  ETA 94s
   ⚡ INTERPOL chunk 47/95 | maq=24,793 | suelo_local=9,009,398 | 91s  ETA 93s
   ⚡ INTERPOL chunk 47/95 | maq=24,793 | suelo_local=9,009,398 | 91s  ETA 93s
   ⚡ INTERPOL chunk 49/95 | maq=30,862 | suelo_local=8,977,386 | 93s  ETA 87s
   ⚡ INTERPOL chunk 49/95 | maq=30,862 | suelo_local=8,977,386 | 93s  ETA 87s
   ⚡ INTERPOL chunk 50/95 | maq=199,681 | suelo_local=5,708,581 | 94s  ETA 85s
   ⚡ INTERPOL chunk 50/95 | maq=199,681 | suelo_local=5,708,581 | 94s  ETA 85s
   ⚡ INTERPOL chunk 51/95 | maq=2,926 | suelo_local=5,171,834 | 95s  ETA 82s
   ⚡ INTERPOL chunk 51/95 | maq=2,926 | suelo_local=5,171,834 | 95s  ETA 82s
   ⚡ INTERPOL chunk 52/95 | maq=19,953 | suelo_local=8,992,322 | 97s  ETA 80s
   ⚡ INTERPOL chunk 52/95 | maq=19,953 | suelo_local=8,992,322 | 97s  ETA 80s
   ⚡ INTERPOL chunk 54/95 | maq=23,963 | suelo_local=8,977,448 | 99s  ETA 75s
   ⚡ INTERPOL chunk 54/95 | maq=23,963 | suelo_local=8,977,448 | 99s  ETA 75s
   ⚡ INTERPOL chunk 55/95 | maq=10,104 | suelo_local=5,751,870 | 101s  ETA 73s
   ⚡ INTERPOL chunk 55/95 | maq=10,104 | suelo_local=5,751,870 | 101s  ETA 73s
   ⚡ INTERPOL chunk 56/95 | maq=31,487 | suelo_local=4,971,493 | 102s  ETA 71s
   ⚡ INTERPOL chunk 56/95 | maq=31,487 | suelo_local=4,971,493 | 102s  ETA 71s
   ⚡ INTERPOL chunk 57/95 | maq=94,687 | suelo_local=8,839,875 | 104s  ETA 69s
   ⚡ INTERPOL chunk 57/95 | maq=94,687 | suelo_local=8,839,875 | 104s  ETA 69s
   ⚡ INTERPOL chunk 58/95 | maq=6 | suelo_local=8,977,195 | 106s  ETA 67s
   ⚡ INTERPOL chunk 58/95 | maq=6 | suelo_local=8,977,195 | 106s  ETA 67s
   ⚡ INTERPOL chunk 59/95 | maq=70,621 | suelo_local=8,916,656 | 108s  ETA 66s
   ⚡ INTERPOL chunk 59/95 | maq=70,621 | suelo_local=8,916,656 | 108s  ETA 66s
   ⚡ INTERPOL chunk 60/95 | maq=577 | suelo_local=5,745,836 | 109s  ETA 64s
   ⚡ INTERPOL chunk 60/95 | maq=577 | suelo_local=5,745,836 | 109s  ETA 64s
   ⚡ INTERPOL chunk 61/95 | maq=65,734 | suelo_local=4,845,191 | 110s  ETA 62s
   ⚡ INTERPOL chunk 61/95 | maq=65,734 | suelo_local=4,845,191 | 110s  ETA 62s
   ⚡ INTERPOL chunk 62/95 | maq=139,364 | suelo_local=8,643,546 | 112s  ETA 60s
   ⚡ INTERPOL chunk 62/95 | maq=139,364 | suelo_local=8,643,546 | 112s  ETA 60s
   ⚡ INTERPOL chunk 63/95 | maq=95,943 | suelo_local=8,699,091 | 114s  ETA 58s
   ⚡ INTERPOL chunk 63/95 | maq=95,943 | suelo_local=8,699,091 | 114s  ETA 58s
   ⚡ INTERPOL chunk 64/95 | maq=129,405 | suelo_local=8,754,055 | 116s  ETA 56s
   ⚡ INTERPOL chunk 64/95 | maq=129,405 | suelo_local=8,754,055 | 116s  ETA 56s
   ⚡ INTERPOL chunk 65/95 | maq=7,714 | suelo_local=5,883,973 | 119s  ETA 55s
   ⚡ INTERPOL chunk 65/95 | maq=7,714 | suelo_local=5,883,973 | 119s  ETA 55s
   ⚡ INTERPOL chunk 66/95 | maq=40,710 | suelo_local=5,013,508 | 120s  ETA 53s
   ⚡ INTERPOL chunk 66/95 | maq=40,710 | suelo_local=5,013,508 | 120s  ETA 53s
   ⚡ INTERPOL chunk 67/95 | maq=127,555 | suelo_local=8,720,628 | 122s  ETA 51s
   ⚡ INTERPOL chunk 67/95 | maq=127,555 | suelo_local=8,720,628 | 122s  ETA 51s
   ⚡ INTERPOL chunk 68/95 | maq=174,268 | suelo_local=8,607,059 | 124s  ETA 49s
   ⚡ INTERPOL chunk 68/95 | maq=174,268 | suelo_local=8,607,059 | 124s  ETA 49s
   ⚡ INTERPOL chunk 69/95 | maq=118,481 | suelo_local=8,743,163 | 126s  ETA 47s
   ⚡ INTERPOL chunk 69/95 | maq=118,481 | suelo_local=8,743,163 | 126s  ETA 47s
   ⚡ INTERPOL chunk 70/95 | maq=17,153 | suelo_local=5,980,627 | 127s  ETA 45s
   ⚡ INTERPOL chunk 70/95 | maq=17,153 | suelo_local=5,980,627 | 127s  ETA 45s
   ⚡ INTERPOL chunk 71/95 | maq=55,856 | suelo_local=5,078,530 | 129s  ETA 43s
   ⚡ INTERPOL chunk 71/95 | maq=55,856 | suelo_local=5,078,530 | 129s  ETA 43s
   ⚡ INTERPOL chunk 72/95 | maq=40,348 | suelo_local=8,909,476 | 131s  ETA 42s
   ⚡ INTERPOL chunk 72/95 | maq=40,348 | suelo_local=8,909,476 | 131s  ETA 42s
   ⚡ INTERPOL chunk 73/95 | maq=85,021 | suelo_local=8,857,519 | 132s  ETA 40s
   ⚡ INTERPOL chunk 73/95 | maq=85,021 | suelo_local=8,857,519 | 132s  ETA 40s
   ⚡ INTERPOL chunk 74/95 | maq=21,993 | suelo_local=8,848,123 | 134s  ETA 38s
   ⚡ INTERPOL chunk 74/95 | maq=21,993 | suelo_local=8,848,123 | 134s  ETA 38s
   ⚡ INTERPOL chunk 75/95 | maq=132,786 | suelo_local=5,801,683 | 136s  ETA 36s
   ⚡ INTERPOL chunk 75/95 | maq=132,786 | suelo_local=5,801,683 | 136s  ETA 36s
   ⚡ INTERPOL chunk 76/95 | maq=14,310 | suelo_local=4,851,664 | 137s  ETA 34s
   ⚡ INTERPOL chunk 76/95 | maq=14,310 | suelo_local=4,851,664 | 137s  ETA 34s
   ⚡ INTERPOL chunk 77/95 | maq=54,624 | suelo_local=8,843,998 | 139s  ETA 32s
   ⚡ INTERPOL chunk 77/95 | maq=54,624 | suelo_local=8,843,998 | 139s  ETA 32s
   ⚡ INTERPOL chunk 78/95 | maq=40,165 | suelo_local=8,852,374 | 141s  ETA 31s
   ⚡ INTERPOL chunk 78/95 | maq=40,165 | suelo_local=8,852,374 | 141s  ETA 31s
   ⚡ INTERPOL chunk 79/95 | maq=148,805 | suelo_local=8,782,783 | 143s  ETA 29s
   ⚡ INTERPOL chunk 79/95 | maq=148,805 | suelo_local=8,782,783 | 143s  ETA 29s
   ⚡ INTERPOL chunk 80/95 | maq=25,101 | suelo_local=5,650,562 | 144s  ETA 27s
   ⚡ INTERPOL chunk 80/95 | maq=25,101 | suelo_local=5,650,562 | 144s  ETA 27s
   ⚡ INTERPOL chunk 81/95 | maq=49,062 | suelo_local=4,666,361 | 145s  ETA 25s
   ⚡ INTERPOL chunk 81/95 | maq=49,062 | suelo_local=4,666,361 | 145s  ETA 25s
   ⚡ INTERPOL chunk 82/95 | maq=82,506 | suelo_local=8,748,475 | 147s  ETA 23s
   ⚡ INTERPOL chunk 82/95 | maq=82,506 | suelo_local=8,748,475 | 147s  ETA 23s
   ⚡ INTERPOL chunk 83/95 | maq=163,066 | suelo_local=8,645,005 | 151s  ETA 22s
   ⚡ INTERPOL chunk 83/95 | maq=163,066 | suelo_local=8,645,005 | 151s  ETA 22s
   ⚡ INTERPOL chunk 84/95 | maq=131,040 | suelo_local=8,439,973 | 153s  ETA 20s
   ⚡ INTERPOL chunk 84/95 | maq=131,040 | suelo_local=8,439,973 | 153s  ETA 20s
   ⚡ INTERPOL chunk 85/95 | maq=70,412 | suelo_local=4,717,825 | 154s  ETA 18s
   ⚡ INTERPOL chunk 85/95 | maq=70,412 | suelo_local=4,717,825 | 154s  ETA 18s
   ⚡ INTERPOL chunk 86/95 | maq=64,091 | suelo_local=4,375,717 | 155s  ETA 16s
   ⚡ INTERPOL chunk 86/95 | maq=64,091 | suelo_local=4,375,717 | 155s  ETA 16s
   ⚡ INTERPOL chunk 87/95 | maq=156,218 | suelo_local=7,913,007 | 157s  ETA 14s
   ⚡ INTERPOL chunk 87/95 | maq=156,218 | suelo_local=7,913,007 | 157s  ETA 14s
   ⚡ INTERPOL chunk 88/95 | maq=40,569 | suelo_local=6,091,862 | 158s  ETA 13s
   ⚡ INTERPOL chunk 88/95 | maq=40,569 | suelo_local=6,091,862 | 158s  ETA 13s
   ⚡ INTERPOL chunk 89/95 | maq=215 | suelo_local=4,325,549 | 159s  ETA 11s
   ⚡ INTERPOL chunk 89/95 | maq=215 | suelo_local=4,325,549 | 159s  ETA 11s
   ⚡ INTERPOL chunk 91/95 | maq=11,432 | suelo_local=1,864,105 | 160s  ETA 7s
   ⚡ INTERPOL chunk 91/95 | maq=11,432 | suelo_local=1,864,105 | 160s  ETA 7s
   ⚡ INTERPOL chunk 92/95 | maq=1 | suelo_local=3,128,067 | 161s  ETA 5s
   ⚡ INTERPOL chunk 92/95 | maq=1 | suelo_local=3,128,067 | 161s  ETA 5s
   📊 Z diagnostico: 6,140,035/6,685,013 puntos con dZ>1cm | dZ medio=0.393m | dZ max=11.538m
   📊 Z diagnostico: 6,140,035/6,685,013 puntos con dZ>1cm | dZ medio=0.393m | dZ max=11.538m
   ✅ Aplanados 6,685,013 puntos
   ✅ INTERPOL: 6,685,013 puntos aplanados
💾 DTM guardado en 188.7s: LINK_260226_LOG176_NDP_PTL_edit_RGB_PointnetV6_DTM.laz
💾 DTM guardado: 188.7s