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
   RAM Disponible: 59.93 GB
   RAM Usada: 4.4%
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
   ⚡ Chunk 3/30 | core=1,412,694 pts | 1s elapsed  ETA 13s
   ⚡ Chunk 4/30 | core=3,255,564 pts | 3s elapsed  ETA 16s
   ⚡ Chunk 5/30 | core=3,988,372 pts | 4s elapsed  ETA 21s
   ⚡ Chunk 6/30 | core=1,901,925 pts | 4s elapsed  ETA 18s
   ⚡ Chunk 7/30 | core=3,161,304 pts | 5s elapsed  ETA 16s
   ⚡ Chunk 8/30 | core=3,979,412 pts | 5s elapsed  ETA 15s
   ⚡ Chunk 9/30 | core=1,757,545 pts | 6s elapsed  ETA 13s
   ⚡ Chunk 10/30 | core=2,682,968 pts | 6s elapsed  ETA 12s
   ⚡ Chunk 11/30 | core=2,955,260 pts | 6s elapsed  ETA 11s
   ⚡ Chunk 12/30 | core=1,426,306 pts | 7s elapsed  ETA 10s
   ⚡ Chunk 13/30 | core=3,265,826 pts | 7s elapsed  ETA 9s
   ⚡ Chunk 14/30 | core=3,995,523 pts | 7s elapsed  ETA 8s
   ⚡ Chunk 15/30 | core=1,611,844 pts | 8s elapsed  ETA 8s
   ⚡ Chunk 16/30 | core=3,216,534 pts | 8s elapsed  ETA 7s
   ⚡ Chunk 17/30 | core=3,991,073 pts | 9s elapsed  ETA 7s
   ⚡ Chunk 18/30 | core=1,543,374 pts | 9s elapsed  ETA 6s
   ⚡ Chunk 19/30 | core=3,202,135 pts | 9s elapsed  ETA 5s
   ⚡ Chunk 20/30 | core=3,987,316 pts | 10s elapsed  ETA 5s
   ⚡ Chunk 21/30 | core=1,624,251 pts | 10s elapsed  ETA 4s
   ⚡ Chunk 22/30 | core=3,180,516 pts | 10s elapsed  ETA 4s
   ⚡ Chunk 23/30 | core=3,989,658 pts | 11s elapsed  ETA 3s
   ⚡ Chunk 24/30 | core=1,594,611 pts | 11s elapsed  ETA 3s
   ⚡ Chunk 25/30 | core=3,008,817 pts | 11s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 12s elapsed  ETA 2s
   ⚡ Chunk 27/30 | core=816,914 pts | 12s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 12s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 12s elapsed  ETA 0s
   ✅ Normales completadas: 12.5s  (6,113,508 pts/s)
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
✅ Inferencia completada en 156.1s - Maquinaria: 287,318 puntos (0.4%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15642 MB
   🚜 Maquinaria: 287,318 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   ⚡ Smart Merge GPU: NVIDIA GeForce RTX 5090
   🔍 Smart Merge [GPU+CPU fallback]: 76,221,593 candidatos en 153 bloques
   🔍 Smart Merge: 76,221,593 candidatos
   ✨ Smart Merge: 38,725,540 puntos unidos
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar INTERPOL: 15642 MB
   📉 Maquinaria: 287,318 pts | Suelo: 76,249,677 pts | RAM arrays: 949 MB
   📉 Maquinaria: 287,318 | Suelo: 76,249,677 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1605.56m, mediana_suelo=1604.11m, gap=1.45m
   📐 Altura: mediana_maq=1605.56m, mediana_suelo=1604.11m, gap=1.45m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL chunk 1/95 | maq=710 | suelo_local=1,395,614 [↓22M→1395k] | 7s  ETA 683s
   ⚡ INTERPOL chunk 1/95 | maq=710 | suelo_local=1,395,614 [↓22M→1395k] | 7s  ETA 683s
   ⚡ INTERPOL chunk 2/95 | maq=320 | suelo_local=1,395,614 [↓22M→1395k] | 9s  ETA 401s
   ⚡ INTERPOL chunk 2/95 | maq=320 | suelo_local=1,395,614 [↓22M→1395k] | 9s  ETA 401s
   ⚡ INTERPOL chunk 3/95 | maq=1,587 | suelo_local=1,395,614 [↓22M→1395k] | 10s  ETA 307s
   ⚡ INTERPOL chunk 3/95 | maq=1,587 | suelo_local=1,395,614 [↓22M→1395k] | 10s  ETA 307s
   ⚡ INTERPOL chunk 4/95 | maq=500 | suelo_local=1,395,614 [↓22M→1395k] | 11s  ETA 258s
   ⚡ INTERPOL chunk 4/95 | maq=500 | suelo_local=1,395,614 [↓22M→1395k] | 11s  ETA 258s
   ⚡ INTERPOL chunk 6/95 | maq=856 | suelo_local=1,666,654 [↓26M→1666k] | 13s  ETA 193s
   ⚡ INTERPOL chunk 6/95 | maq=856 | suelo_local=1,666,654 [↓26M→1666k] | 13s  ETA 193s
   ⚡ INTERPOL chunk 7/95 | maq=628 | suelo_local=1,666,654 [↓26M→1666k] | 15s  ETA 183s
   ⚡ INTERPOL chunk 7/95 | maq=628 | suelo_local=1,666,654 [↓26M→1666k] | 15s  ETA 183s
   ⚡ INTERPOL chunk 8/95 | maq=422 | suelo_local=1,666,654 [↓26M→1666k] | 16s  ETA 175s
   ⚡ INTERPOL chunk 8/95 | maq=422 | suelo_local=1,666,654 [↓26M→1666k] | 16s  ETA 175s
   ⚡ INTERPOL chunk 9/95 | maq=3,279 | suelo_local=1,666,654 [↓26M→1666k] | 18s  ETA 169s
   ⚡ INTERPOL chunk 9/95 | maq=3,279 | suelo_local=1,666,654 [↓26M→1666k] | 18s  ETA 169s
   ⚡ INTERPOL chunk 10/95 | maq=601 | suelo_local=1,666,412 [↓26M→1666k] | 19s  ETA 164s
   ⚡ INTERPOL chunk 10/95 | maq=601 | suelo_local=1,666,412 [↓26M→1666k] | 19s  ETA 164s
   ⚡ INTERPOL chunk 11/95 | maq=666 | suelo_local=1,856,681 [↓29M→1856k] | 21s  ETA 161s
   ⚡ INTERPOL chunk 11/95 | maq=666 | suelo_local=1,856,681 [↓29M→1856k] | 21s  ETA 161s
   ⚡ INTERPOL chunk 12/95 | maq=4,065 | suelo_local=1,856,681 [↓29M→1856k] | 23s  ETA 158s
   ⚡ INTERPOL chunk 12/95 | maq=4,065 | suelo_local=1,856,681 [↓29M→1856k] | 23s  ETA 158s
   ⚡ INTERPOL chunk 13/95 | maq=881 | suelo_local=1,856,681 [↓29M→1856k] | 25s  ETA 156s
   ⚡ INTERPOL chunk 13/95 | maq=881 | suelo_local=1,856,681 [↓29M→1856k] | 25s  ETA 156s
   ⚡ INTERPOL chunk 14/95 | maq=1,786 | suelo_local=1,856,681 [↓29M→1856k] | 26s  ETA 153s
   ⚡ INTERPOL chunk 14/95 | maq=1,786 | suelo_local=1,856,681 [↓29M→1856k] | 26s  ETA 153s
   ⚡ INTERPOL chunk 15/95 | maq=3,578 | suelo_local=1,856,441 [↓29M→1856k] | 28s  ETA 150s
   ⚡ INTERPOL chunk 15/95 | maq=3,578 | suelo_local=1,856,441 [↓29M→1856k] | 28s  ETA 150s
   ⚡ INTERPOL chunk 16/95 | maq=121 | suelo_local=2,116,460 [↓33M→2116k] | 30s  ETA 149s
   ⚡ INTERPOL chunk 16/95 | maq=121 | suelo_local=2,116,460 [↓33M→2116k] | 30s  ETA 149s
   ⚡ INTERPOL chunk 17/95 | maq=790 | suelo_local=2,116,460 [↓33M→2116k] | 32s  ETA 148s
   ⚡ INTERPOL chunk 17/95 | maq=790 | suelo_local=2,116,460 [↓33M→2116k] | 32s  ETA 148s
   ⚡ INTERPOL chunk 18/95 | maq=1,934 | suelo_local=2,116,460 [↓33M→2116k] | 37s  ETA 157s
   ⚡ INTERPOL chunk 18/95 | maq=1,934 | suelo_local=2,116,460 [↓33M→2116k] | 37s  ETA 157s
   ⚡ INTERPOL chunk 19/95 | maq=6,267 | suelo_local=2,116,460 [↓33M→2116k] | 39s  ETA 156s
   ⚡ INTERPOL chunk 19/95 | maq=6,267 | suelo_local=2,116,460 [↓33M→2116k] | 39s  ETA 156s
   ⚡ INTERPOL chunk 20/95 | maq=8,005 | suelo_local=2,116,213 [↓33M→2116k] | 41s  ETA 154s
   ⚡ INTERPOL chunk 20/95 | maq=8,005 | suelo_local=2,116,213 [↓33M→2116k] | 41s  ETA 154s
   ⚡ INTERPOL chunk 21/95 | maq=931 | suelo_local=2,385,308 [↓38M→2385k] | 44s  ETA 154s
   ⚡ INTERPOL chunk 21/95 | maq=931 | suelo_local=2,385,308 [↓38M→2385k] | 44s  ETA 154s
   ⚡ INTERPOL chunk 22/95 | maq=1,025 | suelo_local=2,385,308 [↓38M→2385k] | 46s  ETA 154s
   ⚡ INTERPOL chunk 22/95 | maq=1,025 | suelo_local=2,385,308 [↓38M→2385k] | 46s  ETA 154s
   ⚡ INTERPOL chunk 23/95 | maq=4,377 | suelo_local=2,385,308 [↓38M→2385k] | 49s  ETA 152s
   ⚡ INTERPOL chunk 23/95 | maq=4,377 | suelo_local=2,385,308 [↓38M→2385k] | 49s  ETA 152s
   ⚡ INTERPOL chunk 24/95 | maq=11,047 | suelo_local=2,385,308 [↓38M→2385k] | 51s  ETA 150s
   ⚡ INTERPOL chunk 24/95 | maq=11,047 | suelo_local=2,385,308 [↓38M→2385k] | 51s  ETA 150s
   ⚡ INTERPOL chunk 25/95 | maq=2,917 | suelo_local=2,385,072 [↓38M→2385k] | 53s  ETA 148s
   ⚡ INTERPOL chunk 25/95 | maq=2,917 | suelo_local=2,385,072 [↓38M→2385k] | 53s  ETA 148s
   ⚡ INTERPOL chunk 26/95 | maq=5,295 | suelo_local=2,404,389 [↓38M→2404k] | 55s  ETA 146s
   ⚡ INTERPOL chunk 26/95 | maq=5,295 | suelo_local=2,404,389 [↓38M→2404k] | 55s  ETA 146s
   ⚡ INTERPOL chunk 27/95 | maq=7,537 | suelo_local=2,404,389 [↓38M→2404k] | 57s  ETA 144s
   ⚡ INTERPOL chunk 27/95 | maq=7,537 | suelo_local=2,404,389 [↓38M→2404k] | 57s  ETA 144s
   ⚡ INTERPOL chunk 28/95 | maq=4,708 | suelo_local=2,404,389 [↓38M→2404k] | 59s  ETA 142s
   ⚡ INTERPOL chunk 28/95 | maq=4,708 | suelo_local=2,404,389 [↓38M→2404k] | 59s  ETA 142s
   ⚡ INTERPOL chunk 29/95 | maq=4,653 | suelo_local=2,404,389 [↓38M→2404k] | 61s  ETA 139s
   ⚡ INTERPOL chunk 29/95 | maq=4,653 | suelo_local=2,404,389 [↓38M→2404k] | 61s  ETA 139s
   ⚡ INTERPOL chunk 30/95 | maq=2,396 | suelo_local=2,404,389 [↓38M→2404k] | 63s  ETA 137s
   ⚡ INTERPOL chunk 30/95 | maq=2,396 | suelo_local=2,404,389 [↓38M→2404k] | 63s  ETA 137s
   ⚡ INTERPOL chunk 31/95 | maq=1,831 | suelo_local=2,386,106 [↓38M→2386k] | 67s  ETA 139s
   ⚡ INTERPOL chunk 31/95 | maq=1,831 | suelo_local=2,386,106 [↓38M→2386k] | 67s  ETA 139s
   ⚡ INTERPOL chunk 32/95 | maq=7,341 | suelo_local=2,386,106 [↓38M→2386k] | 69s  ETA 137s
   ⚡ INTERPOL chunk 32/95 | maq=7,341 | suelo_local=2,386,106 [↓38M→2386k] | 69s  ETA 137s
   ⚡ INTERPOL chunk 33/95 | maq=10,086 | suelo_local=2,386,106 [↓38M→2386k] | 72s  ETA 135s
   ⚡ INTERPOL chunk 33/95 | maq=10,086 | suelo_local=2,386,106 [↓38M→2386k] | 72s  ETA 135s
   ⚡ INTERPOL chunk 34/95 | maq=11,979 | suelo_local=2,386,106 [↓38M→2386k] | 74s  ETA 132s
   ⚡ INTERPOL chunk 34/95 | maq=11,979 | suelo_local=2,386,106 [↓38M→2386k] | 74s  ETA 132s
   ⚡ INTERPOL chunk 35/95 | maq=8,304 | suelo_local=2,386,106 [↓38M→2386k] | 76s  ETA 130s
   ⚡ INTERPOL chunk 35/95 | maq=8,304 | suelo_local=2,386,106 [↓38M→2386k] | 76s  ETA 130s
   ⚡ INTERPOL chunk 36/95 | maq=177 | suelo_local=2,370,865 [↓37M→2370k] | 78s  ETA 128s
   ⚡ INTERPOL chunk 36/95 | maq=177 | suelo_local=2,370,865 [↓37M→2370k] | 78s  ETA 128s
   ⚡ INTERPOL chunk 37/95 | maq=7,252 | suelo_local=2,370,865 [↓37M→2370k] | 80s  ETA 126s
   ⚡ INTERPOL chunk 37/95 | maq=7,252 | suelo_local=2,370,865 [↓37M→2370k] | 80s  ETA 126s
   ⚡ INTERPOL chunk 38/95 | maq=4,175 | suelo_local=2,370,865 [↓37M→2370k] | 82s  ETA 123s
   ⚡ INTERPOL chunk 38/95 | maq=4,175 | suelo_local=2,370,865 [↓37M→2370k] | 82s  ETA 123s
   ⚡ INTERPOL chunk 39/95 | maq=1,993 | suelo_local=2,370,865 [↓37M→2370k] | 84s  ETA 121s
   ⚡ INTERPOL chunk 39/95 | maq=1,993 | suelo_local=2,370,865 [↓37M→2370k] | 84s  ETA 121s
   ⚡ INTERPOL chunk 40/95 | maq=1,567 | suelo_local=2,370,865 [↓37M→2370k] | 86s  ETA 119s
   ⚡ INTERPOL chunk 40/95 | maq=1,567 | suelo_local=2,370,865 [↓37M→2370k] | 86s  ETA 119s
   ⚡ INTERPOL chunk 41/95 | maq=30 | suelo_local=2,361,369 [↓37M→2361k] | 89s  ETA 117s
   ⚡ INTERPOL chunk 41/95 | maq=30 | suelo_local=2,361,369 [↓37M→2361k] | 89s  ETA 117s
   ⚡ INTERPOL chunk 42/95 | maq=936 | suelo_local=2,361,369 [↓37M→2361k] | 91s  ETA 114s
   ⚡ INTERPOL chunk 42/95 | maq=936 | suelo_local=2,361,369 [↓37M→2361k] | 91s  ETA 114s
   ⚡ INTERPOL chunk 43/95 | maq=120 | suelo_local=2,361,369 [↓37M→2361k] | 93s  ETA 112s
   ⚡ INTERPOL chunk 43/95 | maq=120 | suelo_local=2,361,369 [↓37M→2361k] | 93s  ETA 112s
   ⚡ INTERPOL chunk 44/95 | maq=1,410 | suelo_local=2,361,369 [↓37M→2361k] | 95s  ETA 110s
   ⚡ INTERPOL chunk 44/95 | maq=1,410 | suelo_local=2,361,369 [↓37M→2361k] | 95s  ETA 110s
   ⚡ INTERPOL chunk 45/95 | maq=2,877 | suelo_local=2,361,369 [↓37M→2361k] | 100s  ETA 111s
   ⚡ INTERPOL chunk 45/95 | maq=2,877 | suelo_local=2,361,369 [↓37M→2361k] | 100s  ETA 111s
   ⚡ INTERPOL chunk 46/95 | maq=124 | suelo_local=2,358,386 [↓37M→2358k] | 102s  ETA 109s
   ⚡ INTERPOL chunk 46/95 | maq=124 | suelo_local=2,358,386 [↓37M→2358k] | 102s  ETA 109s
   ⚡ INTERPOL chunk 47/95 | maq=697 | suelo_local=2,358,386 [↓37M→2358k] | 104s  ETA 107s
   ⚡ INTERPOL chunk 47/95 | maq=697 | suelo_local=2,358,386 [↓37M→2358k] | 104s  ETA 107s
   ⚡ INTERPOL chunk 49/95 | maq=657 | suelo_local=2,358,386 [↓37M→2358k] | 107s  ETA 100s
   ⚡ INTERPOL chunk 49/95 | maq=657 | suelo_local=2,358,386 [↓37M→2358k] | 107s  ETA 100s
   ⚡ INTERPOL chunk 50/95 | maq=12,178 | suelo_local=2,358,386 [↓37M→2358k] | 109s  ETA 98s
   ⚡ INTERPOL chunk 50/95 | maq=12,178 | suelo_local=2,358,386 [↓37M→2358k] | 109s  ETA 98s
   ⚡ INTERPOL chunk 51/95 | maq=275 | suelo_local=2,361,648 [↓37M→2361k] | 111s  ETA 96s
   ⚡ INTERPOL chunk 51/95 | maq=275 | suelo_local=2,361,648 [↓37M→2361k] | 111s  ETA 96s
   ⚡ INTERPOL chunk 52/95 | maq=747 | suelo_local=2,361,648 [↓37M→2361k] | 113s  ETA 94s
   ⚡ INTERPOL chunk 52/95 | maq=747 | suelo_local=2,361,648 [↓37M→2361k] | 113s  ETA 94s
   ⚡ INTERPOL chunk 54/95 | maq=1,479 | suelo_local=2,361,648 [↓37M→2361k] | 115s  ETA 88s
   ⚡ INTERPOL chunk 54/95 | maq=1,479 | suelo_local=2,361,648 [↓37M→2361k] | 115s  ETA 88s
   ⚡ INTERPOL chunk 55/95 | maq=198 | suelo_local=2,361,648 [↓37M→2361k] | 117s  ETA 85s
   ⚡ INTERPOL chunk 55/95 | maq=198 | suelo_local=2,361,648 [↓37M→2361k] | 117s  ETA 85s
   ⚡ INTERPOL chunk 56/95 | maq=1,317 | suelo_local=2,439,299 [↓39M→2439k] | 120s  ETA 83s
   ⚡ INTERPOL chunk 56/95 | maq=1,317 | suelo_local=2,439,299 [↓39M→2439k] | 120s  ETA 83s
   ⚡ INTERPOL chunk 57/95 | maq=10,456 | suelo_local=2,439,299 [↓39M→2439k] | 122s  ETA 81s
   ⚡ INTERPOL chunk 57/95 | maq=10,456 | suelo_local=2,439,299 [↓39M→2439k] | 122s  ETA 81s
   ⚡ INTERPOL chunk 58/95 | maq=14 | suelo_local=2,439,299 [↓39M→2439k] | 124s  ETA 79s
   ⚡ INTERPOL chunk 58/95 | maq=14 | suelo_local=2,439,299 [↓39M→2439k] | 124s  ETA 79s
   ⚡ INTERPOL chunk 59/95 | maq=3,035 | suelo_local=2,439,299 [↓39M→2439k] | 126s  ETA 77s
   ⚡ INTERPOL chunk 59/95 | maq=3,035 | suelo_local=2,439,299 [↓39M→2439k] | 126s  ETA 77s
   ⚡ INTERPOL chunk 61/95 | maq=6,763 | suelo_local=2,439,287 [↓39M→2439k] | 128s  ETA 71s
   ⚡ INTERPOL chunk 61/95 | maq=6,763 | suelo_local=2,439,287 [↓39M→2439k] | 128s  ETA 71s
   ⚡ INTERPOL chunk 62/95 | maq=10,908 | suelo_local=2,439,287 [↓39M→2439k] | 132s  ETA 70s
   ⚡ INTERPOL chunk 62/95 | maq=10,908 | suelo_local=2,439,287 [↓39M→2439k] | 132s  ETA 70s
   ⚡ INTERPOL chunk 63/95 | maq=6,491 | suelo_local=2,439,287 [↓39M→2439k] | 135s  ETA 68s
   ⚡ INTERPOL chunk 63/95 | maq=6,491 | suelo_local=2,439,287 [↓39M→2439k] | 135s  ETA 68s
   ⚡ INTERPOL chunk 64/95 | maq=6,188 | suelo_local=2,439,287 [↓39M→2439k] | 137s  ETA 66s
   ⚡ INTERPOL chunk 64/95 | maq=6,188 | suelo_local=2,439,287 [↓39M→2439k] | 137s  ETA 66s
   ⚡ INTERPOL chunk 65/95 | maq=297 | suelo_local=2,439,287 [↓39M→2439k] | 139s  ETA 64s
   ⚡ INTERPOL chunk 65/95 | maq=297 | suelo_local=2,439,287 [↓39M→2439k] | 139s  ETA 64s
   ⚡ INTERPOL chunk 66/95 | maq=4,035 | suelo_local=2,339,489 [↓37M→2339k] | 141s  ETA 62s
   ⚡ INTERPOL chunk 66/95 | maq=4,035 | suelo_local=2,339,489 [↓37M→2339k] | 141s  ETA 62s
   ⚡ INTERPOL chunk 67/95 | maq=3,369 | suelo_local=2,339,489 [↓37M→2339k] | 143s  ETA 60s
   ⚡ INTERPOL chunk 67/95 | maq=3,369 | suelo_local=2,339,489 [↓37M→2339k] | 143s  ETA 60s
   ⚡ INTERPOL chunk 68/95 | maq=8,703 | suelo_local=2,339,489 [↓37M→2339k] | 145s  ETA 58s
   ⚡ INTERPOL chunk 68/95 | maq=8,703 | suelo_local=2,339,489 [↓37M→2339k] | 145s  ETA 58s
   ⚡ INTERPOL chunk 69/95 | maq=4,581 | suelo_local=2,339,489 [↓37M→2339k] | 147s  ETA 55s
   ⚡ INTERPOL chunk 69/95 | maq=4,581 | suelo_local=2,339,489 [↓37M→2339k] | 147s  ETA 55s
   ⚡ INTERPOL chunk 70/95 | maq=902 | suelo_local=2,339,489 [↓37M→2339k] | 149s  ETA 53s
   ⚡ INTERPOL chunk 70/95 | maq=902 | suelo_local=2,339,489 [↓37M→2339k] | 149s  ETA 53s
   ⚡ INTERPOL chunk 71/95 | maq=916 | suelo_local=2,090,561 [↓33M→2090k] | 151s  ETA 51s
   ⚡ INTERPOL chunk 71/95 | maq=916 | suelo_local=2,090,561 [↓33M→2090k] | 151s  ETA 51s
   ⚡ INTERPOL chunk 72/95 | maq=1,131 | suelo_local=2,090,561 [↓33M→2090k] | 153s  ETA 49s
   ⚡ INTERPOL chunk 72/95 | maq=1,131 | suelo_local=2,090,561 [↓33M→2090k] | 153s  ETA 49s
   ⚡ INTERPOL chunk 73/95 | maq=3,009 | suelo_local=2,090,561 [↓33M→2090k] | 155s  ETA 47s
   ⚡ INTERPOL chunk 73/95 | maq=3,009 | suelo_local=2,090,561 [↓33M→2090k] | 155s  ETA 47s
   ⚡ INTERPOL chunk 74/95 | maq=791 | suelo_local=2,090,561 [↓33M→2090k] | 158s  ETA 45s
   ⚡ INTERPOL chunk 74/95 | maq=791 | suelo_local=2,090,561 [↓33M→2090k] | 158s  ETA 45s
   ⚡ INTERPOL chunk 75/95 | maq=4,542 | suelo_local=2,090,561 [↓33M→2090k] | 160s  ETA 43s
   ⚡ INTERPOL chunk 75/95 | maq=4,542 | suelo_local=2,090,561 [↓33M→2090k] | 160s  ETA 43s
   ⚡ INTERPOL chunk 76/95 | maq=275 | suelo_local=1,819,182 [↓29M→1819k] | 164s  ETA 41s
   ⚡ INTERPOL chunk 76/95 | maq=275 | suelo_local=1,819,182 [↓29M→1819k] | 164s  ETA 41s
   ⚡ INTERPOL chunk 77/95 | maq=1,231 | suelo_local=1,819,182 [↓29M→1819k] | 166s  ETA 39s
   ⚡ INTERPOL chunk 77/95 | maq=1,231 | suelo_local=1,819,182 [↓29M→1819k] | 166s  ETA 39s
   ⚡ INTERPOL chunk 78/95 | maq=1,390 | suelo_local=1,819,182 [↓29M→1819k] | 167s  ETA 36s
   ⚡ INTERPOL chunk 78/95 | maq=1,390 | suelo_local=1,819,182 [↓29M→1819k] | 167s  ETA 36s
   ⚡ INTERPOL chunk 79/95 | maq=4,553 | suelo_local=1,819,182 [↓29M→1819k] | 169s  ETA 34s
   ⚡ INTERPOL chunk 79/95 | maq=4,553 | suelo_local=1,819,182 [↓29M→1819k] | 169s  ETA 34s
   ⚡ INTERPOL chunk 80/95 | maq=1,097 | suelo_local=1,819,182 [↓29M→1819k] | 171s  ETA 32s
   ⚡ INTERPOL chunk 80/95 | maq=1,097 | suelo_local=1,819,182 [↓29M→1819k] | 171s  ETA 32s
   ⚡ INTERPOL chunk 81/95 | maq=631 | suelo_local=1,548,560 [↓24M→1548k] | 172s  ETA 30s
   ⚡ INTERPOL chunk 81/95 | maq=631 | suelo_local=1,548,560 [↓24M→1548k] | 172s  ETA 30s
   ⚡ INTERPOL chunk 82/95 | maq=4,105 | suelo_local=1,548,560 [↓24M→1548k] | 174s  ETA 28s
   ⚡ INTERPOL chunk 82/95 | maq=4,105 | suelo_local=1,548,560 [↓24M→1548k] | 174s  ETA 28s
   ⚡ INTERPOL chunk 83/95 | maq=9,218 | suelo_local=1,548,560 [↓24M→1548k] | 175s  ETA 25s
   ⚡ INTERPOL chunk 83/95 | maq=9,218 | suelo_local=1,548,560 [↓24M→1548k] | 175s  ETA 25s
   ⚡ INTERPOL chunk 84/95 | maq=7,469 | suelo_local=1,548,560 [↓24M→1548k] | 176s  ETA 23s
   ⚡ INTERPOL chunk 84/95 | maq=7,469 | suelo_local=1,548,560 [↓24M→1548k] | 176s  ETA 23s
   ⚡ INTERPOL chunk 85/95 | maq=3,743 | suelo_local=1,548,560 [↓24M→1548k] | 178s  ETA 21s
   ⚡ INTERPOL chunk 85/95 | maq=3,743 | suelo_local=1,548,560 [↓24M→1548k] | 178s  ETA 21s
   ⚡ INTERPOL chunk 86/95 | maq=5,087 | suelo_local=1,276,854 [↓20M→1276k] | 179s  ETA 19s
   ⚡ INTERPOL chunk 86/95 | maq=5,087 | suelo_local=1,276,854 [↓20M→1276k] | 179s  ETA 19s
   ⚡ INTERPOL chunk 87/95 | maq=10,955 | suelo_local=1,276,854 [↓20M→1276k] | 180s  ETA 17s
   ⚡ INTERPOL chunk 87/95 | maq=10,955 | suelo_local=1,276,854 [↓20M→1276k] | 180s  ETA 17s
   ⚡ INTERPOL chunk 88/95 | maq=1,195 | suelo_local=1,276,854 [↓20M→1276k] | 181s  ETA 14s
   ⚡ INTERPOL chunk 88/95 | maq=1,195 | suelo_local=1,276,854 [↓20M→1276k] | 181s  ETA 14s
   ⚡ INTERPOL chunk 89/95 | maq=21 | suelo_local=1,276,854 [↓20M→1276k] | 183s  ETA 12s
   ⚡ INTERPOL chunk 89/95 | maq=21 | suelo_local=1,276,854 [↓20M→1276k] | 183s  ETA 12s
   ⚡ INTERPOL chunk 91/95 | maq=585 | suelo_local=1,002,052 [↓16M→1002k] | 183s  ETA 8s
   ⚡ INTERPOL chunk 91/95 | maq=585 | suelo_local=1,002,052 [↓16M→1002k] | 183s  ETA 8s
   📊 Z diagnostico: 240,884/287,318 puntos con dZ>1cm | dZ medio=0.165m | dZ max=6.727m
   📊 Z diagnostico: 240,884/287,318 puntos con dZ>1cm | dZ medio=0.165m | dZ max=6.727m
   ✅ Aplanados 287,318 puntos
   ✅ INTERPOL: 287,318 puntos aplanados
💾 DTM guardado en 192.4s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 192.4s