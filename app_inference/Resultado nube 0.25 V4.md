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
🎯 INICIANDO INFERENCIA: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m.laz
======================================================================
   🔍 [Inicio de inferencia] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   📦 Tamaño del archivo: 244.36 MB
   🔍 [Antes de extraer features] GPU Memory: Usada=0.00GB, Reservada=0.01GB, Libre=31.83GB, Total=31.84GB
   🧮 Calculando normales en chunks espaciales (r=3.5m, ~50m x 50m por chunk)...
   🔥 Normales: usando GPU (Open3D Tensor CUDA)
   📐 Nube: 76,536,995 puntos → ~30 chunks (10×3) de 500m
   ⚡ Chunk 1/30 | core=3,071,381 pts | 1s elapsed  ETA 20s
   ⚡ Chunk 2/30 | core=3,779,447 pts | 1s elapsed  ETA 16s
   ⚡ Chunk 3/30 | core=1,412,694 pts | 1s elapsed  ETA 13s
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
   ⚡ Chunk 20/30 | core=3,987,316 pts | 8s elapsed  ETA 4s
   ⚡ Chunk 21/30 | core=1,624,251 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 22/30 | core=3,180,516 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 23/30 | core=3,989,658 pts | 9s elapsed  ETA 3s
   ⚡ Chunk 24/30 | core=1,594,611 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 25/30 | core=3,008,817 pts | 11s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 12s elapsed  ETA 2s
   ⚡ Chunk 27/30 | core=816,914 pts | 12s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 12s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 12s elapsed  ETA 0s
   ✅ Normales completadas: 12.5s  (6,136,230 pts/s)
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
✅ Inferencia completada en 154.7s - Maquinaria: 285,046 puntos (0.4%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15645 MB
   🚜 Maquinaria: 285,046 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   ⚡ Smart Merge GPU: NVIDIA GeForce RTX 5090
   🔍 Smart Merge [GPU+CPU fallback]: 76,241,654 candidatos en 153 bloques
   🔍 Smart Merge: 76,241,654 candidatos
   ✨ Smart Merge: 38,731,820 puntos unidos
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar INTERPOL: 15645 MB
   📉 Maquinaria: 285,046 pts | Suelo: 76,251,949 pts | RAM arrays: 949 MB
   📉 Maquinaria: 285,046 | Suelo: 76,251,949 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1605.55m, mediana_suelo=1604.11m, gap=1.44m
   📐 Altura: mediana_maq=1605.55m, mediana_suelo=1604.11m, gap=1.44m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL GPU: NVIDIA GeForce RTX 5090 | VRAM libre=31.8GB, budget=9.6GB
   ⚡ INTERPOL chunk 1/95 [GPU] | maq=537 | suelo_local=1,385,565 [↓22M→1385k] | 10s  ETA 965s
   ⚡ INTERPOL chunk 1/95 [GPU] | maq=537 | suelo_local=1,385,565 [↓22M→1385k] | 10s  ETA 965s
   ⚡ INTERPOL chunk 2/95 [GPU] | maq=115 | suelo_local=1,385,565 [↓22M→1385k] | 12s  ETA 535s
   ⚡ INTERPOL chunk 2/95 [GPU] | maq=115 | suelo_local=1,385,565 [↓22M→1385k] | 12s  ETA 535s
   ⚡ INTERPOL chunk 3/95 [GPU] | maq=1,220 | suelo_local=1,385,565 [↓22M→1385k] | 13s  ETA 396s
   ⚡ INTERPOL chunk 3/95 [GPU] | maq=1,220 | suelo_local=1,385,565 [↓22M→1385k] | 13s  ETA 396s
   ⚡ INTERPOL chunk 4/95 [GPU] | maq=258 | suelo_local=1,385,565 [↓22M→1385k] | 14s  ETA 320s
   ⚡ INTERPOL chunk 4/95 [GPU] | maq=258 | suelo_local=1,385,565 [↓22M→1385k] | 14s  ETA 320s
   ⚡ INTERPOL chunk 6/95 [GPU] | maq=718 | suelo_local=1,658,134 [↓26M→1658k] | 16s  ETA 234s
   ⚡ INTERPOL chunk 6/95 [GPU] | maq=718 | suelo_local=1,658,134 [↓26M→1658k] | 16s  ETA 234s
   ⚡ INTERPOL chunk 7/95 [GPU] | maq=758 | suelo_local=1,658,134 [↓26M→1658k] | 17s  ETA 219s
   ⚡ INTERPOL chunk 7/95 [GPU] | maq=758 | suelo_local=1,658,134 [↓26M→1658k] | 17s  ETA 219s
   ⚡ INTERPOL chunk 8/95 [GPU] | maq=343 | suelo_local=1,658,134 [↓26M→1658k] | 19s  ETA 204s
   ⚡ INTERPOL chunk 8/95 [GPU] | maq=343 | suelo_local=1,658,134 [↓26M→1658k] | 19s  ETA 204s
   ⚡ INTERPOL chunk 9/95 [GPU] | maq=3,114 | suelo_local=1,658,134 [↓26M→1658k] | 21s  ETA 198s
   ⚡ INTERPOL chunk 9/95 [GPU] | maq=3,114 | suelo_local=1,658,134 [↓26M→1658k] | 21s  ETA 198s
   ⚡ INTERPOL chunk 10/95 [GPU] | maq=458 | suelo_local=1,657,944 [↓26M→1657k] | 22s  ETA 188s
   ⚡ INTERPOL chunk 10/95 [GPU] | maq=458 | suelo_local=1,657,944 [↓26M→1657k] | 22s  ETA 188s
   ⚡ INTERPOL chunk 11/95 [GPU] | maq=965 | suelo_local=1,848,861 [↓29M→1848k] | 24s  ETA 184s
   ⚡ INTERPOL chunk 11/95 [GPU] | maq=965 | suelo_local=1,848,861 [↓29M→1848k] | 24s  ETA 184s
   ⚡ INTERPOL chunk 12/95 [GPU] | maq=3,945 | suelo_local=1,848,861 [↓29M→1848k] | 26s  ETA 182s
   ⚡ INTERPOL chunk 12/95 [GPU] | maq=3,945 | suelo_local=1,848,861 [↓29M→1848k] | 26s  ETA 182s
   ⚡ INTERPOL chunk 13/95 [GPU] | maq=743 | suelo_local=1,848,861 [↓29M→1848k] | 28s  ETA 177s
   ⚡ INTERPOL chunk 13/95 [GPU] | maq=743 | suelo_local=1,848,861 [↓29M→1848k] | 28s  ETA 177s
   ⚡ INTERPOL chunk 14/95 [GPU] | maq=1,938 | suelo_local=1,848,861 [↓29M→1848k] | 30s  ETA 175s
   ⚡ INTERPOL chunk 14/95 [GPU] | maq=1,938 | suelo_local=1,848,861 [↓29M→1848k] | 30s  ETA 175s
   ⚡ INTERPOL chunk 15/95 [GPU] | maq=3,152 | suelo_local=1,848,600 [↓29M→1848k] | 32s  ETA 172s
   ⚡ INTERPOL chunk 15/95 [GPU] | maq=3,152 | suelo_local=1,848,600 [↓29M→1848k] | 32s  ETA 172s
   ⚡ INTERPOL chunk 16/95 [GPU] | maq=76 | suelo_local=2,105,724 [↓33M→2105k] | 34s  ETA 169s
   ⚡ INTERPOL chunk 16/95 [GPU] | maq=76 | suelo_local=2,105,724 [↓33M→2105k] | 34s  ETA 169s
   ⚡ INTERPOL chunk 17/95 [GPU] | maq=842 | suelo_local=2,105,724 [↓33M→2105k] | 36s  ETA 166s
   ⚡ INTERPOL chunk 17/95 [GPU] | maq=842 | suelo_local=2,105,724 [↓33M→2105k] | 36s  ETA 166s
   ⚡ INTERPOL chunk 18/95 [GPU] | maq=2,393 | suelo_local=2,105,724 [↓33M→2105k] | 41s  ETA 174s
   ⚡ INTERPOL chunk 18/95 [GPU] | maq=2,393 | suelo_local=2,105,724 [↓33M→2105k] | 41s  ETA 174s
   ⚡ INTERPOL chunk 19/95 [GPU] | maq=4,876 | suelo_local=2,105,724 [↓33M→2105k] | 43s  ETA 173s
   ⚡ INTERPOL chunk 19/95 [GPU] | maq=4,876 | suelo_local=2,105,724 [↓33M→2105k] | 43s  ETA 173s
   ⚡ INTERPOL chunk 20/95 [GPU] | maq=7,984 | suelo_local=2,105,433 [↓33M→2105k] | 46s  ETA 172s
   ⚡ INTERPOL chunk 20/95 [GPU] | maq=7,984 | suelo_local=2,105,433 [↓33M→2105k] | 46s  ETA 172s
   ⚡ INTERPOL chunk 21/95 [GPU] | maq=785 | suelo_local=2,377,901 [↓38M→2377k] | 48s  ETA 169s
   ⚡ INTERPOL chunk 21/95 [GPU] | maq=785 | suelo_local=2,377,901 [↓38M→2377k] | 48s  ETA 169s
   ⚡ INTERPOL chunk 22/95 [GPU] | maq=865 | suelo_local=2,377,901 [↓38M→2377k] | 50s  ETA 167s
   ⚡ INTERPOL chunk 22/95 [GPU] | maq=865 | suelo_local=2,377,901 [↓38M→2377k] | 50s  ETA 167s
   ⚡ INTERPOL chunk 23/95 [GPU] | maq=4,831 | suelo_local=2,377,901 [↓38M→2377k] | 53s  ETA 166s
   ⚡ INTERPOL chunk 23/95 [GPU] | maq=4,831 | suelo_local=2,377,901 [↓38M→2377k] | 53s  ETA 166s
   ⚡ INTERPOL chunk 24/95 [GPU] | maq=11,075 | suelo_local=2,377,901 [↓38M→2377k] | 57s  ETA 168s
   ⚡ INTERPOL chunk 24/95 [GPU] | maq=11,075 | suelo_local=2,377,901 [↓38M→2377k] | 57s  ETA 168s
   ⚡ INTERPOL chunk 25/95 [GPU] | maq=3,045 | suelo_local=2,377,624 [↓38M→2377k] | 59s  ETA 166s
   ⚡ INTERPOL chunk 25/95 [GPU] | maq=3,045 | suelo_local=2,377,624 [↓38M→2377k] | 59s  ETA 166s
   ⚡ INTERPOL chunk 26/95 [GPU] | maq=5,150 | suelo_local=2,404,537 [↓38M→2404k] | 62s  ETA 164s
   ⚡ INTERPOL chunk 26/95 [GPU] | maq=5,150 | suelo_local=2,404,537 [↓38M→2404k] | 62s  ETA 164s
   ⚡ INTERPOL chunk 27/95 [GPU] | maq=6,391 | suelo_local=2,404,537 [↓38M→2404k] | 65s  ETA 164s
   ⚡ INTERPOL chunk 27/95 [GPU] | maq=6,391 | suelo_local=2,404,537 [↓38M→2404k] | 65s  ETA 164s
   ⚡ INTERPOL chunk 28/95 [GPU] | maq=4,322 | suelo_local=2,404,537 [↓38M→2404k] | 68s  ETA 162s
   ⚡ INTERPOL chunk 28/95 [GPU] | maq=4,322 | suelo_local=2,404,537 [↓38M→2404k] | 68s  ETA 162s
   ⚡ INTERPOL chunk 29/95 [GPU] | maq=4,465 | suelo_local=2,404,537 [↓38M→2404k] | 72s  ETA 164s
   ⚡ INTERPOL chunk 29/95 [GPU] | maq=4,465 | suelo_local=2,404,537 [↓38M→2404k] | 72s  ETA 164s
   ⚡ INTERPOL chunk 30/95 [GPU] | maq=2,714 | suelo_local=2,404,537 [↓38M→2404k] | 75s  ETA 162s
   ⚡ INTERPOL chunk 30/95 [GPU] | maq=2,714 | suelo_local=2,404,537 [↓38M→2404k] | 75s  ETA 162s
   ⚡ INTERPOL chunk 31/95 [GPU] | maq=2,357 | suelo_local=2,386,832 [↓38M→2386k] | 77s  ETA 160s
   ⚡ INTERPOL chunk 31/95 [GPU] | maq=2,357 | suelo_local=2,386,832 [↓38M→2386k] | 77s  ETA 160s
   ⚡ INTERPOL chunk 32/95 [GPU] | maq=7,187 | suelo_local=2,386,832 [↓38M→2386k] | 80s  ETA 158s
   ⚡ INTERPOL chunk 32/95 [GPU] | maq=7,187 | suelo_local=2,386,832 [↓38M→2386k] | 80s  ETA 158s
   ⚡ INTERPOL chunk 33/95 [GPU] | maq=10,276 | suelo_local=2,386,832 [↓38M→2386k] | 83s  ETA 157s
   ⚡ INTERPOL chunk 33/95 [GPU] | maq=10,276 | suelo_local=2,386,832 [↓38M→2386k] | 83s  ETA 157s
   ⚡ INTERPOL chunk 34/95 [GPU] | maq=11,856 | suelo_local=2,386,832 [↓38M→2386k] | 87s  ETA 156s
   ⚡ INTERPOL chunk 34/95 [GPU] | maq=11,856 | suelo_local=2,386,832 [↓38M→2386k] | 87s  ETA 156s
   ⚡ INTERPOL chunk 35/95 [GPU] | maq=8,057 | suelo_local=2,386,832 [↓38M→2386k] | 90s  ETA 154s
   ⚡ INTERPOL chunk 35/95 [GPU] | maq=8,057 | suelo_local=2,386,832 [↓38M→2386k] | 90s  ETA 154s
   ⚡ INTERPOL chunk 36/95 [GPU] | maq=431 | suelo_local=2,371,339 [↓37M→2371k] | 92s  ETA 151s
   ⚡ INTERPOL chunk 36/95 [GPU] | maq=431 | suelo_local=2,371,339 [↓37M→2371k] | 92s  ETA 151s
   ⚡ INTERPOL chunk 37/95 [GPU] | maq=7,517 | suelo_local=2,371,339 [↓37M→2371k] | 95s  ETA 149s
   ⚡ INTERPOL chunk 37/95 [GPU] | maq=7,517 | suelo_local=2,371,339 [↓37M→2371k] | 95s  ETA 149s
   ⚡ INTERPOL chunk 38/95 [GPU] | maq=5,118 | suelo_local=2,371,339 [↓37M→2371k] | 98s  ETA 147s
   ⚡ INTERPOL chunk 38/95 [GPU] | maq=5,118 | suelo_local=2,371,339 [↓37M→2371k] | 98s  ETA 147s
   ⚡ INTERPOL chunk 39/95 [GPU] | maq=2,420 | suelo_local=2,371,339 [↓37M→2371k] | 100s  ETA 144s
   ⚡ INTERPOL chunk 39/95 [GPU] | maq=2,420 | suelo_local=2,371,339 [↓37M→2371k] | 100s  ETA 144s
   ⚡ INTERPOL chunk 40/95 [GPU] | maq=1,555 | suelo_local=2,371,339 [↓37M→2371k] | 105s  ETA 144s
   ⚡ INTERPOL chunk 40/95 [GPU] | maq=1,555 | suelo_local=2,371,339 [↓37M→2371k] | 105s  ETA 144s
   ⚡ INTERPOL chunk 41/95 [GPU] | maq=116 | suelo_local=2,361,985 [↓37M→2361k] | 107s  ETA 141s
   ⚡ INTERPOL chunk 41/95 [GPU] | maq=116 | suelo_local=2,361,985 [↓37M→2361k] | 107s  ETA 141s
   ⚡ INTERPOL chunk 42/95 [GPU] | maq=1,043 | suelo_local=2,361,985 [↓37M→2361k] | 109s  ETA 137s
   ⚡ INTERPOL chunk 42/95 [GPU] | maq=1,043 | suelo_local=2,361,985 [↓37M→2361k] | 109s  ETA 137s
   ⚡ INTERPOL chunk 43/95 [GPU] | maq=93 | suelo_local=2,361,985 [↓37M→2361k] | 111s  ETA 134s
   ⚡ INTERPOL chunk 43/95 [GPU] | maq=93 | suelo_local=2,361,985 [↓37M→2361k] | 111s  ETA 134s
   ⚡ INTERPOL chunk 44/95 [GPU] | maq=1,634 | suelo_local=2,361,985 [↓37M→2361k] | 113s  ETA 131s
   ⚡ INTERPOL chunk 44/95 [GPU] | maq=1,634 | suelo_local=2,361,985 [↓37M→2361k] | 113s  ETA 131s
   ⚡ INTERPOL chunk 45/95 [GPU] | maq=2,737 | suelo_local=2,361,985 [↓37M→2361k] | 116s  ETA 129s
   ⚡ INTERPOL chunk 45/95 [GPU] | maq=2,737 | suelo_local=2,361,985 [↓37M→2361k] | 116s  ETA 129s
   ⚡ INTERPOL chunk 46/95 [GPU] | maq=102 | suelo_local=2,358,822 [↓37M→2358k] | 118s  ETA 125s
   ⚡ INTERPOL chunk 46/95 [GPU] | maq=102 | suelo_local=2,358,822 [↓37M→2358k] | 118s  ETA 125s
   ⚡ INTERPOL chunk 47/95 [GPU] | maq=753 | suelo_local=2,358,822 [↓37M→2358k] | 120s  ETA 122s
   ⚡ INTERPOL chunk 47/95 [GPU] | maq=753 | suelo_local=2,358,822 [↓37M→2358k] | 120s  ETA 122s
   ⚡ INTERPOL chunk 49/95 [GPU] | maq=872 | suelo_local=2,358,822 [↓37M→2358k] | 123s  ETA 115s
   ⚡ INTERPOL chunk 49/95 [GPU] | maq=872 | suelo_local=2,358,822 [↓37M→2358k] | 123s  ETA 115s
   ⚡ INTERPOL chunk 50/95 [GPU] | maq=12,133 | suelo_local=2,358,822 [↓37M→2358k] | 126s  ETA 113s
   ⚡ INTERPOL chunk 50/95 [GPU] | maq=12,133 | suelo_local=2,358,822 [↓37M→2358k] | 126s  ETA 113s
   ⚡ INTERPOL chunk 51/95 [GPU] | maq=188 | suelo_local=2,360,367 [↓37M→2360k] | 128s  ETA 111s
   ⚡ INTERPOL chunk 51/95 [GPU] | maq=188 | suelo_local=2,360,367 [↓37M→2360k] | 128s  ETA 111s
   ⚡ INTERPOL chunk 52/95 [GPU] | maq=725 | suelo_local=2,360,367 [↓37M→2360k] | 130s  ETA 108s
   ⚡ INTERPOL chunk 52/95 [GPU] | maq=725 | suelo_local=2,360,367 [↓37M→2360k] | 130s  ETA 108s
   ⚡ INTERPOL chunk 54/95 [GPU] | maq=1,563 | suelo_local=2,360,367 [↓37M→2360k] | 133s  ETA 101s
   ⚡ INTERPOL chunk 54/95 [GPU] | maq=1,563 | suelo_local=2,360,367 [↓37M→2360k] | 133s  ETA 101s
   ⚡ INTERPOL chunk 55/95 [GPU] | maq=141 | suelo_local=2,360,367 [↓37M→2360k] | 137s  ETA 100s
   ⚡ INTERPOL chunk 55/95 [GPU] | maq=141 | suelo_local=2,360,367 [↓37M→2360k] | 137s  ETA 100s
   ⚡ INTERPOL chunk 56/95 [GPU] | maq=1,343 | suelo_local=2,437,632 [↓39M→2437k] | 140s  ETA 97s
   ⚡ INTERPOL chunk 56/95 [GPU] | maq=1,343 | suelo_local=2,437,632 [↓39M→2437k] | 140s  ETA 97s
   ⚡ INTERPOL chunk 57/95 [GPU] | maq=9,778 | suelo_local=2,437,632 [↓39M→2437k] | 143s  ETA 95s
   ⚡ INTERPOL chunk 57/95 [GPU] | maq=9,778 | suelo_local=2,437,632 [↓39M→2437k] | 143s  ETA 95s
   ⚡ INTERPOL chunk 58/95 [GPU] | maq=19 | suelo_local=2,437,632 [↓39M→2437k] | 145s  ETA 92s
   ⚡ INTERPOL chunk 58/95 [GPU] | maq=19 | suelo_local=2,437,632 [↓39M→2437k] | 145s  ETA 92s
   ⚡ INTERPOL chunk 59/95 [GPU] | maq=3,052 | suelo_local=2,437,632 [↓39M→2437k] | 148s  ETA 90s
   ⚡ INTERPOL chunk 59/95 [GPU] | maq=3,052 | suelo_local=2,437,632 [↓39M→2437k] | 148s  ETA 90s
   ⚡ INTERPOL chunk 61/95 [GPU] | maq=6,445 | suelo_local=2,440,894 [↓39M→2440k] | 150s  ETA 84s
   ⚡ INTERPOL chunk 61/95 [GPU] | maq=6,445 | suelo_local=2,440,894 [↓39M→2440k] | 150s  ETA 84s
   ⚡ INTERPOL chunk 62/95 [GPU] | maq=11,784 | suelo_local=2,440,894 [↓39M→2440k] | 154s  ETA 82s
   ⚡ INTERPOL chunk 62/95 [GPU] | maq=11,784 | suelo_local=2,440,894 [↓39M→2440k] | 154s  ETA 82s
   ⚡ INTERPOL chunk 63/95 [GPU] | maq=6,098 | suelo_local=2,440,894 [↓39M→2440k] | 157s  ETA 80s
   ⚡ INTERPOL chunk 63/95 [GPU] | maq=6,098 | suelo_local=2,440,894 [↓39M→2440k] | 157s  ETA 80s
   ⚡ INTERPOL chunk 64/95 [GPU] | maq=6,213 | suelo_local=2,440,894 [↓39M→2440k] | 160s  ETA 77s
   ⚡ INTERPOL chunk 64/95 [GPU] | maq=6,213 | suelo_local=2,440,894 [↓39M→2440k] | 160s  ETA 77s
   ⚡ INTERPOL chunk 65/95 [GPU] | maq=130 | suelo_local=2,440,894 [↓39M→2440k] | 162s  ETA 75s
   ⚡ INTERPOL chunk 65/95 [GPU] | maq=130 | suelo_local=2,440,894 [↓39M→2440k] | 162s  ETA 75s
   ⚡ INTERPOL chunk 66/95 [GPU] | maq=4,416 | suelo_local=2,346,350 [↓37M→2346k] | 164s  ETA 72s
   ⚡ INTERPOL chunk 66/95 [GPU] | maq=4,416 | suelo_local=2,346,350 [↓37M→2346k] | 164s  ETA 72s
   ⚡ INTERPOL chunk 67/95 [GPU] | maq=3,665 | suelo_local=2,346,350 [↓37M→2346k] | 169s  ETA 71s
   ⚡ INTERPOL chunk 67/95 [GPU] | maq=3,665 | suelo_local=2,346,350 [↓37M→2346k] | 169s  ETA 71s
   ⚡ INTERPOL chunk 68/95 [GPU] | maq=7,946 | suelo_local=2,346,350 [↓37M→2346k] | 172s  ETA 68s
   ⚡ INTERPOL chunk 68/95 [GPU] | maq=7,946 | suelo_local=2,346,350 [↓37M→2346k] | 172s  ETA 68s
   ⚡ INTERPOL chunk 69/95 [GPU] | maq=4,404 | suelo_local=2,346,350 [↓37M→2346k] | 175s  ETA 66s
   ⚡ INTERPOL chunk 69/95 [GPU] | maq=4,404 | suelo_local=2,346,350 [↓37M→2346k] | 175s  ETA 66s
   ⚡ INTERPOL chunk 70/95 [GPU] | maq=855 | suelo_local=2,346,350 [↓37M→2346k] | 177s  ETA 63s
   ⚡ INTERPOL chunk 70/95 [GPU] | maq=855 | suelo_local=2,346,350 [↓37M→2346k] | 177s  ETA 63s
   ⚡ INTERPOL chunk 71/95 [GPU] | maq=858 | suelo_local=2,100,696 [↓33M→2100k] | 179s  ETA 60s
   ⚡ INTERPOL chunk 71/95 [GPU] | maq=858 | suelo_local=2,100,696 [↓33M→2100k] | 179s  ETA 60s
   ⚡ INTERPOL chunk 72/95 [GPU] | maq=1,084 | suelo_local=2,100,696 [↓33M→2100k] | 181s  ETA 58s
   ⚡ INTERPOL chunk 72/95 [GPU] | maq=1,084 | suelo_local=2,100,696 [↓33M→2100k] | 181s  ETA 58s
   ⚡ INTERPOL chunk 73/95 [GPU] | maq=3,313 | suelo_local=2,100,696 [↓33M→2100k] | 184s  ETA 55s
   ⚡ INTERPOL chunk 73/95 [GPU] | maq=3,313 | suelo_local=2,100,696 [↓33M→2100k] | 184s  ETA 55s
   ⚡ INTERPOL chunk 74/95 [GPU] | maq=484 | suelo_local=2,100,696 [↓33M→2100k] | 186s  ETA 53s
   ⚡ INTERPOL chunk 74/95 [GPU] | maq=484 | suelo_local=2,100,696 [↓33M→2100k] | 186s  ETA 53s
   ⚡ INTERPOL chunk 75/95 [GPU] | maq=4,156 | suelo_local=2,100,696 [↓33M→2100k] | 188s  ETA 50s
   ⚡ INTERPOL chunk 75/95 [GPU] | maq=4,156 | suelo_local=2,100,696 [↓33M→2100k] | 188s  ETA 50s
   ⚡ INTERPOL chunk 76/95 [GPU] | maq=206 | suelo_local=1,829,512 [↓29M→1829k] | 190s  ETA 47s
   ⚡ INTERPOL chunk 76/95 [GPU] | maq=206 | suelo_local=1,829,512 [↓29M→1829k] | 190s  ETA 47s
   ⚡ INTERPOL chunk 77/95 [GPU] | maq=1,126 | suelo_local=1,829,512 [↓29M→1829k] | 192s  ETA 45s
   ⚡ INTERPOL chunk 77/95 [GPU] | maq=1,126 | suelo_local=1,829,512 [↓29M→1829k] | 192s  ETA 45s
   ⚡ INTERPOL chunk 78/95 [GPU] | maq=1,095 | suelo_local=1,829,512 [↓29M→1829k] | 194s  ETA 42s
   ⚡ INTERPOL chunk 78/95 [GPU] | maq=1,095 | suelo_local=1,829,512 [↓29M→1829k] | 194s  ETA 42s
   ⚡ INTERPOL chunk 79/95 [GPU] | maq=4,409 | suelo_local=1,829,512 [↓29M→1829k] | 196s  ETA 40s
   ⚡ INTERPOL chunk 79/95 [GPU] | maq=4,409 | suelo_local=1,829,512 [↓29M→1829k] | 196s  ETA 40s
   ⚡ INTERPOL chunk 80/95 [GPU] | maq=1,462 | suelo_local=1,829,512 [↓29M→1829k] | 200s  ETA 37s
   ⚡ INTERPOL chunk 80/95 [GPU] | maq=1,462 | suelo_local=1,829,512 [↓29M→1829k] | 200s  ETA 37s
   ⚡ INTERPOL chunk 81/95 [GPU] | maq=732 | suelo_local=1,558,731 [↓25M→1558k] | 201s  ETA 35s
   ⚡ INTERPOL chunk 81/95 [GPU] | maq=732 | suelo_local=1,558,731 [↓25M→1558k] | 201s  ETA 35s
   ⚡ INTERPOL chunk 82/95 [GPU] | maq=4,296 | suelo_local=1,558,731 [↓25M→1558k] | 203s  ETA 32s
   ⚡ INTERPOL chunk 82/95 [GPU] | maq=4,296 | suelo_local=1,558,731 [↓25M→1558k] | 203s  ETA 32s
   ⚡ INTERPOL chunk 83/95 [GPU] | maq=8,900 | suelo_local=1,558,731 [↓25M→1558k] | 205s  ETA 30s
   ⚡ INTERPOL chunk 83/95 [GPU] | maq=8,900 | suelo_local=1,558,731 [↓25M→1558k] | 205s  ETA 30s
   ⚡ INTERPOL chunk 84/95 [GPU] | maq=7,813 | suelo_local=1,558,731 [↓25M→1558k] | 207s  ETA 27s
   ⚡ INTERPOL chunk 84/95 [GPU] | maq=7,813 | suelo_local=1,558,731 [↓25M→1558k] | 207s  ETA 27s
   ⚡ INTERPOL chunk 85/95 [GPU] | maq=3,711 | suelo_local=1,558,731 [↓25M→1558k] | 209s  ETA 25s
   ⚡ INTERPOL chunk 85/95 [GPU] | maq=3,711 | suelo_local=1,558,731 [↓25M→1558k] | 209s  ETA 25s
   ⚡ INTERPOL chunk 86/95 [GPU] | maq=4,913 | suelo_local=1,286,981 [↓20M→1286k] | 211s  ETA 22s
   ⚡ INTERPOL chunk 86/95 [GPU] | maq=4,913 | suelo_local=1,286,981 [↓20M→1286k] | 211s  ETA 22s
   ⚡ INTERPOL chunk 87/95 [GPU] | maq=10,755 | suelo_local=1,286,981 [↓20M→1286k] | 213s  ETA 20s
   ⚡ INTERPOL chunk 87/95 [GPU] | maq=10,755 | suelo_local=1,286,981 [↓20M→1286k] | 213s  ETA 20s
   ⚡ INTERPOL chunk 88/95 [GPU] | maq=1,818 | suelo_local=1,286,981 [↓20M→1286k] | 214s  ETA 17s
   ⚡ INTERPOL chunk 88/95 [GPU] | maq=1,818 | suelo_local=1,286,981 [↓20M→1286k] | 214s  ETA 17s
   ⚡ INTERPOL chunk 91/95 [GPU] | maq=815 | suelo_local=1,012,292 [↓16M→1012k] | 215s  ETA 9s
   ⚡ INTERPOL chunk 91/95 [GPU] | maq=815 | suelo_local=1,012,292 [↓16M→1012k] | 215s  ETA 9s
   📊 Z diagnostico: 284,542/285,046 puntos con dZ>1cm | dZ medio=6.523m | dZ max=50.422m
   📊 Z diagnostico: 284,542/285,046 puntos con dZ>1cm | dZ medio=6.523m | dZ max=50.422m
   ✅ Aplanados 285,046 puntos
   ✅ INTERPOL: 285,046 puntos aplanados
💾 DTM guardado en 224.0s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 224.0s