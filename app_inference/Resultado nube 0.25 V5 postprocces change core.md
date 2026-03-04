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
   RAM Disponible: 59.86 GB
   RAM Usada: 4.5%
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
   ⚡ Chunk 4/30 | core=3,255,564 pts | 2s elapsed  ETA 12s
   ⚡ Chunk 5/30 | core=3,988,372 pts | 2s elapsed  ETA 11s
   ⚡ Chunk 6/30 | core=1,901,925 pts | 3s elapsed  ETA 10s
   ⚡ Chunk 7/30 | core=3,161,304 pts | 3s elapsed  ETA 10s
   ⚡ Chunk 8/30 | core=3,979,412 pts | 3s elapsed  ETA 9s
   ⚡ Chunk 9/30 | core=1,757,545 pts | 4s elapsed  ETA 9s
   ⚡ Chunk 10/30 | core=2,682,968 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 11/30 | core=2,955,260 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 12/30 | core=1,426,306 pts | 5s elapsed  ETA 7s
   ⚡ Chunk 13/30 | core=3,265,826 pts | 5s elapsed  ETA 7s
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
   ⚡ Chunk 25/30 | core=3,008,817 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 27/30 | core=816,914 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 13s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 13s elapsed  ETA 0s
   ✅ Normales completadas: 13.2s  (5,816,352 pts/s)
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
✅ Inferencia completada en 160.2s - Maquinaria: 206,559 puntos (0.3%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15693 MB
   🚜 Maquinaria: 206,559 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   ⚡ Smart Merge GPU: NVIDIA GeForce RTX 5090
   🔍 Smart Merge [GPU+CPU fallback]: 76,219,117 candidatos en 153 bloques
   🔍 Smart Merge: 76,219,117 candidatos
   ⚠️ Smart Merge abortado: 31,995,611 pts exceden umbral (1,032,795 = 5× maq original). Usando clasificación original sin merge.
   🔢 Objetos encontrados: 926
   ⚡ Procesando en paralelo 926 objetos...
   ✅ Rellenados 235,102 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   📊 RSS al iniciar INTERPOL: 15693 MB
   📉 Maquinaria: 441,548 pts | Suelo: 76,095,447 pts | RAM arrays: 949 MB
   📉 Maquinaria: 441,548 | Suelo: 76,095,447 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1606.94m, mediana_suelo=1604.10m, gap=2.84m
   📐 Altura: mediana_maq=1606.94m, mediana_suelo=1604.10m, gap=2.84m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL modo: CPU
   ⚡ INTERPOL modo: CPU
   ⚡ INTERPOL chunk 1/95 [CPU] | maq=427 | suelo_local=1,395,618 [↓22M→1395k] | 6s  ETA 571s
   ⚡ INTERPOL chunk 1/95 [CPU] | maq=427 | suelo_local=1,395,618 [↓22M→1395k] | 6s  ETA 571s
   ⚡ INTERPOL chunk 2/95 [CPU] | maq=300 | suelo_local=1,395,618 [↓22M→1395k] | 8s  ETA 350s
   ⚡ INTERPOL chunk 2/95 [CPU] | maq=300 | suelo_local=1,395,618 [↓22M→1395k] | 8s  ETA 350s
   ⚡ INTERPOL chunk 3/95 [CPU] | maq=1,404 | suelo_local=1,395,618 [↓22M→1395k] | 12s  ETA 366s
   ⚡ INTERPOL chunk 3/95 [CPU] | maq=1,404 | suelo_local=1,395,618 [↓22M→1395k] | 12s  ETA 366s
   ⚡ INTERPOL chunk 4/95 [CPU] | maq=352 | suelo_local=1,395,618 [↓22M→1395k] | 13s  ETA 301s
   ⚡ INTERPOL chunk 4/95 [CPU] | maq=352 | suelo_local=1,395,618 [↓22M→1395k] | 13s  ETA 301s
   ⚡ INTERPOL chunk 6/95 [CPU] | maq=789 | suelo_local=1,664,967 [↓26M→1664k] | 15s  ETA 220s
   ⚡ INTERPOL chunk 6/95 [CPU] | maq=789 | suelo_local=1,664,967 [↓26M→1664k] | 15s  ETA 220s
   ⚡ INTERPOL chunk 7/95 [CPU] | maq=562 | suelo_local=1,664,967 [↓26M→1664k] | 16s  ETA 207s
   ⚡ INTERPOL chunk 7/95 [CPU] | maq=562 | suelo_local=1,664,967 [↓26M→1664k] | 16s  ETA 207s
   ⚡ INTERPOL chunk 8/95 [CPU] | maq=262 | suelo_local=1,664,967 [↓26M→1664k] | 18s  ETA 195s
   ⚡ INTERPOL chunk 8/95 [CPU] | maq=262 | suelo_local=1,664,967 [↓26M→1664k] | 18s  ETA 195s
   ⚡ INTERPOL chunk 9/95 [CPU] | maq=4,349 | suelo_local=1,664,967 [↓26M→1664k] | 20s  ETA 188s
   ⚡ INTERPOL chunk 9/95 [CPU] | maq=4,349 | suelo_local=1,664,967 [↓26M→1664k] | 20s  ETA 188s
   ⚡ INTERPOL chunk 10/95 [CPU] | maq=709 | suelo_local=1,664,724 [↓26M→1664k] | 21s  ETA 183s
   ⚡ INTERPOL chunk 10/95 [CPU] | maq=709 | suelo_local=1,664,724 [↓26M→1664k] | 21s  ETA 183s
   ⚡ INTERPOL chunk 11/95 [CPU] | maq=1,303 | suelo_local=1,852,811 [↓29M→1852k] | 23s  ETA 178s
   ⚡ INTERPOL chunk 11/95 [CPU] | maq=1,303 | suelo_local=1,852,811 [↓29M→1852k] | 23s  ETA 178s
   ⚡ INTERPOL chunk 12/95 [CPU] | maq=6,274 | suelo_local=1,852,811 [↓29M→1852k] | 25s  ETA 173s
   ⚡ INTERPOL chunk 12/95 [CPU] | maq=6,274 | suelo_local=1,852,811 [↓29M→1852k] | 25s  ETA 173s
   ⚡ INTERPOL chunk 13/95 [CPU] | maq=915 | suelo_local=1,852,811 [↓29M→1852k] | 27s  ETA 167s
   ⚡ INTERPOL chunk 13/95 [CPU] | maq=915 | suelo_local=1,852,811 [↓29M→1852k] | 27s  ETA 167s
   ⚡ INTERPOL chunk 14/95 [CPU] | maq=2,025 | suelo_local=1,852,811 [↓29M→1852k] | 28s  ETA 163s
   ⚡ INTERPOL chunk 14/95 [CPU] | maq=2,025 | suelo_local=1,852,811 [↓29M→1852k] | 28s  ETA 163s
   ⚡ INTERPOL chunk 15/95 [CPU] | maq=3,164 | suelo_local=1,852,589 [↓29M→1852k] | 30s  ETA 161s
   ⚡ INTERPOL chunk 15/95 [CPU] | maq=3,164 | suelo_local=1,852,589 [↓29M→1852k] | 30s  ETA 161s
   ⚡ INTERPOL chunk 16/95 [CPU] | maq=101 | suelo_local=2,111,565 [↓33M→2111k] | 32s  ETA 159s
   ⚡ INTERPOL chunk 16/95 [CPU] | maq=101 | suelo_local=2,111,565 [↓33M→2111k] | 32s  ETA 159s
   ⚡ INTERPOL chunk 17/95 [CPU] | maq=1,156 | suelo_local=2,111,565 [↓33M→2111k] | 34s  ETA 157s
   ⚡ INTERPOL chunk 17/95 [CPU] | maq=1,156 | suelo_local=2,111,565 [↓33M→2111k] | 34s  ETA 157s
   ⚡ INTERPOL chunk 18/95 [CPU] | maq=1,873 | suelo_local=2,111,565 [↓33M→2111k] | 36s  ETA 156s
   ⚡ INTERPOL chunk 18/95 [CPU] | maq=1,873 | suelo_local=2,111,565 [↓33M→2111k] | 36s  ETA 156s
   ⚡ INTERPOL chunk 19/95 [CPU] | maq=6,023 | suelo_local=2,111,565 [↓33M→2111k] | 39s  ETA 154s
   ⚡ INTERPOL chunk 19/95 [CPU] | maq=6,023 | suelo_local=2,111,565 [↓33M→2111k] | 39s  ETA 154s
   ⚡ INTERPOL chunk 20/95 [CPU] | maq=8,719 | suelo_local=2,111,324 [↓33M→2111k] | 41s  ETA 153s
   ⚡ INTERPOL chunk 20/95 [CPU] | maq=8,719 | suelo_local=2,111,324 [↓33M→2111k] | 41s  ETA 153s
   ⚡ INTERPOL chunk 21/95 [CPU] | maq=852 | suelo_local=2,380,578 [↓38M→2380k] | 46s  ETA 162s
   ⚡ INTERPOL chunk 21/95 [CPU] | maq=852 | suelo_local=2,380,578 [↓38M→2380k] | 46s  ETA 162s
   ⚡ INTERPOL chunk 22/95 [CPU] | maq=1,204 | suelo_local=2,380,578 [↓38M→2380k] | 49s  ETA 161s
   ⚡ INTERPOL chunk 22/95 [CPU] | maq=1,204 | suelo_local=2,380,578 [↓38M→2380k] | 49s  ETA 161s
   ⚡ INTERPOL chunk 23/95 [CPU] | maq=5,937 | suelo_local=2,380,578 [↓38M→2380k] | 51s  ETA 160s
   ⚡ INTERPOL chunk 23/95 [CPU] | maq=5,937 | suelo_local=2,380,578 [↓38M→2380k] | 51s  ETA 160s
   ⚡ INTERPOL chunk 24/95 [CPU] | maq=14,379 | suelo_local=2,380,578 [↓38M→2380k] | 53s  ETA 158s
   ⚡ INTERPOL chunk 24/95 [CPU] | maq=14,379 | suelo_local=2,380,578 [↓38M→2380k] | 53s  ETA 158s
   ⚡ INTERPOL chunk 25/95 [CPU] | maq=5,006 | suelo_local=2,380,334 [↓38M→2380k] | 56s  ETA 156s
   ⚡ INTERPOL chunk 25/95 [CPU] | maq=5,006 | suelo_local=2,380,334 [↓38M→2380k] | 56s  ETA 156s
   ⚡ INTERPOL chunk 26/95 [CPU] | maq=9,979 | suelo_local=2,399,086 [↓38M→2399k] | 58s  ETA 153s
   ⚡ INTERPOL chunk 26/95 [CPU] | maq=9,979 | suelo_local=2,399,086 [↓38M→2399k] | 58s  ETA 153s
   ⚡ INTERPOL chunk 27/95 [CPU] | maq=8,506 | suelo_local=2,399,086 [↓38M→2399k] | 60s  ETA 152s
   ⚡ INTERPOL chunk 27/95 [CPU] | maq=8,506 | suelo_local=2,399,086 [↓38M→2399k] | 60s  ETA 152s
   ⚡ INTERPOL chunk 28/95 [CPU] | maq=8,017 | suelo_local=2,399,086 [↓38M→2399k] | 62s  ETA 150s
   ⚡ INTERPOL chunk 28/95 [CPU] | maq=8,017 | suelo_local=2,399,086 [↓38M→2399k] | 62s  ETA 150s
   ⚡ INTERPOL chunk 29/95 [CPU] | maq=5,464 | suelo_local=2,399,086 [↓38M→2399k] | 65s  ETA 147s
   ⚡ INTERPOL chunk 29/95 [CPU] | maq=5,464 | suelo_local=2,399,086 [↓38M→2399k] | 65s  ETA 147s
   ⚡ INTERPOL chunk 30/95 [CPU] | maq=4,612 | suelo_local=2,399,086 [↓38M→2399k] | 67s  ETA 145s
   ⚡ INTERPOL chunk 30/95 [CPU] | maq=4,612 | suelo_local=2,399,086 [↓38M→2399k] | 67s  ETA 145s
   ⚡ INTERPOL chunk 31/95 [CPU] | maq=2,398 | suelo_local=2,381,402 [↓38M→2381k] | 69s  ETA 143s
   ⚡ INTERPOL chunk 31/95 [CPU] | maq=2,398 | suelo_local=2,381,402 [↓38M→2381k] | 69s  ETA 143s
   ⚡ INTERPOL chunk 32/95 [CPU] | maq=17,281 | suelo_local=2,381,402 [↓38M→2381k] | 72s  ETA 141s
   ⚡ INTERPOL chunk 32/95 [CPU] | maq=17,281 | suelo_local=2,381,402 [↓38M→2381k] | 72s  ETA 141s
   ⚡ INTERPOL chunk 33/95 [CPU] | maq=23,909 | suelo_local=2,381,402 [↓38M→2381k] | 77s  ETA 145s
   ⚡ INTERPOL chunk 33/95 [CPU] | maq=23,909 | suelo_local=2,381,402 [↓38M→2381k] | 77s  ETA 145s
   ⚡ INTERPOL chunk 34/95 [CPU] | maq=24,397 | suelo_local=2,381,402 [↓38M→2381k] | 80s  ETA 143s
   ⚡ INTERPOL chunk 34/95 [CPU] | maq=24,397 | suelo_local=2,381,402 [↓38M→2381k] | 80s  ETA 143s
   ⚡ INTERPOL chunk 35/95 [CPU] | maq=14,633 | suelo_local=2,381,402 [↓38M→2381k] | 82s  ETA 141s
   ⚡ INTERPOL chunk 35/95 [CPU] | maq=14,633 | suelo_local=2,381,402 [↓38M→2381k] | 82s  ETA 141s
   ⚡ INTERPOL chunk 36/95 [CPU] | maq=289 | suelo_local=2,366,035 [↓37M→2366k] | 85s  ETA 139s
   ⚡ INTERPOL chunk 36/95 [CPU] | maq=289 | suelo_local=2,366,035 [↓37M→2366k] | 85s  ETA 139s
   ⚡ INTERPOL chunk 37/95 [CPU] | maq=9,815 | suelo_local=2,366,035 [↓37M→2366k] | 87s  ETA 136s
   ⚡ INTERPOL chunk 37/95 [CPU] | maq=9,815 | suelo_local=2,366,035 [↓37M→2366k] | 87s  ETA 136s
   ⚡ INTERPOL chunk 38/95 [CPU] | maq=7,176 | suelo_local=2,366,035 [↓37M→2366k] | 89s  ETA 134s
   ⚡ INTERPOL chunk 38/95 [CPU] | maq=7,176 | suelo_local=2,366,035 [↓37M→2366k] | 89s  ETA 134s
   ⚡ INTERPOL chunk 39/95 [CPU] | maq=1,921 | suelo_local=2,366,035 [↓37M→2366k] | 91s  ETA 131s
   ⚡ INTERPOL chunk 39/95 [CPU] | maq=1,921 | suelo_local=2,366,035 [↓37M→2366k] | 91s  ETA 131s
   ⚡ INTERPOL chunk 40/95 [CPU] | maq=2,568 | suelo_local=2,366,035 [↓37M→2366k] | 94s  ETA 129s
   ⚡ INTERPOL chunk 40/95 [CPU] | maq=2,568 | suelo_local=2,366,035 [↓37M→2366k] | 94s  ETA 129s
   ⚡ INTERPOL chunk 42/95 [CPU] | maq=891 | suelo_local=2,356,043 [↓37M→2356k] | 96s  ETA 121s
   ⚡ INTERPOL chunk 42/95 [CPU] | maq=891 | suelo_local=2,356,043 [↓37M→2356k] | 96s  ETA 121s
   ⚡ INTERPOL chunk 43/95 [CPU] | maq=355 | suelo_local=2,356,043 [↓37M→2356k] | 99s  ETA 119s
   ⚡ INTERPOL chunk 43/95 [CPU] | maq=355 | suelo_local=2,356,043 [↓37M→2356k] | 99s  ETA 119s
   ⚡ INTERPOL chunk 44/95 [CPU] | maq=1,177 | suelo_local=2,356,043 [↓37M→2356k] | 101s  ETA 117s
   ⚡ INTERPOL chunk 44/95 [CPU] | maq=1,177 | suelo_local=2,356,043 [↓37M→2356k] | 101s  ETA 117s
   ⚡ INTERPOL chunk 45/95 [CPU] | maq=4,396 | suelo_local=2,356,043 [↓37M→2356k] | 103s  ETA 115s
   ⚡ INTERPOL chunk 45/95 [CPU] | maq=4,396 | suelo_local=2,356,043 [↓37M→2356k] | 103s  ETA 115s
   ⚡ INTERPOL chunk 46/95 [CPU] | maq=44 | suelo_local=2,353,037 [↓37M→2353k] | 105s  ETA 112s
   ⚡ INTERPOL chunk 46/95 [CPU] | maq=44 | suelo_local=2,353,037 [↓37M→2353k] | 105s  ETA 112s
   ⚡ INTERPOL chunk 47/95 [CPU] | maq=1,790 | suelo_local=2,353,037 [↓37M→2353k] | 110s  ETA 113s
   ⚡ INTERPOL chunk 47/95 [CPU] | maq=1,790 | suelo_local=2,353,037 [↓37M→2353k] | 110s  ETA 113s
   ⚡ INTERPOL chunk 49/95 [CPU] | maq=853 | suelo_local=2,353,037 [↓37M→2353k] | 113s  ETA 106s
   ⚡ INTERPOL chunk 49/95 [CPU] | maq=853 | suelo_local=2,353,037 [↓37M→2353k] | 113s  ETA 106s
   ⚡ INTERPOL chunk 50/95 [CPU] | maq=6,158 | suelo_local=2,353,037 [↓37M→2353k] | 115s  ETA 103s
   ⚡ INTERPOL chunk 50/95 [CPU] | maq=6,158 | suelo_local=2,353,037 [↓37M→2353k] | 115s  ETA 103s
   ⚡ INTERPOL chunk 51/95 [CPU] | maq=104 | suelo_local=2,356,536 [↓37M→2356k] | 117s  ETA 101s
   ⚡ INTERPOL chunk 51/95 [CPU] | maq=104 | suelo_local=2,356,536 [↓37M→2356k] | 117s  ETA 101s
   ⚡ INTERPOL chunk 52/95 [CPU] | maq=641 | suelo_local=2,356,536 [↓37M→2356k] | 119s  ETA 99s
   ⚡ INTERPOL chunk 52/95 [CPU] | maq=641 | suelo_local=2,356,536 [↓37M→2356k] | 119s  ETA 99s
   ⚡ INTERPOL chunk 54/95 [CPU] | maq=1,021 | suelo_local=2,356,536 [↓37M→2356k] | 122s  ETA 92s
   ⚡ INTERPOL chunk 54/95 [CPU] | maq=1,021 | suelo_local=2,356,536 [↓37M→2356k] | 122s  ETA 92s
   ⚡ INTERPOL chunk 55/95 [CPU] | maq=71 | suelo_local=2,356,536 [↓37M→2356k] | 124s  ETA 90s
   ⚡ INTERPOL chunk 55/95 [CPU] | maq=71 | suelo_local=2,356,536 [↓37M→2356k] | 124s  ETA 90s
   ⚡ INTERPOL chunk 56/95 [CPU] | maq=1,329 | suelo_local=2,436,464 [↓39M→2436k] | 126s  ETA 88s
   ⚡ INTERPOL chunk 56/95 [CPU] | maq=1,329 | suelo_local=2,436,464 [↓39M→2436k] | 126s  ETA 88s
   ⚡ INTERPOL chunk 57/95 [CPU] | maq=8,783 | suelo_local=2,436,464 [↓39M→2436k] | 129s  ETA 86s
   ⚡ INTERPOL chunk 57/95 [CPU] | maq=8,783 | suelo_local=2,436,464 [↓39M→2436k] | 129s  ETA 86s
   ⚡ INTERPOL chunk 59/95 [CPU] | maq=1,950 | suelo_local=2,436,464 [↓39M→2436k] | 131s  ETA 80s
   ⚡ INTERPOL chunk 59/95 [CPU] | maq=1,950 | suelo_local=2,436,464 [↓39M→2436k] | 131s  ETA 80s
   ⚡ INTERPOL chunk 61/95 [CPU] | maq=2,646 | suelo_local=2,434,898 [↓39M→2434k] | 134s  ETA 75s
   ⚡ INTERPOL chunk 61/95 [CPU] | maq=2,646 | suelo_local=2,434,898 [↓39M→2434k] | 134s  ETA 75s
   ⚡ INTERPOL chunk 62/95 [CPU] | maq=12,313 | suelo_local=2,434,898 [↓39M→2434k] | 136s  ETA 72s
   ⚡ INTERPOL chunk 62/95 [CPU] | maq=12,313 | suelo_local=2,434,898 [↓39M→2434k] | 136s  ETA 72s
   ⚡ INTERPOL chunk 63/95 [CPU] | maq=7,973 | suelo_local=2,434,898 [↓39M→2434k] | 138s  ETA 70s
   ⚡ INTERPOL chunk 63/95 [CPU] | maq=7,973 | suelo_local=2,434,898 [↓39M→2434k] | 138s  ETA 70s
   ⚡ INTERPOL chunk 64/95 [CPU] | maq=8,476 | suelo_local=2,434,898 [↓39M→2434k] | 144s  ETA 70s
   ⚡ INTERPOL chunk 64/95 [CPU] | maq=8,476 | suelo_local=2,434,898 [↓39M→2434k] | 144s  ETA 70s
   ⚡ INTERPOL chunk 65/95 [CPU] | maq=268 | suelo_local=2,434,898 [↓39M→2434k] | 147s  ETA 68s
   ⚡ INTERPOL chunk 65/95 [CPU] | maq=268 | suelo_local=2,434,898 [↓39M→2434k] | 147s  ETA 68s
   ⚡ INTERPOL chunk 66/95 [CPU] | maq=2,754 | suelo_local=2,332,843 [↓37M→2332k] | 149s  ETA 66s
   ⚡ INTERPOL chunk 66/95 [CPU] | maq=2,754 | suelo_local=2,332,843 [↓37M→2332k] | 149s  ETA 66s
   ⚡ INTERPOL chunk 67/95 [CPU] | maq=5,165 | suelo_local=2,332,843 [↓37M→2332k] | 152s  ETA 63s
   ⚡ INTERPOL chunk 67/95 [CPU] | maq=5,165 | suelo_local=2,332,843 [↓37M→2332k] | 152s  ETA 63s
   ⚡ INTERPOL chunk 68/95 [CPU] | maq=15,761 | suelo_local=2,332,843 [↓37M→2332k] | 154s  ETA 61s
   ⚡ INTERPOL chunk 68/95 [CPU] | maq=15,761 | suelo_local=2,332,843 [↓37M→2332k] | 154s  ETA 61s
   ⚡ INTERPOL chunk 69/95 [CPU] | maq=5,957 | suelo_local=2,332,843 [↓37M→2332k] | 156s  ETA 59s
   ⚡ INTERPOL chunk 69/95 [CPU] | maq=5,957 | suelo_local=2,332,843 [↓37M→2332k] | 156s  ETA 59s
   ⚡ INTERPOL chunk 70/95 [CPU] | maq=719 | suelo_local=2,332,843 [↓37M→2332k] | 158s  ETA 56s
   ⚡ INTERPOL chunk 70/95 [CPU] | maq=719 | suelo_local=2,332,843 [↓37M→2332k] | 158s  ETA 56s
   ⚡ INTERPOL chunk 71/95 [CPU] | maq=1,968 | suelo_local=2,083,847 [↓33M→2083k] | 160s  ETA 54s
   ⚡ INTERPOL chunk 71/95 [CPU] | maq=1,968 | suelo_local=2,083,847 [↓33M→2083k] | 160s  ETA 54s
   ⚡ INTERPOL chunk 72/95 [CPU] | maq=1,973 | suelo_local=2,083,847 [↓33M→2083k] | 162s  ETA 52s
   ⚡ INTERPOL chunk 72/95 [CPU] | maq=1,973 | suelo_local=2,083,847 [↓33M→2083k] | 162s  ETA 52s
   ⚡ INTERPOL chunk 73/95 [CPU] | maq=6,638 | suelo_local=2,083,847 [↓33M→2083k] | 164s  ETA 49s
   ⚡ INTERPOL chunk 73/95 [CPU] | maq=6,638 | suelo_local=2,083,847 [↓33M→2083k] | 164s  ETA 49s
   ⚡ INTERPOL chunk 74/95 [CPU] | maq=1,055 | suelo_local=2,083,847 [↓33M→2083k] | 166s  ETA 47s
   ⚡ INTERPOL chunk 74/95 [CPU] | maq=1,055 | suelo_local=2,083,847 [↓33M→2083k] | 166s  ETA 47s
   ⚡ INTERPOL chunk 75/95 [CPU] | maq=6,436 | suelo_local=2,083,847 [↓33M→2083k] | 168s  ETA 45s
   ⚡ INTERPOL chunk 75/95 [CPU] | maq=6,436 | suelo_local=2,083,847 [↓33M→2083k] | 168s  ETA 45s
   ⚡ INTERPOL chunk 76/95 [CPU] | maq=338 | suelo_local=1,812,437 [↓29M→1812k] | 170s  ETA 43s
   ⚡ INTERPOL chunk 76/95 [CPU] | maq=338 | suelo_local=1,812,437 [↓29M→1812k] | 170s  ETA 43s
   ⚡ INTERPOL chunk 77/95 [CPU] | maq=1,925 | suelo_local=1,812,437 [↓29M→1812k] | 172s  ETA 40s
   ⚡ INTERPOL chunk 77/95 [CPU] | maq=1,925 | suelo_local=1,812,437 [↓29M→1812k] | 172s  ETA 40s
   ⚡ INTERPOL chunk 78/95 [CPU] | maq=2,379 | suelo_local=1,812,437 [↓29M→1812k] | 177s  ETA 39s
   ⚡ INTERPOL chunk 78/95 [CPU] | maq=2,379 | suelo_local=1,812,437 [↓29M→1812k] | 177s  ETA 39s
   ⚡ INTERPOL chunk 79/95 [CPU] | maq=7,621 | suelo_local=1,812,437 [↓29M→1812k] | 179s  ETA 36s
   ⚡ INTERPOL chunk 79/95 [CPU] | maq=7,621 | suelo_local=1,812,437 [↓29M→1812k] | 179s  ETA 36s
   ⚡ INTERPOL chunk 80/95 [CPU] | maq=1,810 | suelo_local=1,812,437 [↓29M→1812k] | 181s  ETA 34s
   ⚡ INTERPOL chunk 80/95 [CPU] | maq=1,810 | suelo_local=1,812,437 [↓29M→1812k] | 181s  ETA 34s
   ⚡ INTERPOL chunk 81/95 [CPU] | maq=466 | suelo_local=1,541,818 [↓24M→1541k] | 183s  ETA 32s
   ⚡ INTERPOL chunk 81/95 [CPU] | maq=466 | suelo_local=1,541,818 [↓24M→1541k] | 183s  ETA 32s
   ⚡ INTERPOL chunk 82/95 [CPU] | maq=6,673 | suelo_local=1,541,818 [↓24M→1541k] | 184s  ETA 29s
   ⚡ INTERPOL chunk 82/95 [CPU] | maq=6,673 | suelo_local=1,541,818 [↓24M→1541k] | 184s  ETA 29s
   ⚡ INTERPOL chunk 83/95 [CPU] | maq=21,114 | suelo_local=1,541,818 [↓24M→1541k] | 186s  ETA 27s
   ⚡ INTERPOL chunk 83/95 [CPU] | maq=21,114 | suelo_local=1,541,818 [↓24M→1541k] | 186s  ETA 27s
   ⚡ INTERPOL chunk 84/95 [CPU] | maq=20,401 | suelo_local=1,541,818 [↓24M→1541k] | 187s  ETA 24s
   ⚡ INTERPOL chunk 84/95 [CPU] | maq=20,401 | suelo_local=1,541,818 [↓24M→1541k] | 187s  ETA 24s
   ⚡ INTERPOL chunk 85/95 [CPU] | maq=10,243 | suelo_local=1,541,818 [↓24M→1541k] | 188s  ETA 22s
   ⚡ INTERPOL chunk 85/95 [CPU] | maq=10,243 | suelo_local=1,541,818 [↓24M→1541k] | 188s  ETA 22s
   ⚡ INTERPOL chunk 86/95 [CPU] | maq=13,776 | suelo_local=1,270,712 [↓20M→1270k] | 190s  ETA 20s
   ⚡ INTERPOL chunk 86/95 [CPU] | maq=13,776 | suelo_local=1,270,712 [↓20M→1270k] | 190s  ETA 20s
   ⚡ INTERPOL chunk 87/95 [CPU] | maq=30,544 | suelo_local=1,270,712 [↓20M→1270k] | 191s  ETA 18s
   ⚡ INTERPOL chunk 87/95 [CPU] | maq=30,544 | suelo_local=1,270,712 [↓20M→1270k] | 191s  ETA 18s
   ⚡ INTERPOL chunk 88/95 [CPU] | maq=4,187 | suelo_local=1,270,712 [↓20M→1270k] | 192s  ETA 15s
   ⚡ INTERPOL chunk 88/95 [CPU] | maq=4,187 | suelo_local=1,270,712 [↓20M→1270k] | 192s  ETA 15s
   ⚡ INTERPOL chunk 89/95 [CPU] | maq=7 | suelo_local=1,270,712 [↓20M→1270k] | 194s  ETA 13s
   ⚡ INTERPOL chunk 89/95 [CPU] | maq=7 | suelo_local=1,270,712 [↓20M→1270k] | 194s  ETA 13s
   ⚡ INTERPOL chunk 91/95 [CPU] | maq=1,349 | suelo_local=996,579 [↓16M→996k] | 195s  ETA 9s
   ⚡ INTERPOL chunk 91/95 [CPU] | maq=1,349 | suelo_local=996,579 [↓16M→996k] | 195s  ETA 9s
   📊 INTERPOL backend: 0 tiles GPU, 84 tiles CPU fallback
   📊 INTERPOL backend: 0 tiles GPU, 84 tiles CPU fallback
   📊 Z diagnostico: 398,675/441,548 puntos con dZ>1cm | dZ medio=0.439m | dZ max=9.790m
   📊 Z diagnostico: 398,675/441,548 puntos con dZ>1cm | dZ medio=0.439m | dZ max=9.790m
   ✅ Aplanados 441,548 puntos
   ✅ INTERPOL: 441,548 puntos aplanados
💾 DTM guardado en 203.5s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 203.5s
