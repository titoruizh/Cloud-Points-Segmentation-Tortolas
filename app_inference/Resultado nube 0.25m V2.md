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
   RAM Disponible: 59.95 GB
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
   ⚡ Chunk 25/30 | core=3,008,817 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 27/30 | core=816,914 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 10s elapsed  ETA 0s
   ✅ Normales completadas: 10.4s  (7,346,150 pts/s)
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
✅ Inferencia completada en 158.7s - Maquinaria: 286,257 puntos (0.4%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15646 MB
   🚜 Maquinaria: 286,257 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   🔍 Smart Merge: 76,237,486 candidatos en 153 bloques
   🔍 Smart Merge: 76,237,486 candidatos
   ✨ Smart Merge: 710,804 puntos unidos
   🔢 Objetos encontrados: 1096
   ⚡ Procesando en paralelo 1096 objetos...
   ✅ Rellenados 415,266 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   📊 RSS al iniciar INTERPOL: 15646 MB
   📉 Maquinaria: 1,412,327 pts | Suelo: 75,124,668 pts | RAM arrays: 949 MB
   📉 Maquinaria: 1,412,327 | Suelo: 75,124,668 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1605.67m, mediana_suelo=1604.09m, gap=1.58m
   📐 Altura: mediana_maq=1605.67m, mediana_suelo=1604.09m, gap=1.58m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL chunk 1/95 | maq=1,614 | suelo_local=1,377,727 [↓22M→1377k] | 7s  ETA 686s
   ⚡ INTERPOL chunk 1/95 | maq=1,614 | suelo_local=1,377,727 [↓22M→1377k] | 7s  ETA 686s
   ⚡ INTERPOL chunk 2/95 | maq=638 | suelo_local=1,377,727 [↓22M→1377k] | 9s  ETA 402s
   ⚡ INTERPOL chunk 2/95 | maq=638 | suelo_local=1,377,727 [↓22M→1377k] | 9s  ETA 402s
   ⚡ INTERPOL chunk 3/95 | maq=6,629 | suelo_local=1,377,727 [↓22M→1377k] | 10s  ETA 306s
   ⚡ INTERPOL chunk 3/95 | maq=6,629 | suelo_local=1,377,727 [↓22M→1377k] | 10s  ETA 306s
   ⚡ INTERPOL chunk 4/95 | maq=2,224 | suelo_local=1,377,727 [↓22M→1377k] | 11s  ETA 259s
   ⚡ INTERPOL chunk 4/95 | maq=2,224 | suelo_local=1,377,727 [↓22M→1377k] | 11s  ETA 259s
   ⚡ INTERPOL chunk 6/95 | maq=2,644 | suelo_local=1,643,662 [↓26M→1643k] | 13s  ETA 192s
   ⚡ INTERPOL chunk 6/95 | maq=2,644 | suelo_local=1,643,662 [↓26M→1643k] | 13s  ETA 192s
   ⚡ INTERPOL chunk 7/95 | maq=2,274 | suelo_local=1,643,662 [↓26M→1643k] | 14s  ETA 181s
   ⚡ INTERPOL chunk 7/95 | maq=2,274 | suelo_local=1,643,662 [↓26M→1643k] | 14s  ETA 181s
   ⚡ INTERPOL chunk 8/95 | maq=2,230 | suelo_local=1,643,662 [↓26M→1643k] | 16s  ETA 175s
   ⚡ INTERPOL chunk 8/95 | maq=2,230 | suelo_local=1,643,662 [↓26M→1643k] | 16s  ETA 175s
   ⚡ INTERPOL chunk 9/95 | maq=17,262 | suelo_local=1,643,662 [↓26M→1643k] | 18s  ETA 170s
   ⚡ INTERPOL chunk 9/95 | maq=17,262 | suelo_local=1,643,662 [↓26M→1643k] | 18s  ETA 170s
   ⚡ INTERPOL chunk 10/95 | maq=2,392 | suelo_local=1,643,444 [↓26M→1643k] | 19s  ETA 165s
   ⚡ INTERPOL chunk 10/95 | maq=2,392 | suelo_local=1,643,444 [↓26M→1643k] | 19s  ETA 165s
   ⚡ INTERPOL chunk 11/95 | maq=7,108 | suelo_local=1,823,047 [↓29M→1823k] | 21s  ETA 161s
   ⚡ INTERPOL chunk 11/95 | maq=7,108 | suelo_local=1,823,047 [↓29M→1823k] | 21s  ETA 161s
   ⚡ INTERPOL chunk 12/95 | maq=20,104 | suelo_local=1,823,047 [↓29M→1823k] | 23s  ETA 158s
   ⚡ INTERPOL chunk 12/95 | maq=20,104 | suelo_local=1,823,047 [↓29M→1823k] | 23s  ETA 158s
   ⚡ INTERPOL chunk 13/95 | maq=3,209 | suelo_local=1,823,047 [↓29M→1823k] | 25s  ETA 156s
   ⚡ INTERPOL chunk 13/95 | maq=3,209 | suelo_local=1,823,047 [↓29M→1823k] | 25s  ETA 156s
   ⚡ INTERPOL chunk 14/95 | maq=7,630 | suelo_local=1,823,047 [↓29M→1823k] | 27s  ETA 154s
   ⚡ INTERPOL chunk 14/95 | maq=7,630 | suelo_local=1,823,047 [↓29M→1823k] | 27s  ETA 154s
   ⚡ INTERPOL chunk 15/95 | maq=13,769 | suelo_local=1,822,848 [↓29M→1822k] | 28s  ETA 151s
   ⚡ INTERPOL chunk 15/95 | maq=13,769 | suelo_local=1,822,848 [↓29M→1822k] | 28s  ETA 151s
   ⚡ INTERPOL chunk 16/95 | maq=124 | suelo_local=2,077,591 [↓33M→2077k] | 30s  ETA 150s
   ⚡ INTERPOL chunk 16/95 | maq=124 | suelo_local=2,077,591 [↓33M→2077k] | 30s  ETA 150s
   ⚡ INTERPOL chunk 17/95 | maq=2,806 | suelo_local=2,077,591 [↓33M→2077k] | 32s  ETA 148s
   ⚡ INTERPOL chunk 17/95 | maq=2,806 | suelo_local=2,077,591 [↓33M→2077k] | 32s  ETA 148s
   ⚡ INTERPOL chunk 18/95 | maq=10,135 | suelo_local=2,077,591 [↓33M→2077k] | 34s  ETA 145s
   ⚡ INTERPOL chunk 18/95 | maq=10,135 | suelo_local=2,077,591 [↓33M→2077k] | 34s  ETA 145s
   ⚡ INTERPOL chunk 19/95 | maq=22,778 | suelo_local=2,077,591 [↓33M→2077k] | 38s  ETA 153s
   ⚡ INTERPOL chunk 19/95 | maq=22,778 | suelo_local=2,077,591 [↓33M→2077k] | 38s  ETA 153s
   ⚡ INTERPOL chunk 20/95 | maq=34,322 | suelo_local=2,077,408 [↓33M→2077k] | 40s  ETA 151s
   ⚡ INTERPOL chunk 20/95 | maq=34,322 | suelo_local=2,077,408 [↓33M→2077k] | 40s  ETA 151s
   ⚡ INTERPOL chunk 21/95 | maq=3,105 | suelo_local=2,347,411 [↓37M→2347k] | 43s  ETA 151s
   ⚡ INTERPOL chunk 21/95 | maq=3,105 | suelo_local=2,347,411 [↓37M→2347k] | 43s  ETA 151s
   ⚡ INTERPOL chunk 22/95 | maq=4,241 | suelo_local=2,347,411 [↓37M→2347k] | 45s  ETA 149s
   ⚡ INTERPOL chunk 22/95 | maq=4,241 | suelo_local=2,347,411 [↓37M→2347k] | 45s  ETA 149s
   ⚡ INTERPOL chunk 23/95 | maq=18,976 | suelo_local=2,347,411 [↓37M→2347k] | 47s  ETA 147s
   ⚡ INTERPOL chunk 23/95 | maq=18,976 | suelo_local=2,347,411 [↓37M→2347k] | 47s  ETA 147s
   ⚡ INTERPOL chunk 24/95 | maq=60,440 | suelo_local=2,347,411 [↓37M→2347k] | 49s  ETA 145s
   ⚡ INTERPOL chunk 24/95 | maq=60,440 | suelo_local=2,347,411 [↓37M→2347k] | 49s  ETA 145s
   ⚡ INTERPOL chunk 25/95 | maq=16,419 | suelo_local=2,347,172 [↓37M→2347k] | 51s  ETA 143s
   ⚡ INTERPOL chunk 25/95 | maq=16,419 | suelo_local=2,347,172 [↓37M→2347k] | 51s  ETA 143s
   ⚡ INTERPOL chunk 26/95 | maq=25,438 | suelo_local=2,369,480 [↓37M→2369k] | 53s  ETA 141s
   ⚡ INTERPOL chunk 26/95 | maq=25,438 | suelo_local=2,369,480 [↓37M→2369k] | 53s  ETA 141s
   ⚡ INTERPOL chunk 27/95 | maq=27,497 | suelo_local=2,369,480 [↓37M→2369k] | 55s  ETA 140s
   ⚡ INTERPOL chunk 27/95 | maq=27,497 | suelo_local=2,369,480 [↓37M→2369k] | 55s  ETA 140s
   ⚡ INTERPOL chunk 28/95 | maq=23,566 | suelo_local=2,369,480 [↓37M→2369k] | 57s  ETA 138s
   ⚡ INTERPOL chunk 28/95 | maq=23,566 | suelo_local=2,369,480 [↓37M→2369k] | 57s  ETA 138s
   ⚡ INTERPOL chunk 29/95 | maq=23,681 | suelo_local=2,369,480 [↓37M→2369k] | 60s  ETA 135s
   ⚡ INTERPOL chunk 29/95 | maq=23,681 | suelo_local=2,369,480 [↓37M→2369k] | 60s  ETA 135s
   ⚡ INTERPOL chunk 30/95 | maq=17,179 | suelo_local=2,369,480 [↓37M→2369k] | 62s  ETA 133s
   ⚡ INTERPOL chunk 30/95 | maq=17,179 | suelo_local=2,369,480 [↓37M→2369k] | 62s  ETA 133s
   ⚡ INTERPOL chunk 31/95 | maq=8,340 | suelo_local=2,352,293 [↓37M→2352k] | 64s  ETA 131s
   ⚡ INTERPOL chunk 31/95 | maq=8,340 | suelo_local=2,352,293 [↓37M→2352k] | 64s  ETA 131s
   ⚡ INTERPOL chunk 32/95 | maq=42,799 | suelo_local=2,352,293 [↓37M→2352k] | 66s  ETA 129s
   ⚡ INTERPOL chunk 32/95 | maq=42,799 | suelo_local=2,352,293 [↓37M→2352k] | 66s  ETA 129s
   ⚡ INTERPOL chunk 33/95 | maq=67,490 | suelo_local=2,352,293 [↓37M→2352k] | 70s  ETA 131s
   ⚡ INTERPOL chunk 33/95 | maq=67,490 | suelo_local=2,352,293 [↓37M→2352k] | 70s  ETA 131s
   ⚡ INTERPOL chunk 34/95 | maq=60,469 | suelo_local=2,352,293 [↓37M→2352k] | 72s  ETA 129s
   ⚡ INTERPOL chunk 34/95 | maq=60,469 | suelo_local=2,352,293 [↓37M→2352k] | 72s  ETA 129s
   ⚡ INTERPOL chunk 35/95 | maq=37,136 | suelo_local=2,352,293 [↓37M→2352k] | 74s  ETA 127s
   ⚡ INTERPOL chunk 35/95 | maq=37,136 | suelo_local=2,352,293 [↓37M→2352k] | 74s  ETA 127s
   ⚡ INTERPOL chunk 36/95 | maq=1,169 | suelo_local=2,337,002 [↓37M→2337k] | 76s  ETA 125s
   ⚡ INTERPOL chunk 36/95 | maq=1,169 | suelo_local=2,337,002 [↓37M→2337k] | 76s  ETA 125s
   ⚡ INTERPOL chunk 37/95 | maq=27,390 | suelo_local=2,337,002 [↓37M→2337k] | 78s  ETA 123s
   ⚡ INTERPOL chunk 37/95 | maq=27,390 | suelo_local=2,337,002 [↓37M→2337k] | 78s  ETA 123s
   ⚡ INTERPOL chunk 38/95 | maq=25,740 | suelo_local=2,337,002 [↓37M→2337k] | 81s  ETA 121s
   ⚡ INTERPOL chunk 38/95 | maq=25,740 | suelo_local=2,337,002 [↓37M→2337k] | 81s  ETA 121s
   ⚡ INTERPOL chunk 39/95 | maq=11,534 | suelo_local=2,337,002 [↓37M→2337k] | 83s  ETA 119s
   ⚡ INTERPOL chunk 39/95 | maq=11,534 | suelo_local=2,337,002 [↓37M→2337k] | 83s  ETA 119s
   ⚡ INTERPOL chunk 40/95 | maq=7,606 | suelo_local=2,337,002 [↓37M→2337k] | 85s  ETA 117s
   ⚡ INTERPOL chunk 40/95 | maq=7,606 | suelo_local=2,337,002 [↓37M→2337k] | 85s  ETA 117s
   ⚡ INTERPOL chunk 41/95 | maq=15 | suelo_local=2,324,345 [↓37M→2324k] | 87s  ETA 114s
   ⚡ INTERPOL chunk 41/95 | maq=15 | suelo_local=2,324,345 [↓37M→2324k] | 87s  ETA 114s
   ⚡ INTERPOL chunk 42/95 | maq=6,760 | suelo_local=2,324,345 [↓37M→2324k] | 89s  ETA 112s
   ⚡ INTERPOL chunk 42/95 | maq=6,760 | suelo_local=2,324,345 [↓37M→2324k] | 89s  ETA 112s
   ⚡ INTERPOL chunk 43/95 | maq=800 | suelo_local=2,324,345 [↓37M→2324k] | 91s  ETA 110s
   ⚡ INTERPOL chunk 43/95 | maq=800 | suelo_local=2,324,345 [↓37M→2324k] | 91s  ETA 110s
   ⚡ INTERPOL chunk 44/95 | maq=6,680 | suelo_local=2,324,345 [↓37M→2324k] | 93s  ETA 108s
   ⚡ INTERPOL chunk 44/95 | maq=6,680 | suelo_local=2,324,345 [↓37M→2324k] | 93s  ETA 108s
   ⚡ INTERPOL chunk 45/95 | maq=16,922 | suelo_local=2,324,345 [↓37M→2324k] | 95s  ETA 106s
   ⚡ INTERPOL chunk 45/95 | maq=16,922 | suelo_local=2,324,345 [↓37M→2324k] | 95s  ETA 106s
   ⚡ INTERPOL chunk 46/95 | maq=207 | suelo_local=2,320,426 [↓37M→2320k] | 97s  ETA 104s
   ⚡ INTERPOL chunk 46/95 | maq=207 | suelo_local=2,320,426 [↓37M→2320k] | 97s  ETA 104s
   ⚡ INTERPOL chunk 47/95 | maq=4,953 | suelo_local=2,320,426 [↓37M→2320k] | 102s  ETA 104s
   ⚡ INTERPOL chunk 47/95 | maq=4,953 | suelo_local=2,320,426 [↓37M→2320k] | 102s  ETA 104s
   ⚡ INTERPOL chunk 49/95 | maq=4,354 | suelo_local=2,320,426 [↓37M→2320k] | 104s  ETA 97s
   ⚡ INTERPOL chunk 49/95 | maq=4,354 | suelo_local=2,320,426 [↓37M→2320k] | 104s  ETA 97s
   ⚡ INTERPOL chunk 50/95 | maq=52,480 | suelo_local=2,320,426 [↓37M→2320k] | 106s  ETA 96s
   ⚡ INTERPOL chunk 50/95 | maq=52,480 | suelo_local=2,320,426 [↓37M→2320k] | 106s  ETA 96s
   ⚡ INTERPOL chunk 51/95 | maq=823 | suelo_local=2,325,439 [↓37M→2325k] | 108s  ETA 94s
   ⚡ INTERPOL chunk 51/95 | maq=823 | suelo_local=2,325,439 [↓37M→2325k] | 108s  ETA 94s
   ⚡ INTERPOL chunk 52/95 | maq=2,675 | suelo_local=2,325,439 [↓37M→2325k] | 111s  ETA 91s
   ⚡ INTERPOL chunk 52/95 | maq=2,675 | suelo_local=2,325,439 [↓37M→2325k] | 111s  ETA 91s
   ⚡ INTERPOL chunk 54/95 | maq=4,334 | suelo_local=2,325,439 [↓37M→2325k] | 113s  ETA 85s
   ⚡ INTERPOL chunk 54/95 | maq=4,334 | suelo_local=2,325,439 [↓37M→2325k] | 113s  ETA 85s
   ⚡ INTERPOL chunk 55/95 | maq=265 | suelo_local=2,325,439 [↓37M→2325k] | 115s  ETA 83s
   ⚡ INTERPOL chunk 55/95 | maq=265 | suelo_local=2,325,439 [↓37M→2325k] | 115s  ETA 83s
   ⚡ INTERPOL chunk 56/95 | maq=5,892 | suelo_local=2,412,130 [↓38M→2412k] | 117s  ETA 81s
   ⚡ INTERPOL chunk 56/95 | maq=5,892 | suelo_local=2,412,130 [↓38M→2412k] | 117s  ETA 81s
   ⚡ INTERPOL chunk 57/95 | maq=35,671 | suelo_local=2,412,130 [↓38M→2412k] | 119s  ETA 79s
   ⚡ INTERPOL chunk 57/95 | maq=35,671 | suelo_local=2,412,130 [↓38M→2412k] | 119s  ETA 79s
   ⚡ INTERPOL chunk 58/95 | maq=10 | suelo_local=2,412,130 [↓38M→2412k] | 121s  ETA 77s
   ⚡ INTERPOL chunk 58/95 | maq=10 | suelo_local=2,412,130 [↓38M→2412k] | 121s  ETA 77s
   ⚡ INTERPOL chunk 59/95 | maq=13,378 | suelo_local=2,412,130 [↓38M→2412k] | 124s  ETA 75s
   ⚡ INTERPOL chunk 59/95 | maq=13,378 | suelo_local=2,412,130 [↓38M→2412k] | 124s  ETA 75s
   ⚡ INTERPOL chunk 61/95 | maq=45,603 | suelo_local=2,411,670 [↓38M→2411k] | 126s  ETA 70s
   ⚡ INTERPOL chunk 61/95 | maq=45,603 | suelo_local=2,411,670 [↓38M→2411k] | 126s  ETA 70s
   ⚡ INTERPOL chunk 62/95 | maq=36,769 | suelo_local=2,411,670 [↓38M→2411k] | 128s  ETA 68s
   ⚡ INTERPOL chunk 62/95 | maq=36,769 | suelo_local=2,411,670 [↓38M→2411k] | 128s  ETA 68s
   ⚡ INTERPOL chunk 63/95 | maq=34,161 | suelo_local=2,411,670 [↓38M→2411k] | 130s  ETA 66s
   ⚡ INTERPOL chunk 63/95 | maq=34,161 | suelo_local=2,411,670 [↓38M→2411k] | 130s  ETA 66s
   ⚡ INTERPOL chunk 64/95 | maq=29,203 | suelo_local=2,411,670 [↓38M→2411k] | 134s  ETA 65s
   ⚡ INTERPOL chunk 64/95 | maq=29,203 | suelo_local=2,411,670 [↓38M→2411k] | 134s  ETA 65s
   ⚡ INTERPOL chunk 65/95 | maq=1,256 | suelo_local=2,411,670 [↓38M→2411k] | 137s  ETA 63s
   ⚡ INTERPOL chunk 65/95 | maq=1,256 | suelo_local=2,411,670 [↓38M→2411k] | 137s  ETA 63s
   ⚡ INTERPOL chunk 66/95 | maq=13,736 | suelo_local=2,312,402 [↓37M→2312k] | 139s  ETA 61s
   ⚡ INTERPOL chunk 66/95 | maq=13,736 | suelo_local=2,312,402 [↓37M→2312k] | 139s  ETA 61s
   ⚡ INTERPOL chunk 67/95 | maq=18,118 | suelo_local=2,312,402 [↓37M→2312k] | 141s  ETA 59s
   ⚡ INTERPOL chunk 67/95 | maq=18,118 | suelo_local=2,312,402 [↓37M→2312k] | 141s  ETA 59s
   ⚡ INTERPOL chunk 68/95 | maq=56,971 | suelo_local=2,312,402 [↓37M→2312k] | 143s  ETA 57s
   ⚡ INTERPOL chunk 68/95 | maq=56,971 | suelo_local=2,312,402 [↓37M→2312k] | 143s  ETA 57s
   ⚡ INTERPOL chunk 69/95 | maq=30,381 | suelo_local=2,312,402 [↓37M→2312k] | 145s  ETA 55s
   ⚡ INTERPOL chunk 69/95 | maq=30,381 | suelo_local=2,312,402 [↓37M→2312k] | 145s  ETA 55s
   ⚡ INTERPOL chunk 70/95 | maq=3,362 | suelo_local=2,312,402 [↓37M→2312k] | 147s  ETA 52s
   ⚡ INTERPOL chunk 70/95 | maq=3,362 | suelo_local=2,312,402 [↓37M→2312k] | 147s  ETA 52s
   ⚡ INTERPOL chunk 71/95 | maq=9,090 | suelo_local=2,067,989 [↓33M→2067k] | 149s  ETA 50s
   ⚡ INTERPOL chunk 71/95 | maq=9,090 | suelo_local=2,067,989 [↓33M→2067k] | 149s  ETA 50s
   ⚡ INTERPOL chunk 72/95 | maq=9,910 | suelo_local=2,067,989 [↓33M→2067k] | 151s  ETA 48s
   ⚡ INTERPOL chunk 72/95 | maq=9,910 | suelo_local=2,067,989 [↓33M→2067k] | 151s  ETA 48s
   ⚡ INTERPOL chunk 73/95 | maq=21,260 | suelo_local=2,067,989 [↓33M→2067k] | 153s  ETA 46s
   ⚡ INTERPOL chunk 73/95 | maq=21,260 | suelo_local=2,067,989 [↓33M→2067k] | 153s  ETA 46s
   ⚡ INTERPOL chunk 74/95 | maq=3,319 | suelo_local=2,067,989 [↓33M→2067k] | 155s  ETA 44s
   ⚡ INTERPOL chunk 74/95 | maq=3,319 | suelo_local=2,067,989 [↓33M→2067k] | 155s  ETA 44s
   ⚡ INTERPOL chunk 75/95 | maq=17,755 | suelo_local=2,067,989 [↓33M→2067k] | 158s  ETA 42s
   ⚡ INTERPOL chunk 75/95 | maq=17,755 | suelo_local=2,067,989 [↓33M→2067k] | 158s  ETA 42s
   ⚡ INTERPOL chunk 76/95 | maq=2,663 | suelo_local=1,797,008 [↓28M→1797k] | 160s  ETA 40s
   ⚡ INTERPOL chunk 76/95 | maq=2,663 | suelo_local=1,797,008 [↓28M→1797k] | 160s  ETA 40s
   ⚡ INTERPOL chunk 77/95 | maq=6,052 | suelo_local=1,797,008 [↓28M→1797k] | 161s  ETA 38s
   ⚡ INTERPOL chunk 77/95 | maq=6,052 | suelo_local=1,797,008 [↓28M→1797k] | 161s  ETA 38s
   ⚡ INTERPOL chunk 78/95 | maq=7,317 | suelo_local=1,797,008 [↓28M→1797k] | 165s  ETA 36s
   ⚡ INTERPOL chunk 78/95 | maq=7,317 | suelo_local=1,797,008 [↓28M→1797k] | 165s  ETA 36s
   ⚡ INTERPOL chunk 79/95 | maq=25,178 | suelo_local=1,797,008 [↓28M→1797k] | 167s  ETA 34s
   ⚡ INTERPOL chunk 79/95 | maq=25,178 | suelo_local=1,797,008 [↓28M→1797k] | 167s  ETA 34s
   ⚡ INTERPOL chunk 80/95 | maq=5,040 | suelo_local=1,797,008 [↓28M→1797k] | 169s  ETA 32s
   ⚡ INTERPOL chunk 80/95 | maq=5,040 | suelo_local=1,797,008 [↓28M→1797k] | 169s  ETA 32s
   ⚡ INTERPOL chunk 81/95 | maq=2,332 | suelo_local=1,528,679 [↓24M→1528k] | 170s  ETA 29s
   ⚡ INTERPOL chunk 81/95 | maq=2,332 | suelo_local=1,528,679 [↓24M→1528k] | 170s  ETA 29s
   ⚡ INTERPOL chunk 82/95 | maq=16,543 | suelo_local=1,528,679 [↓24M→1528k] | 172s  ETA 27s
   ⚡ INTERPOL chunk 82/95 | maq=16,543 | suelo_local=1,528,679 [↓24M→1528k] | 172s  ETA 27s
   ⚡ INTERPOL chunk 83/95 | maq=46,049 | suelo_local=1,528,679 [↓24M→1528k] | 173s  ETA 25s
   ⚡ INTERPOL chunk 83/95 | maq=46,049 | suelo_local=1,528,679 [↓24M→1528k] | 173s  ETA 25s
   ⚡ INTERPOL chunk 84/95 | maq=35,140 | suelo_local=1,528,679 [↓24M→1528k] | 175s  ETA 23s
   ⚡ INTERPOL chunk 84/95 | maq=35,140 | suelo_local=1,528,679 [↓24M→1528k] | 175s  ETA 23s
   ⚡ INTERPOL chunk 85/95 | maq=19,632 | suelo_local=1,528,679 [↓24M→1528k] | 176s  ETA 21s
   ⚡ INTERPOL chunk 85/95 | maq=19,632 | suelo_local=1,528,679 [↓24M→1528k] | 176s  ETA 21s
   ⚡ INTERPOL chunk 86/95 | maq=24,984 | suelo_local=1,263,097 [↓20M→1263k] | 177s  ETA 19s
   ⚡ INTERPOL chunk 86/95 | maq=24,984 | suelo_local=1,263,097 [↓20M→1263k] | 177s  ETA 19s
   ⚡ INTERPOL chunk 87/95 | maq=51,276 | suelo_local=1,263,097 [↓20M→1263k] | 178s  ETA 16s
   ⚡ INTERPOL chunk 87/95 | maq=51,276 | suelo_local=1,263,097 [↓20M→1263k] | 178s  ETA 16s
   ⚡ INTERPOL chunk 88/95 | maq=8,783 | suelo_local=1,263,097 [↓20M→1263k] | 180s  ETA 14s
   ⚡ INTERPOL chunk 88/95 | maq=8,783 | suelo_local=1,263,097 [↓20M→1263k] | 180s  ETA 14s
   ⚡ INTERPOL chunk 89/95 | maq=17 | suelo_local=1,263,097 [↓20M→1263k] | 181s  ETA 12s
   ⚡ INTERPOL chunk 89/95 | maq=17 | suelo_local=1,263,097 [↓20M→1263k] | 181s  ETA 12s
   ⚡ INTERPOL chunk 91/95 | maq=4,101 | suelo_local=994,031 [↓15M→994k] | 182s  ETA 8s
   ⚡ INTERPOL chunk 91/95 | maq=4,101 | suelo_local=994,031 [↓15M→994k] | 182s  ETA 8s
   📊 Z diagnostico: 1,295,041/1,412,327 puntos con dZ>1cm | dZ medio=0.416m | dZ max=10.419m
   📊 Z diagnostico: 1,295,041/1,412,327 puntos con dZ>1cm | dZ medio=0.416m | dZ max=10.419m
   ✅ Aplanados 1,412,327 puntos
   ✅ INTERPOL: 1,412,327 puntos aplanados
💾 DTM guardado en 190.8s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 190.8s