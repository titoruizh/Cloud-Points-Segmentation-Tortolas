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
   RAM Disponible: 59.97 GB
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
   ⚡ Chunk 1/30 | core=3,071,381 pts | 1s elapsed  ETA 22s
   ⚡ Chunk 2/30 | core=3,779,447 pts | 1s elapsed  ETA 17s
   ⚡ Chunk 3/30 | core=1,412,694 pts | 2s elapsed  ETA 14s
   ⚡ Chunk 4/30 | core=3,255,564 pts | 2s elapsed  ETA 12s
   ⚡ Chunk 5/30 | core=3,988,372 pts | 2s elapsed  ETA 12s
   ⚡ Chunk 6/30 | core=1,901,925 pts | 3s elapsed  ETA 11s
   ⚡ Chunk 7/30 | core=3,161,304 pts | 3s elapsed  ETA 10s
   ⚡ Chunk 8/30 | core=3,979,412 pts | 4s elapsed  ETA 10s
   ⚡ Chunk 9/30 | core=1,757,545 pts | 4s elapsed  ETA 9s
   ⚡ Chunk 10/30 | core=2,682,968 pts | 4s elapsed  ETA 8s
   ⚡ Chunk 11/30 | core=2,955,260 pts | 5s elapsed  ETA 8s
   ⚡ Chunk 12/30 | core=1,426,306 pts | 5s elapsed  ETA 7s
   ⚡ Chunk 13/30 | core=3,265,826 pts | 5s elapsed  ETA 7s
   ⚡ Chunk 14/30 | core=3,995,523 pts | 6s elapsed  ETA 6s
   ⚡ Chunk 15/30 | core=1,611,844 pts | 6s elapsed  ETA 6s
   ⚡ Chunk 16/30 | core=3,216,534 pts | 6s elapsed  ETA 5s
   ⚡ Chunk 17/30 | core=3,991,073 pts | 7s elapsed  ETA 5s
   ⚡ Chunk 18/30 | core=1,543,374 pts | 7s elapsed  ETA 5s
   ⚡ Chunk 19/30 | core=3,202,135 pts | 7s elapsed  ETA 4s
   ⚡ Chunk 20/30 | core=3,987,316 pts | 8s elapsed  ETA 4s
   ⚡ Chunk 21/30 | core=1,624,251 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 22/30 | core=3,180,516 pts | 8s elapsed  ETA 3s
   ⚡ Chunk 23/30 | core=3,989,658 pts | 9s elapsed  ETA 3s
   ⚡ Chunk 24/30 | core=1,594,611 pts | 9s elapsed  ETA 2s
   ⚡ Chunk 25/30 | core=3,008,817 pts | 10s elapsed  ETA 2s
   ⚡ Chunk 26/30 | core=3,502,222 pts | 10s elapsed  ETA 2s
   ⚡ Chunk 27/30 | core=816,914 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 28/30 | core=612,804 pts | 10s elapsed  ETA 1s
   ⚡ Chunk 29/30 | core=21,399 pts | 11s elapsed  ETA 0s
   ✅ Normales completadas: 10.7s  (7,126,451 pts/s)
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
✅ Inferencia completada en 161.8s - Maquinaria: 285,964 puntos (0.4%)
======================================================================

   🧹 GPU liberada → VRAM reservada: 0.02 GB
   🧹 Limpieza completa
   🔄 Lanzando FIX_TECHO en proceso limpio...
🏗️ FIX_TECHO: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz...

🏗️ FIX_TECHO iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6.laz
   📊 RSS al iniciar FIX_TECHO: 15644 MB
   🚜 Maquinaria: 285,964 puntos
   🧩 Clusterizando con DBSCAN...
   🧠 Ejecutando Smart Merge (Gap Filling)...
   🔍 Smart Merge: 76,239,934 candidatos en 153 bloques
   🔍 Smart Merge: 76,239,934 candidatos
   ✨ Smart Merge: 700,421 puntos unidos
   🔢 Objetos encontrados: 1127
   ⚡ Procesando en paralelo 1127 objetos...
   ✅ Rellenados 411,666 puntos de techo
💾 Guardado: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   🔄 Lanzando INTERPOL en proceso limpio...
🚜 INTERPOL: Cargando LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz...

🚜 INTERPOL iniciando: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_Clasificado.laz
   📊 RSS al iniciar INTERPOL: 15644 MB
   📉 Maquinaria: 1,398,045 pts | Suelo: 75,138,950 pts | RAM arrays: 949 MB
   📉 Maquinaria: 1,398,045 | Suelo: 75,138,950 | Total: 76,536,995 | RAM: 949 MB
   📐 Altura: mediana_maq=1605.61m, mediana_suelo=1604.09m, gap=1.52m
   📐 Altura: mediana_maq=1605.61m, mediana_suelo=1604.09m, gap=1.52m
   📐 IDW en chunks: 95 tiles (19×5) de 250m
   📐 IDW chunks: 95 (19×5) de 250m
   ⚡ INTERPOL chunk 1/95 | maq=1,638 | suelo_local=22,065,560 | 9s  ETA 803s
   ⚡ INTERPOL chunk 1/95 | maq=1,638 | suelo_local=22,065,560 | 9s  ETA 803s
   ⚡ INTERPOL chunk 2/95 | maq=417 | suelo_local=22,065,560 | 13s  ETA 608s
   ⚡ INTERPOL chunk 2/95 | maq=417 | suelo_local=22,065,560 | 13s  ETA 608s
   ⚡ INTERPOL chunk 3/95 | maq=6,739 | suelo_local=22,065,560 | 18s  ETA 542s
   ⚡ INTERPOL chunk 3/95 | maq=6,739 | suelo_local=22,065,560 | 18s  ETA 542s
   ⚡ INTERPOL chunk 4/95 | maq=3,662 | suelo_local=22,065,560 | 24s  ETA 545s
   ⚡ INTERPOL chunk 4/95 | maq=3,662 | suelo_local=22,065,560 | 24s  ETA 545s
   ⚡ INTERPOL chunk 6/95 | maq=2,610 | suelo_local=26,319,365 | 30s  ETA 440s
   ⚡ INTERPOL chunk 6/95 | maq=2,610 | suelo_local=26,319,365 | 30s  ETA 440s
   ⚡ INTERPOL chunk 7/95 | maq=2,725 | suelo_local=26,319,365 | 35s  ETA 440s
   ⚡ INTERPOL chunk 7/95 | maq=2,725 | suelo_local=26,319,365 | 35s  ETA 440s
   ⚡ INTERPOL chunk 8/95 | maq=2,138 | suelo_local=26,319,365 | 40s  ETA 439s
   ⚡ INTERPOL chunk 8/95 | maq=2,138 | suelo_local=26,319,365 | 40s  ETA 439s
   ⚡ INTERPOL chunk 9/95 | maq=18,912 | suelo_local=26,319,365 | 46s  ETA 437s
   ⚡ INTERPOL chunk 9/95 | maq=18,912 | suelo_local=26,319,365 | 46s  ETA 437s
   ⚡ INTERPOL chunk 10/95 | maq=3,056 | suelo_local=26,316,576 | 51s  ETA 434s
   ⚡ INTERPOL chunk 10/95 | maq=3,056 | suelo_local=26,316,576 | 51s  ETA 434s
   ⚡ INTERPOL chunk 11/95 | maq=6,054 | suelo_local=29,081,698 | 59s  ETA 448s
   ⚡ INTERPOL chunk 11/95 | maq=6,054 | suelo_local=29,081,698 | 59s  ETA 448s
   ⚡ INTERPOL chunk 12/95 | maq=19,852 | suelo_local=29,081,698 | 65s  ETA 447s
   ⚡ INTERPOL chunk 12/95 | maq=19,852 | suelo_local=29,081,698 | 65s  ETA 447s
   ⚡ INTERPOL chunk 13/95 | maq=2,635 | suelo_local=29,081,698 | 70s  ETA 444s
   ⚡ INTERPOL chunk 13/95 | maq=2,635 | suelo_local=29,081,698 | 70s  ETA 444s
   ⚡ INTERPOL chunk 14/95 | maq=8,025 | suelo_local=29,081,698 | 76s  ETA 441s
   ⚡ INTERPOL chunk 14/95 | maq=8,025 | suelo_local=29,081,698 | 76s  ETA 441s
   ⚡ INTERPOL chunk 15/95 | maq=15,762 | suelo_local=29,078,909 | 82s  ETA 438s
   ⚡ INTERPOL chunk 15/95 | maq=15,762 | suelo_local=29,078,909 | 82s  ETA 438s
   ⚡ INTERPOL chunk 16/95 | maq=363 | suelo_local=33,142,337 | 91s  ETA 447s
   ⚡ INTERPOL chunk 16/95 | maq=363 | suelo_local=33,142,337 | 91s  ETA 447s
   ⚡ INTERPOL chunk 17/95 | maq=3,290 | suelo_local=33,142,337 | 97s  ETA 446s
   ⚡ INTERPOL chunk 17/95 | maq=3,290 | suelo_local=33,142,337 | 97s  ETA 446s
   ⚡ INTERPOL chunk 18/95 | maq=12,155 | suelo_local=33,142,337 | 104s  ETA 445s
   ⚡ INTERPOL chunk 18/95 | maq=12,155 | suelo_local=33,142,337 | 104s  ETA 445s
   ⚡ INTERPOL chunk 19/95 | maq=21,706 | suelo_local=33,142,337 | 111s  ETA 443s
   ⚡ INTERPOL chunk 19/95 | maq=21,706 | suelo_local=33,142,337 | 111s  ETA 443s
   ⚡ INTERPOL chunk 20/95 | maq=35,913 | suelo_local=33,139,548 | 119s  ETA 447s
   ⚡ INTERPOL chunk 20/95 | maq=35,913 | suelo_local=33,139,548 | 119s  ETA 447s
   ⚡ INTERPOL chunk 21/95 | maq=3,332 | suelo_local=37,501,117 | 127s  ETA 447s
   ⚡ INTERPOL chunk 21/95 | maq=3,332 | suelo_local=37,501,117 | 127s  ETA 447s
   ⚡ INTERPOL chunk 22/95 | maq=3,372 | suelo_local=37,501,117 | 135s  ETA 447s
   ⚡ INTERPOL chunk 22/95 | maq=3,372 | suelo_local=37,501,117 | 135s  ETA 447s
   ⚡ INTERPOL chunk 23/95 | maq=16,636 | suelo_local=37,501,117 | 143s  ETA 447s
   ⚡ INTERPOL chunk 23/95 | maq=16,636 | suelo_local=37,501,117 | 143s  ETA 447s
   ⚡ INTERPOL chunk 24/95 | maq=61,605 | suelo_local=37,501,117 | 152s  ETA 451s
   ⚡ INTERPOL chunk 24/95 | maq=61,605 | suelo_local=37,501,117 | 152s  ETA 451s
   ⚡ INTERPOL chunk 25/95 | maq=18,570 | suelo_local=37,498,328 | 160s  ETA 448s
   ⚡ INTERPOL chunk 25/95 | maq=18,570 | suelo_local=37,498,328 | 160s  ETA 448s
   ⚡ INTERPOL chunk 26/95 | maq=25,624 | suelo_local=37,877,798 | 168s  ETA 446s
   ⚡ INTERPOL chunk 26/95 | maq=25,624 | suelo_local=37,877,798 | 168s  ETA 446s
   ⚡ INTERPOL chunk 27/95 | maq=29,515 | suelo_local=37,877,798 | 176s  ETA 443s
   ⚡ INTERPOL chunk 27/95 | maq=29,515 | suelo_local=37,877,798 | 176s  ETA 443s
   ⚡ INTERPOL chunk 28/95 | maq=21,111 | suelo_local=37,877,798 | 185s  ETA 443s
   ⚡ INTERPOL chunk 28/95 | maq=21,111 | suelo_local=37,877,798 | 185s  ETA 443s
   ⚡ INTERPOL chunk 29/95 | maq=20,309 | suelo_local=37,877,798 | 193s  ETA 440s
   ⚡ INTERPOL chunk 29/95 | maq=20,309 | suelo_local=37,877,798 | 193s  ETA 440s
   ⚡ INTERPOL chunk 30/95 | maq=15,964 | suelo_local=37,877,798 | 201s  ETA 435s
   ⚡ INTERPOL chunk 30/95 | maq=15,964 | suelo_local=37,877,798 | 201s  ETA 435s
   ⚡ INTERPOL chunk 31/95 | maq=8,012 | suelo_local=37,610,538 | 209s  ETA 431s
   ⚡ INTERPOL chunk 31/95 | maq=8,012 | suelo_local=37,610,538 | 209s  ETA 431s
   ⚡ INTERPOL chunk 32/95 | maq=40,351 | suelo_local=37,610,538 | 218s  ETA 430s
   ⚡ INTERPOL chunk 32/95 | maq=40,351 | suelo_local=37,610,538 | 218s  ETA 430s
   ⚡ INTERPOL chunk 33/95 | maq=68,485 | suelo_local=37,610,538 | 226s  ETA 425s
   ⚡ INTERPOL chunk 33/95 | maq=68,485 | suelo_local=37,610,538 | 226s  ETA 425s
   ⚡ INTERPOL chunk 34/95 | maq=64,853 | suelo_local=37,610,538 | 234s  ETA 420s
   ⚡ INTERPOL chunk 34/95 | maq=64,853 | suelo_local=37,610,538 | 234s  ETA 420s
   ⚡ INTERPOL chunk 35/95 | maq=37,934 | suelo_local=37,610,538 | 242s  ETA 414s
   ⚡ INTERPOL chunk 35/95 | maq=37,934 | suelo_local=37,610,538 | 242s  ETA 414s
   ⚡ INTERPOL chunk 36/95 | maq=991 | suelo_local=37,375,513 | 251s  ETA 412s
   ⚡ INTERPOL chunk 36/95 | maq=991 | suelo_local=37,375,513 | 251s  ETA 412s
   ⚡ INTERPOL chunk 37/95 | maq=27,993 | suelo_local=37,375,513 | 259s  ETA 406s
   ⚡ INTERPOL chunk 37/95 | maq=27,993 | suelo_local=37,375,513 | 259s  ETA 406s
   ⚡ INTERPOL chunk 38/95 | maq=25,530 | suelo_local=37,375,513 | 267s  ETA 400s
   ⚡ INTERPOL chunk 38/95 | maq=25,530 | suelo_local=37,375,513 | 267s  ETA 400s
   ⚡ INTERPOL chunk 39/95 | maq=9,484 | suelo_local=37,375,513 | 276s  ETA 396s
   ⚡ INTERPOL chunk 39/95 | maq=9,484 | suelo_local=37,375,513 | 276s  ETA 396s
   ⚡ INTERPOL chunk 40/95 | maq=9,021 | suelo_local=37,375,513 | 284s  ETA 390s
   ⚡ INTERPOL chunk 40/95 | maq=9,021 | suelo_local=37,375,513 | 284s  ETA 390s
   ⚡ INTERPOL chunk 41/95 | maq=95 | suelo_local=37,157,034 | 292s  ETA 384s
   ⚡ INTERPOL chunk 41/95 | maq=95 | suelo_local=37,157,034 | 292s  ETA 384s
   ⚡ INTERPOL chunk 42/95 | maq=4,590 | suelo_local=37,157,034 | 299s  ETA 378s
   ⚡ INTERPOL chunk 42/95 | maq=4,590 | suelo_local=37,157,034 | 299s  ETA 378s
   ⚡ INTERPOL chunk 43/95 | maq=563 | suelo_local=37,157,034 | 309s  ETA 373s
   ⚡ INTERPOL chunk 43/95 | maq=563 | suelo_local=37,157,034 | 309s  ETA 373s
   ⚡ INTERPOL chunk 44/95 | maq=7,774 | suelo_local=37,157,034 | 316s  ETA 367s
   ⚡ INTERPOL chunk 44/95 | maq=7,774 | suelo_local=37,157,034 | 316s  ETA 367s
   ⚡ INTERPOL chunk 45/95 | maq=17,389 | suelo_local=37,157,034 | 324s  ETA 360s
   ⚡ INTERPOL chunk 45/95 | maq=17,389 | suelo_local=37,157,034 | 324s  ETA 360s
   ⚡ INTERPOL chunk 46/95 | maq=181 | suelo_local=37,103,819 | 332s  ETA 354s
   ⚡ INTERPOL chunk 46/95 | maq=181 | suelo_local=37,103,819 | 332s  ETA 354s
   ⚡ INTERPOL chunk 47/95 | maq=6,352 | suelo_local=37,103,819 | 342s  ETA 349s
   ⚡ INTERPOL chunk 47/95 | maq=6,352 | suelo_local=37,103,819 | 342s  ETA 349s
   ⚡ INTERPOL chunk 49/95 | maq=3,458 | suelo_local=37,103,819 | 349s  ETA 328s
   ⚡ INTERPOL chunk 49/95 | maq=3,458 | suelo_local=37,103,819 | 349s  ETA 328s
   ⚡ INTERPOL chunk 50/95 | maq=48,522 | suelo_local=37,103,819 | 357s  ETA 321s
   ⚡ INTERPOL chunk 50/95 | maq=48,522 | suelo_local=37,103,819 | 357s  ETA 321s
   ⚡ INTERPOL chunk 51/95 | maq=1,126 | suelo_local=37,212,689 | 365s  ETA 315s
   ⚡ INTERPOL chunk 51/95 | maq=1,126 | suelo_local=37,212,689 | 365s  ETA 315s
   ⚡ INTERPOL chunk 52/95 | maq=2,870 | suelo_local=37,212,689 | 374s  ETA 310s
   ⚡ INTERPOL chunk 52/95 | maq=2,870 | suelo_local=37,212,689 | 374s  ETA 310s
   ⚡ INTERPOL chunk 54/95 | maq=5,908 | suelo_local=37,212,689 | 382s  ETA 290s
   ⚡ INTERPOL chunk 54/95 | maq=5,908 | suelo_local=37,212,689 | 382s  ETA 290s
   ⚡ INTERPOL chunk 55/95 | maq=139 | suelo_local=37,212,689 | 390s  ETA 283s
   ⚡ INTERPOL chunk 55/95 | maq=139 | suelo_local=37,212,689 | 390s  ETA 283s
   ⚡ INTERPOL chunk 56/95 | maq=5,606 | suelo_local=38,737,967 | 398s  ETA 277s
   ⚡ INTERPOL chunk 56/95 | maq=5,606 | suelo_local=38,737,967 | 398s  ETA 277s
   ⚡ INTERPOL chunk 57/95 | maq=37,848 | suelo_local=38,737,967 | 408s  ETA 272s
   ⚡ INTERPOL chunk 57/95 | maq=37,848 | suelo_local=38,737,967 | 408s  ETA 272s
   ⚡ INTERPOL chunk 58/95 | maq=16 | suelo_local=38,737,967 | 416s  ETA 265s
   ⚡ INTERPOL chunk 58/95 | maq=16 | suelo_local=38,737,967 | 416s  ETA 265s
   ⚡ INTERPOL chunk 59/95 | maq=10,586 | suelo_local=38,737,967 | 424s  ETA 259s
   ⚡ INTERPOL chunk 59/95 | maq=10,586 | suelo_local=38,737,967 | 424s  ETA 259s
   ⚡ INTERPOL chunk 61/95 | maq=38,830 | suelo_local=38,751,845 | 432s  ETA 241s
   ⚡ INTERPOL chunk 61/95 | maq=38,830 | suelo_local=38,751,845 | 432s  ETA 241s
   ⚡ INTERPOL chunk 62/95 | maq=37,336 | suelo_local=38,751,845 | 441s  ETA 235s
   ⚡ INTERPOL chunk 62/95 | maq=37,336 | suelo_local=38,751,845 | 441s  ETA 235s
   ⚡ INTERPOL chunk 63/95 | maq=32,366 | suelo_local=38,751,845 | 449s  ETA 228s
   ⚡ INTERPOL chunk 63/95 | maq=32,366 | suelo_local=38,751,845 | 449s  ETA 228s
   ⚡ INTERPOL chunk 64/95 | maq=32,647 | suelo_local=38,751,845 | 459s  ETA 223s
   ⚡ INTERPOL chunk 64/95 | maq=32,647 | suelo_local=38,751,845 | 459s  ETA 223s
   ⚡ INTERPOL chunk 65/95 | maq=1,194 | suelo_local=38,751,845 | 468s  ETA 216s
   ⚡ INTERPOL chunk 65/95 | maq=1,194 | suelo_local=38,751,845 | 468s  ETA 216s
   ⚡ INTERPOL chunk 66/95 | maq=14,023 | suelo_local=37,140,215 | 476s  ETA 209s
   ⚡ INTERPOL chunk 66/95 | maq=14,023 | suelo_local=37,140,215 | 476s  ETA 209s
   ⚡ INTERPOL chunk 67/95 | maq=18,210 | suelo_local=37,140,215 | 484s  ETA 202s
   ⚡ INTERPOL chunk 67/95 | maq=18,210 | suelo_local=37,140,215 | 484s  ETA 202s
   ⚡ INTERPOL chunk 68/95 | maq=54,657 | suelo_local=37,140,215 | 492s  ETA 195s
   ⚡ INTERPOL chunk 68/95 | maq=54,657 | suelo_local=37,140,215 | 492s  ETA 195s
   ⚡ INTERPOL chunk 69/95 | maq=27,673 | suelo_local=37,140,215 | 502s  ETA 189s
   ⚡ INTERPOL chunk 69/95 | maq=27,673 | suelo_local=37,140,215 | 502s  ETA 189s
   ⚡ INTERPOL chunk 70/95 | maq=3,069 | suelo_local=37,140,215 | 509s  ETA 182s
   ⚡ INTERPOL chunk 70/95 | maq=3,069 | suelo_local=37,140,215 | 509s  ETA 182s
   ⚡ INTERPOL chunk 71/95 | maq=7,516 | suelo_local=33,207,910 | 516s  ETA 175s
   ⚡ INTERPOL chunk 71/95 | maq=7,516 | suelo_local=33,207,910 | 516s  ETA 175s
   ⚡ INTERPOL chunk 72/95 | maq=9,444 | suelo_local=33,207,910 | 523s  ETA 167s
   ⚡ INTERPOL chunk 72/95 | maq=9,444 | suelo_local=33,207,910 | 523s  ETA 167s
   ⚡ INTERPOL chunk 73/95 | maq=19,313 | suelo_local=33,207,910 | 532s  ETA 160s
   ⚡ INTERPOL chunk 73/95 | maq=19,313 | suelo_local=33,207,910 | 532s  ETA 160s
   ⚡ INTERPOL chunk 74/95 | maq=3,320 | suelo_local=33,207,910 | 539s  ETA 153s
   ⚡ INTERPOL chunk 74/95 | maq=3,320 | suelo_local=33,207,910 | 539s  ETA 153s
   ⚡ INTERPOL chunk 75/95 | maq=16,308 | suelo_local=33,207,910 | 545s  ETA 145s
   ⚡ INTERPOL chunk 75/95 | maq=16,308 | suelo_local=33,207,910 | 545s  ETA 145s
   ⚡ INTERPOL chunk 76/95 | maq=1,254 | suelo_local=28,837,478 | 551s  ETA 138s
   ⚡ INTERPOL chunk 76/95 | maq=1,254 | suelo_local=28,837,478 | 551s  ETA 138s
   ⚡ INTERPOL chunk 77/95 | maq=6,061 | suelo_local=28,837,478 | 557s  ETA 130s
   ⚡ INTERPOL chunk 77/95 | maq=6,061 | suelo_local=28,837,478 | 557s  ETA 130s
   ⚡ INTERPOL chunk 78/95 | maq=7,110 | suelo_local=28,837,478 | 565s  ETA 123s
   ⚡ INTERPOL chunk 78/95 | maq=7,110 | suelo_local=28,837,478 | 565s  ETA 123s
   ⚡ INTERPOL chunk 79/95 | maq=23,993 | suelo_local=28,837,478 | 571s  ETA 116s
   ⚡ INTERPOL chunk 79/95 | maq=23,993 | suelo_local=28,837,478 | 571s  ETA 116s
   ⚡ INTERPOL chunk 80/95 | maq=5,499 | suelo_local=28,837,478 | 577s  ETA 108s
   ⚡ INTERPOL chunk 80/95 | maq=5,499 | suelo_local=28,837,478 | 577s  ETA 108s
   ⚡ INTERPOL chunk 81/95 | maq=2,431 | suelo_local=24,522,725 | 582s  ETA 101s
   ⚡ INTERPOL chunk 81/95 | maq=2,431 | suelo_local=24,522,725 | 582s  ETA 101s
   ⚡ INTERPOL chunk 82/95 | maq=17,248 | suelo_local=24,522,725 | 587s  ETA 93s
   ⚡ INTERPOL chunk 82/95 | maq=17,248 | suelo_local=24,522,725 | 587s  ETA 93s
   ⚡ INTERPOL chunk 83/95 | maq=45,199 | suelo_local=24,522,725 | 593s  ETA 86s
   ⚡ INTERPOL chunk 83/95 | maq=45,199 | suelo_local=24,522,725 | 593s  ETA 86s
   ⚡ INTERPOL chunk 84/95 | maq=34,255 | suelo_local=24,522,725 | 599s  ETA 78s
   ⚡ INTERPOL chunk 84/95 | maq=34,255 | suelo_local=24,522,725 | 599s  ETA 78s
   ⚡ INTERPOL chunk 85/95 | maq=19,954 | suelo_local=24,522,725 | 604s  ETA 71s
   ⚡ INTERPOL chunk 85/95 | maq=19,954 | suelo_local=24,522,725 | 604s  ETA 71s
   ⚡ INTERPOL chunk 86/95 | maq=25,226 | suelo_local=20,282,770 | 609s  ETA 64s
   ⚡ INTERPOL chunk 86/95 | maq=25,226 | suelo_local=20,282,770 | 609s  ETA 64s
   ⚡ INTERPOL chunk 87/95 | maq=52,483 | suelo_local=20,282,770 | 613s  ETA 56s
   ⚡ INTERPOL chunk 87/95 | maq=52,483 | suelo_local=20,282,770 | 613s  ETA 56s
   ⚡ INTERPOL chunk 88/95 | maq=8,819 | suelo_local=20,282,770 | 617s  ETA 49s
   ⚡ INTERPOL chunk 88/95 | maq=8,819 | suelo_local=20,282,770 | 617s  ETA 49s
   ⚡ INTERPOL chunk 89/95 | maq=77 | suelo_local=20,282,770 | 621s  ETA 42s
   ⚡ INTERPOL chunk 89/95 | maq=77 | suelo_local=20,282,770 | 621s  ETA 42s
   ⚡ INTERPOL chunk 91/95 | maq=3,163 | suelo_local=15,969,571 | 625s  ETA 27s
   ⚡ INTERPOL chunk 91/95 | maq=3,163 | suelo_local=15,969,571 | 625s  ETA 27s
   📊 Z diagnostico: 1,260,995/1,398,045 puntos con dZ>1cm | dZ medio=0.397m | dZ max=10.574m
   📊 Z diagnostico: 1,260,995/1,398,045 puntos con dZ>1cm | dZ medio=0.397m | dZ max=10.574m
   ✅ Aplanados 1,398,045 puntos
   ✅ INTERPOL: 1,398,045 puntos aplanados
💾 DTM guardado en 635.2s: LINK_260226_LOG176_NDP_PTL_edit_RGB_0.25m_PointnetV6_DTM.laz
💾 DTM guardado: 635.2s