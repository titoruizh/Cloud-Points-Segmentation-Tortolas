# Cloud Point Research - Documentación

Documentación técnica del proyecto de segmentación de nubes de puntos con Deep Learning.

---

## 📚 Estructura de Documentación

### 🚀 **GPU Optimization (Fase 1)**
📁 **[phase1_gpu_acceleration/](phase1_gpu_acceleration/)**

Migración de operaciones CPU → CUDA para maximizar RTX 5090 (Blackwell).

**Quick Start**:
```bash
# Verificar entorno
python3 docs/phase1_gpu_acceleration/test_cuda_support.py

# Ver guía
./docs/phase1_gpu_acceleration/QUICKSTART.sh
```

**Archivos**:
- [README.md](phase1_gpu_acceleration/README.md) - Índice principal
- [IMPLEMENTATION.md](phase1_gpu_acceleration/IMPLEMENTATION.md) - Guía de implementación
- [ANALYSIS.md](phase1_gpu_acceleration/ANALYSIS.md) - Análisis técnico completo
- [test_cuda_support.py](phase1_gpu_acceleration/test_cuda_support.py) - Test automatizado
- [code_examples.py](phase1_gpu_acceleration/code_examples.py) - Snippets reutilizables

**Ganancia**: +20-30% throughput | 3-5x speedup normales

---

### 📊 **Reportes Técnicos**

Evolución del proyecto por versión de modelos:

- [TECHNICAL_REPORT_V1.md](TECHNICAL_REPORT_V1.md) - PointNet baseline
- [TECHNICAL_REPORT_V2.md](TECHNICAL_REPORT_V2.md) - PointNet++ inicial
- [TECHNICAL_REPORT_V3.md](TECHNICAL_REPORT_V3.md) - RandLANet exploration
- [TECHNICAL_REPORT_V4.md](TECHNICAL_REPORT_V4.md) - RGB integration
- [TECHNICAL_REPORT_V5.md](TECHNICAL_REPORT_V5.md) - No-Verticalidad
- [TECHNICAL_REPORT_V6.md](TECHNICAL_REPORT_V6.md) - High density (0.25m)

---

### 📁 **Otras Carpetas**

- **[guides/](guides/)** - Guías de usuario y desarrollo
- **[reports/](reports/)** - Reportes de experimentos y sweeps
- **[setup/](setup/)** - Configuración de entorno

---

## 🎯 Quick Links

| Tarea | Archivo |
|-------|---------|
| **Optimización GPU** | [phase1_gpu_acceleration/README.md](phase1_gpu_acceleration/README.md) |
| **Test CUDA** | [phase1_gpu_acceleration/test_cuda_support.py](phase1_gpu_acceleration/test_cuda_support.py) |
| **Última versión modelo** | [TECHNICAL_REPORT_V6.md](TECHNICAL_REPORT_V6.md) |
| **Código ejemplos GPU** | [phase1_gpu_acceleration/code_examples.py](phase1_gpu_acceleration/code_examples.py) |

---

## 🔍 Búsqueda Rápida

- **¿Cómo acelerar entrenamiento?** → [phase1_gpu_acceleration/](phase1_gpu_acceleration/)
- **¿Métricas de modelos?** → [TECHNICAL_REPORT_V6.md](TECHNICAL_REPORT_V6.md)
- **¿Setup inicial?** → [setup/](setup/)
- **¿Guías de uso?** → [guides/](guides/)

---

**Última Actualización**: 28 de Enero, 2026  
**Estado Proyecto**: ✅ Fase 1 GPU Completada
