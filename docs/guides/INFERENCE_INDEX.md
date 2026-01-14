# 📚 Índice de Documentación - Sistema de Inferencia

## 🚀 Inicio Rápido
1. **Verificar Setup:** `./check_inference_setup.sh`
2. **Leer:** [INFERENCE_README.md](INFERENCE_README.md) (2 minutos)
3. **Ejecutar:** `./run_inference_mini.sh` o `./run_inference_pointnet2.sh`

---

## 📖 Documentación Completa

### Para Usuarios
| Documento | Descripción | Cuándo Leer |
|-----------|-------------|-------------|
| [INFERENCE_README.md](INFERENCE_README.md) | Guía ultra-rápida con comandos | 🟢 Empezar aquí |
| [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) | Guía completa con todos los parámetros | 🟡 Para entender opciones |
| [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) | Comparativa detallada de modelos | 🟡 Elegir mejor arquitectura |

### Scripts Ejecutables
| Script | Propósito | Uso |
|--------|-----------|-----|
| `check_inference_setup.sh` | Verificar instalación | `./check_inference_setup.sh` |
| `run_inference_mini.sh` | Ejecutar MiniPointNet | `./run_inference_mini.sh` |
| `run_inference_pointnet2.sh` | Ejecutar PointNet++ | `./run_inference_pointnet2.sh` |
| `run_inference.sh` | Script general antiguo | `./run_inference.sh` |

### Código Principal
- **inference.py** - Script principal con toda la lógica

---

## 🎯 Casos de Uso por Documento

### "Quiero clasificar rápido, sin complicaciones"
→ [INFERENCE_README.md](INFERENCE_README.md)
```bash
./run_inference_mini.sh
```

### "Necesito máxima precisión, tengo tiempo"
→ [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) (Sección PointNet++)
```bash
./run_inference_pointnet2.sh
```

### "No sé qué modelo usar para mi caso"
→ [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)

### "Quiero entender todos los parámetros"
→ [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

### "¿Qué es el sistema de votación?"
→ [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) (Sección "Sistema de Votación")

### "Tengo errores o resultados raros"
→ [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) (Sección "Troubleshooting")

---

## 📊 Estructura de Archivos

```
Cloud-Point-Research/
│
├── 📄 INFERENCE_README.md              # Guía rápida (empezar aquí)
├── 📄 INFERENCE_GUIDE.md               # Guía completa
├── 📄 ARCHITECTURE_COMPARISON.md       # Comparativa de modelos
├── 📄 INFERENCE_INDEX.md               # Este archivo
│
├── 🐍 inference.py                     # Script principal
│
├── 📜 check_inference_setup.sh         # Verificador
├── 📜 run_inference_mini.sh            # Atajo MiniPointNet
├── 📜 run_inference_pointnet2.sh       # Atajo PointNet++
└── 📜 run_inference.sh                 # Script general
```

---

## 🎓 Flujo de Aprendizaje Recomendado

### Nivel 1: Principiante (5 minutos)
1. Leer [INFERENCE_README.md](INFERENCE_README.md)
2. Ejecutar `./check_inference_setup.sh`
3. Probar `./run_inference_mini.sh`

### Nivel 2: Usuario (15 minutos)
1. Leer [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) (ejemplos por arquitectura)
2. Comparar resultados MiniPointNet vs PointNet++
3. Ajustar parámetros según necesidad

### Nivel 3: Avanzado (30 minutos)
1. Leer [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)
2. Experimentar con diferentes configuraciones
3. Optimizar para tu caso específico
4. Entender sistema de votación

---

## 🔧 Comandos de Ayuda

```bash
# Ver ayuda del script principal
python3 inference.py --help

# Verificar instalación
./check_inference_setup.sh

# Ver documentación rápida
cat INFERENCE_README.md

# Ver guía completa
cat INFERENCE_GUIDE.md

# Ver comparativa
cat ARCHITECTURE_COMPARISON.md
```

---

## 📞 Referencia Rápida de Parámetros

| Parámetro | MiniPointNet | PointNet++ | RandLANet |
|-----------|--------------|------------|-----------|
| `--architecture` | MiniPointNet | PointNet2 | RandLANet |
| `--block-size` | 10.0 | 10.0-15.0 | 20.0 |
| `--stride` | 10.0 (rápido)<br>5.0 (preciso) | 5.0-7.5<br>(siempre < block) | 20.0 |
| Solapamiento | Opcional | **Recomendado** | No necesario |
| Votación | Auto si stride < block | ✅ Siempre | Auto si stride < block |

---

## 🎯 Outputs Organizados

Las salidas se guardan automáticamente en:
```
data/test_results/
├── MiniPointNet/
├── PointNet2/
└── RandLANet/
```

Cada carpeta contiene los archivos `.las` clasificados de esa arquitectura.

---

## 📝 Historial de Versiones

- **v2.0** (Actual) - Sistema de votación + organización por arquitectura
- **v1.0** - Primera versión con MiniPointNet básico

---

## 🤝 Contribuciones

Para agregar soporte de nuevas arquitecturas, editar:
1. `inference.py` - Agregar en función `load_model()`
2. `INFERENCE_GUIDE.md` - Agregar sección con ejemplos
3. `ARCHITECTURE_COMPARISON.md` - Agregar comparativa
4. Crear script `run_inference_NUEVA.sh`
