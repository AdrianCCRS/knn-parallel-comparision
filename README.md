# KNN Paralelo: OpenMP vs CUDA

Comparación empírica del algoritmo K-Nearest Neighbors bajo tres paradigmas:
secuencial (C), paralelo en CPU (OpenMP) y paralelo en GPU (CUDA).
Se evalúa speedup, escalabilidad y eficiencia sobre datasets sintéticos
con dimensionalidad y tamaño controlados.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Requisitos

| Componente | Versión      |
|-----------|-------------|
| GCC       | ≥ 11        |
| NVCC      | ≥ 11.8      |
| Python    | ≥ 3.9       |
| GPU       | Compute Capability ≥ 6.0 |
| SO        | Ubuntu 22.04 LTS |

Paquetes Python: `numpy pandas matplotlib seaborn scikit-learn`

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn
make all
```

## Uso rápido

```bash
# Generar datasets sintéticos
python data_gen.py --n 1000 --d 10 --output data/demo

# Compilar (seq + omp + cuda)
make all

# Validar correctitud entre implementaciones
make validate

# Benchmark completo (~4-6 h)
bash scripts/benchmark.sh
```

## Estructura del proyecto

```
├── src/
│   ├── knn_seq.c          # Implementación secuencial (C)
│   ├── knn_omp.c          # Implementación OpenMP (C + OpenMP)
│   ├── knn_cuda.cu        # Host code CUDA + main()
│   └── kernels.cuh        # Kernels GPU: compute_distances, find_k_nearest
├── scripts/
│   ├── benchmark.sh       # Benchmark automatizado (barrido N×D×K)
│   └── validate.sh        # Verifica predicciones idénticas entre implementaciones
├── analysis/
│   └── figures/           # Figuras del benchmark (generadas)
├── results/
│   ├── benchmark_results.csv   # Datos crudos del benchmark (generado)
│   └── errors.log              # Log de errores (generado)
├── data_gen.py            # Generador de datasets sintéticos con sklearn
├── Makefile               # Targets: all, seq, omp, cuda, benchmark, validate, clean
└── AGENTS.md              # Convenciones, contratos de interfaz y restricciones
```

## Decisiones de diseño

- **Distancia euclídea al cuadrado** — se omite la raíz cuadrada; no afecta el ranking de vecinos.
- **Max-heap de tamaño K** para selección de vecinos en CPU (O(N log K)).
- **`schedule(dynamic, 32)` en OpenMP** — balancea overhead de scheduling vs granularidad.
- **Tiling 16×16 en kernel CUDA** — 1 KB por array en shared memory, buena ocupación de SMs.
- **`thrust::sort_by_key` para top-K en GPU** — ordenamiento completo por simplicidad.
- **5 repeticiones por configuración** — promedio ± desviación estándar para robustez estadística.
- **Semilla fija 42** — reproducibilidad total.

## Formato de datos

Los binarios leen un formato `.npy` simplificado (no NumPy real):

```
Bytes 0–3:   int32 N   (número de puntos)
Bytes 4–7:   int32 D   (número de características)
Bytes 8+:    float32[N*D]  row-major
```

La salida de predicciones es texto plano, una clase por línea:
```
0
1
2
0
...
```

## CLI unificada

Los tres binarios comparten la misma interfaz:

```bash
./bin/knn_seq  --train <path.npy> --query <path.npy> --k <int> --output <path.txt>
./bin/knn_omp  --train <path.npy> --query <path.npy> --k <int> --output <path.txt> --threads <int>
./bin/knn_cuda --train <path.npy> --query <path.npy> --k <int> --output <path.txt> --device <int>
```

## Resultados principales

Benchmark sobre N ∈ {1000, 5000, 10000}, D ∈ {2, 10, 50, 100, 500, 1000}, K ∈ {1, 3, 5, 10}.

| Implementación | Speedup medio | Speedup máximo | Eficiencia (8 hilos) |
|---------------|---------------|----------------|----------------------|
| Secuencial    | 1.00x         | 1.00x          | —                    |
| OpenMP (8 h)  | 6.48x         | 9.08x          | 80.8%                |
| CUDA          | 64.08x        | 197.49x        | —                    |

**Punto de cruce:** CUDA supera al mejor OpenMP a partir de N×D ≈ 50,000 (ej. N=1000, D=50 o N=5000, D=10). Por debajo de ese umbral, el overhead de transferencia PCIe y lanzamiento de kernels hace preferible OpenMP.

Para D≥100, CUDA es consistentemente 10-20× más rápido que OpenMP independientemente de N.

Ver `analysis/technical_analysis.md` para el análisis completo.

## Cómo reproducir

1. Clonar el repositorio y crear el virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Compilar las tres implementaciones:
   ```bash
   make all
   ```

3. Validar correctitud:
   ```bash
   make validate
   # Esperado: PASS en las 2 comparaciones
   ```

4. Ejecutar benchmark completo (cluster con GPU):
   ```bash
   bash scripts/benchmark.sh
   ```
   
5. Los resultados se guardarán en `results/benchmark_results.csv`.


## Licencia

MIT License. Ver archivo `LICENSE`.
