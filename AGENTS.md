# AGENTS.md — KNN Paralelo: OpenMP vs CUDA

> Este archivo es el contexto canónico del proyecto. Todo agente que trabaje
> en este repositorio debe leerlo completo antes de escribir una sola línea.

---

## Qué es este proyecto

Implementación y comparación del algoritmo **K-Nearest Neighbors (KNN)** bajo
tres paradigmas:

| Implementación | Archivo | Tecnología |
|---------------|---------|------------|
| Secuencial (referencia) | `src/knn_seq.c` | C puro, GCC |
| Paralela CPU | `src/knn_omp.c` | C + OpenMP |
| Paralela GPU | `src/knn_cuda.cu` + `src/kernels.cuh` | CUDA C++ |

El objetivo es medir empíricamente cuál arquitectura gana según el tamaño del
dataset (N) y el número de características (D).

---

## Estructura del Repositorio

```
knn-parallel/
├── src/
│   ├── knn_seq.c          # Implementación secuencial de referencia
│   ├── knn_omp.c          # Implementación OpenMP (CPU multi-hilo)
│   ├── knn_cuda.cu        # Host code CUDA + main()
│   └── kernels.cuh        # Kernels GPU: compute_distances, find_k_nearest
├── scripts/
│   ├── benchmark.sh       # Automatiza todos los experimentos
│   └── validate.sh        # Verifica correctitud entre implementaciones
├── analysis/
│   ├── results.ipynb      # Notebook de visualización y análisis
│   └── figures/           # PNGs generados por el notebook (fig_1..fig_5)
├── results/
│   ├── benchmark_results.csv   # Salida del benchmark (generado, no editar)
│   └── errors.log              # Log de errores del benchmark (generado)
├── data_gen.py            # Genera datasets sintéticos .npy y .csv
├── Makefile               # Targets: all, seq, omp, cuda, benchmark, validate, clean
├── README.md              # Instrucciones de instalación y uso
└── AGENTS.md              # Este archivo
```

---

## Contratos de Interfaz — NO los rompas

Estas firmas son fijas. Todas las implementaciones deben respetarlas.

### Formato de datos en disco

```c
// Archivo binario .npy (formato simplificado propio, NO numpy real):
// Bytes 0-3:   int32  N  (número de puntos)
// Bytes 4-7:   int32  D  (número de características)
// Bytes 8+:    float32[N*D]  row-major
```

```python
# data_gen.py escribe exactamente este formato.
# knn_seq.c, knn_omp.c y knn_cuda.cu deben leer exactamente este formato.
```

### Función principal KNN (igual en los tres .c/.cu)

```c
void knn_predict(
    const float* train,       // [N x D] row-major, solo lectura
    const float* labels,      // [N]     clases enteras como float
    int N,                    // número de puntos de entrenamiento
    int D,                    // número de características
    const float* query,       // [Q x D] row-major, puntos a clasificar
    int Q,                    // número de puntos de consulta
    int k,                    // número de vecinos
    float* predictions        // [Q] salida: clase predicha por mayoría
);
```

### CLI unificada (los tres binarios deben aceptar estos argumentos)

```bash
./bin/knn_<impl> \
  --train  <path.npy>   \   # dataset de entrenamiento
  --query  <path.npy>   \   # puntos de consulta
  --k      <int>        \   # número de vecinos (default: 5)
  --output <path.txt>   \   # predicciones, una por línea
  --threads <int>           # solo knn_omp: número de hilos OMP
  --device  <int>           # solo knn_cuda: índice de GPU (default: 0)
```

### Formato de salida (predictions)

```
# Archivo de texto plano, una predicción por línea, sin header:
0
1
1
0
2
...
```

### Formato CSV de resultados

```
n,d,k,impl,threads,run,time_ms,transfer_ms,compute_ms,speedup_vs_seq
```

- `impl`: uno de `seq`, `omp`, `cuda`
- `threads`: número de hilos OMP; para seq y cuda usar `1`
- `transfer_ms`: solo CUDA (H→D + D→H); para seq/omp usar `0`
- `compute_ms`: tiempo sin transferencias; para seq/omp igual a `time_ms`
- `speedup_vs_seq`: calculado por benchmark.sh al final de cada config

---

## Decisiones de Diseño Tomadas

Estas decisiones ya están cerradas. No las reabras, no las cambies, no preguntes.

**1. Distancia euclidea al cuadrado** — Se omite la raíz cuadrada en todas las
implementaciones. No afecta el ranking de vecinos y ahorra cómputo.

**2. Max-heap de tamaño k para selección de vecinos** — En CPU (seq y omp) se
mantiene un heap de k elementos durante el scan de N puntos. Complejidad O(N log k).

**3. schedule(dynamic, 32) en OpenMP** — El chunk size 32 balancea overhead de
scheduling vs granularidad. No cambiar sin benchmark que justifique.

**4. Tiling 16×16 en kernel CUDA** — Tiles de 16×16 floats en shared memory
(1 KB por array). Balance entre ocupación del SM y presión de registros.

**5. Thrust para sort en GPU** — Se usa `thrust::sort_by_key` para ordenar
distancias. No reimplementar sort desde cero salvo que sea requerimiento explícito.

**6. Semilla aleatoria fija: 42** — Todos los datasets generados usan `random_seed=42`
para reproducibilidad. Nunca usar semillas aleatorias en tests o benchmarks.

**7. 5 repeticiones por configuración** — El benchmark ejecuta 5 runs y reporta
promedio ± desviación estándar. No cambiar sin actualizar el análisis estadístico.

---

## Restricciones Técnicas del Entorno

```
Compilador C:    GCC ≥ 11     Flags: -O2 -Wall -fopenmp
Compilador CUDA: NVCC ≥ 11.8  Flags: -O2 -arch=sm_70
Python:          ≥ 3.10       Deps: numpy, pandas, matplotlib, seaborn, scikit-learn
GPU mínima:      Compute Capability 6.0 (Pascal)
SO:              Ubuntu 22.04 LTS
```

**VRAM budget:** Asumir ≤ 8 GB. Si N×D×4 bytes > 4 GB, el código debe usar
procesamiento por lotes (batched). Implementar esto si N > 200K con D > 500.

---

## Convenciones de Código

- **Idioma del código:** inglés (nombres de variables, comentarios en código)
- **Idioma de documentación:** español (README, informe, comentarios de diseño)
- **Indentación:** 4 espacios en C y Python; 2 espacios en bash
- **Nombres de variables:** snake_case en C y Python
- **Sin warnings:** el código debe compilar sin warnings con `-Wall`
- **Sin memoria sin liberar:** usar valgrind en CPU; cuda-memcheck en GPU
- **Timing:** siempre con `clock_gettime(CLOCK_MONOTONIC)` en CPU,
  `cudaEvent_t` en GPU. Nunca con `time()` o `gettimeofday()`

---

## Lo que Nunca Debes Hacer

- **No uses numpy .npy real** — el formato binario del proyecto es propio y más simple
- **No cambies las firmas de función** — otros módulos dependen de ellas
- **No agregues dependencias externas** en C — solo stdlib y OpenMP/CUDA
- **No imprimas dentro de knn_predict()** — solo el main() imprime
- **No hagas cudaMalloc dentro de kernels** — toda memoria se aloja en el host code
- **No uses variables globales** en ninguna implementación
- **No hardcodees rutas** — todo via argumentos CLI o variables de entorno
- **No modifiques benchmark_results.csv ni errors.log** — son generados automáticamente

---

## Cómo Verificar que tu Código es Correcto

Antes de marcar cualquier tarea como completa, ejecuta:

```bash
# 1. Compila sin errores ni warnings
make all 2>&1 | grep -E "error:|warning:" | wc -l   # debe ser 0

# 2. Valida correctitud (predicciones idénticas entre implementaciones)
make validate   # debe imprimir PASS en todas las comparaciones

# 3. Prueba de humo con dataset pequeño
python data_gen.py --n 500 --d 10 --output /tmp/smoke
./bin/knn_seq --train /tmp/smoke_train.npy --query /tmp/smoke_query.npy \
              --k 3 --output /tmp/pred_seq.txt
wc -l /tmp/pred_seq.txt   # debe ser igual a Q (número de queries)
```

Si cualquier paso falla, no entregues el código.

---

## Variables de Entorno Relevantes

```bash
OMP_NUM_THREADS=8          # hilos por defecto para knn_omp
CUDA_VISIBLE_DEVICES=0     # GPU a usar
KNN_DATA_DIR=./data        # directorio de datasets generados
KNN_RESULTS_DIR=./results  # directorio de salida del benchmark
```

---

## Glosario

| Término | Significado en este proyecto |
|---------|------------------------------|
| N | Número de puntos de entrenamiento |
| D | Número de características (dimensiones) |
| Q | Número de puntos de consulta (query set) |
| k | Número de vecinos más cercanos |
| T | Número de hilos OpenMP |
| SM | Streaming Multiprocessor (unidad de cómputo en GPU NVIDIA) |
| smem | Shared memory dentro de un bloque CUDA |
| H→D | Transferencia host (CPU RAM) a device (GPU VRAM) |
| D→H | Transferencia device a host |
| Speedup | T_seq / T_paralelo para la misma configuración (N, D, k) |
| Eficiencia | Speedup / número_de_procesadores |
