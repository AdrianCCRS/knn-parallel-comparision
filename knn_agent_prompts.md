# KNN Paralelo: Instrucciones de Orquestación por Agente

> **Uso:** Cada bloque `PROMPT PARA AGENTE` es el texto exacto que envías al modelo indicado.
> Copia, pega y ejecuta. Los agentes no conocen contexto previo salvo lo que incluyas.

---

## FASE 1 — Configuración y Base (Semana 1-2)

### Tarea 1.1 — Generador de Datasets

**Modelo:** `gpt-4o-mini` / `claude-haiku` (tarea rutinaria de codificación)

```
PROMPT PARA AGENTE 1.1:

Eres un programador experto en Python científico. Escribe un script completo
llamado `data_gen.py` que genere datasets sintéticos para benchmarking de KNN.

REQUISITOS EXACTOS:
- Función: generate_dataset(n_samples, n_features, n_classes=2, random_seed=42)
- Usa sklearn.datasets.make_classification
- Guarda en formato CSV (para debug) y binario NumPy .npy (para velocidad)
- Argparse CLI: --n [int] --d [int] --k [int] --seed [int] --output [path]
- Imprime: shape, tiempo de generación, path de salida
- Validación: n >= 100, d >= 2, d < n

Escribe SOLO el código Python, sin explicaciones. El archivo debe ser ejecutable
como script standalone.
```

---

### Tarea 1.2 — Implementación Secuencial KNN en C

**Modelo:** `gpt-4o-mini` / `claude-haiku`

```
PROMPT PARA AGENTE 1.2:

Escribe el archivo `src/knn_seq.c` con implementación secuencial de KNN en C puro.

ESPECIFICACIONES:
- Función principal: knn_predict(float* train, float* labels, int N, int D,
                                  float* query, int Q, int k, float* predictions)
  - train: matriz [N x D] row-major
  - labels: vector [N] con clases (float)
  - query: matriz [Q x D] de puntos a clasificar
  - predictions: salida [Q]
- Distancia euclidea al cuadrado (sin sqrt, más rápido)
- Para k vecinos: usa un heap de tamaño k (max-heap) para eficiencia
- Timing: mide con clock_gettime(CLOCK_MONOTONIC)
- main(): carga datos desde archivo .npy binario (formato simple: header int N,D luego floats)
- Argparse simple: argc/argv para --train --query --k --output

Escribe SOLO el código C. Incluye todos los #include necesarios.
```

---

### Tarea 1.3 — Makefile del Proyecto

**Modelo:** `gpt-4o-mini` / `claude-haiku`

```
PROMPT PARA AGENTE 1.3:

Escribe un Makefile completo para un proyecto con los siguientes targets:

ARCHIVOS FUENTE:
- src/knn_seq.c       → compila con gcc -O2
- src/knn_omp.c       → compila con gcc -O2 -fopenmp
- src/knn_cuda.cu     → compila con nvcc -O2 -arch=sm_70

TARGETS REQUERIDOS:
- make all            → compila los tres binarios en bin/
- make seq            → solo secuencial
- make omp            → solo OpenMP
- make cuda           → solo CUDA
- make benchmark      → llama scripts/benchmark.sh
- make clean          → limpia bin/ y results/
- make validate       → ejecuta scripts/validate.sh

VARIABLES configurables en el Makefile:
- CC, NVCC, CFLAGS, CUDA_ARCH, OMP_THREADS

Escribe SOLO el Makefile, sin explicaciones.
```

---

## FASE 2 — Implementación OpenMP (Semana 2-3)

### Tarea 2.1 — KNN con OpenMP

**Modelo:** `claude-sonnet` / `gpt-4o` (requiere conocimiento de patrones paralelos)

```
PROMPT PARA AGENTE 2.1:

Eres un experto en programación paralela con OpenMP. Escribe `src/knn_omp.c`,
versión paralela de KNN basada en esta firma de referencia del secuencial:

  knn_predict(float* train, float* labels, int N, int D,
              float* query, int Q, int k, float* predictions)

ESTRATEGIA DE PARALELIZACIÓN:
1. Paralelizar el bucle externo sobre Q (puntos de consulta) con:
   #pragma omp parallel for schedule(dynamic, 32) num_threads(T)
2. Cada hilo mantiene su propio buffer privado de distancias (N floats)
3. Memoria compartida de solo lectura: train[], labels[] (no hay race condition)
4. Variable T configurable via OMP_NUM_THREADS o argumento CLI --threads

TAMBIÉN implementa:
- Función para medir tiempo con omp_get_wtime()
- Verificación de resultados vs archivo de referencia (tolerancia 0.0)
- Impresión de: num_threads, tiempo_total, tiempo_por_query

CONSIDERA:
- False sharing: alinear buffers privados a línea de caché (64 bytes)
- schedule(dynamic, 32): bueno si Q no es múltiplo de T

Escribe SOLO el código C con OpenMP. Sin explicaciones.
```

---

### Tarea 2.2 — Script de Validación de Correctitud

**Modelo:** `gpt-4o-mini` / `claude-haiku`

```
PROMPT PARA AGENTE 2.2:

Escribe `scripts/validate.sh`, un script Bash que verifica que las tres
implementaciones de KNN producen los mismos resultados.

LÓGICA:
1. Genera un dataset pequeño fijo: N=1000, D=10, Q=100, k=5, seed=42
2. Ejecuta bin/knn_seq y guarda predictions_seq.txt
3. Ejecuta bin/knn_omp y guarda predictions_omp.txt
4. Ejecuta bin/knn_cuda y guarda predictions_cuda.txt (si existe el binario)
5. Compara predictions_seq vs predictions_omp → deben ser IDÉNTICOS (diff)
6. Compara predictions_seq vs predictions_cuda → deben ser IDÉNTICOS (diff)
7. Imprime PASS / FAIL con colores (verde/rojo) para cada comparación

Maneja el caso donde knn_cuda no existe (imprime SKIP en amarillo).
Escribe SOLO el script bash con shebang. Hazlo ejecutable (chmod +x implícito).
```

---

## FASE 3 — Implementación CUDA (Semana 3-5)

### Tarea 3.1 — Kernel de Distancias CUDA

**Modelo:** `claude-opus` / `gpt-4o` — **MODELO PESADO OBLIGATORIO**
*(Diseño de kernel CUDA es la parte más crítica y propensa a errores sutiles)*

```
PROMPT PARA AGENTE 3.1:

Eres un experto CUDA con 10 años de experiencia optimizando kernels GPU.
Escribe `src/kernels.cuh` con los kernels de KNN para NVIDIA GPU.

KERNEL 1 — compute_distances:
  __global__ void compute_distances(
      const float* __restrict__ train,   // [N x D] row-major en device
      const float* __restrict__ query,   // [Q x D] row-major en device
      float* distances,                   // [Q x N] salida en device
      int N, int D, int Q)

  ESTRATEGIA:
  - Grid 2D: gridDim.x = ceil(N/BLOCK_X), gridDim.y = ceil(Q/BLOCK_Y)
  - Block: BLOCK_X=16, BLOCK_Y=16 (256 threads/block)
  - Tiling con shared memory: carga tiles de train y query en smem
  - Tile size: 16x16 floats = 1KB por array → total 2KB shared memory por block
  - Distancia euclidea al cuadrado (sin sqrt)
  - Usa __restrict__ para hints de alias

KERNEL 2 — find_k_nearest (usando Thrust):
  Después del kernel 1, para cada query row en distances[q*N ... q*N+N-1]:
  - Usa thrust::sort con índices para obtener top-k
  - Aplica voting por mayoría para clasificación

TAMBIÉN incluye:
  - Macro CHECK_CUDA(call) para manejo de errores
  - Función host: void knn_cuda_predict(...) que maneja:
    cudaMalloc, cudaMemcpy H→D, kernel launch, cudaMemcpy D→H, cudaFree
  - Timing con cudaEvent_t (separar: transfer_time vs compute_time)
  - Comentarios explicando cada decisión de diseño

Escribe SOLO el archivo .cuh con guards #ifndef. Sin explicaciones fuera del código.
```

---

### Tarea 3.2 — Host Code CUDA y Main

**Modelo:** `claude-sonnet` / `gpt-4o`

```
PROMPT PARA AGENTE 3.2:

Escribe `src/knn_cuda.cu`, el archivo principal CUDA que usa los kernels
definidos en kernels.cuh (ya implementado).

ASUME que kernels.cuh define:
  - void knn_cuda_predict(float* h_train, float* h_labels, int N, int D,
                           float* h_query, int Q, int k, float* h_predictions,
                           float* out_transfer_ms, float* out_compute_ms)
  - Macro CHECK_CUDA(call)

IMPLEMENTA en knn_cuda.cu:
1. main() con argumentos CLI: --train --query --k --output --device
2. Selección de GPU: cudaSetDevice(device_id)
3. Impresión de info del device: nombre, memoria total, compute capability
4. Carga de datos desde formato binario (mismo que knn_seq.c usa)
5. Llamada a knn_cuda_predict()
6. Guardado de predictions en archivo
7. Reporte de tiempos: transfer_to_gpu, compute, transfer_from_gpu, total

IMPORTANTE: El archivo .cu solo tiene el flujo de datos, NO los kernels.
Los kernels están en kernels.cuh (incluir con #include "kernels.cuh").

Escribe SOLO el código .cu. Sin explicaciones.
```

---

### Tarea 3.3 — Optimización CUDA con Shared Memory

**Modelo:** `claude-opus` / `gpt-4o` — **MODELO PESADO**
*(Optimización avanzada, alta probabilidad de bugs si usa modelo débil)*

```
PROMPT PARA AGENTE 3.3:

Revisa y optimiza el kernel compute_distances de kernels.cuh.
Implementa las siguientes optimizaciones en orden de prioridad:

OPTIMIZACIÓN 1 — Tiled Matrix Multiplication pattern:
  - Divide la multiplicación distancia en tiles de TxT (T=16 o T=32)
  - Carga tile de train en __shared__ float s_train[T][T]
  - Carga tile de query en __shared__ float s_query[T][T]
  - __syncthreads() antes y después de usar smem
  - Reduce accesos a memoria global de O(Q*N*D) a O(Q*N*D/T)

OPTIMIZACIÓN 2 — Memory coalescing:
  - Asegura que threads consecutivos accedan a posiciones consecutivas en memoria
  - Transpose train matrix si es necesario para acceso coalesced
  - Documenta el patrón de acceso con un comentario ASCII

OPTIMIZACIÓN 3 — Reducción para k-NN selection:
  - En lugar de sort completo (O(N log N)), implementa partial sort top-k
  - Usa un max-heap de tamaño k en registros del thread (para k<=32)
  - Esto es O(N*k) pero con menor overhead para k pequeño

Para cada optimización:
- Muestra el código ANTES y DESPUÉS (como comentario)
- Indica el speedup teórico esperado

Escribe el archivo kernels.cuh completo y optimizado.
```

---

## FASE 4 — Experimentación (Semana 5-6)

### Tarea 4.1 — Script de Benchmark Automatizado

**Modelo:** `gpt-4o-mini` / `claude-haiku`

```
PROMPT PARA AGENTE 4.1:

Escribe `scripts/benchmark.sh`, un script Bash que automatiza todos
los experimentos de benchmarking del proyecto KNN.

MATRIZ DE EXPERIMENTOS:
  N_VALUES=(1000 5000 10000 50000 100000 500000)
  D_VALUES=(2 10 50 100 500 1000)
  K_VALUES=(1 3 5 10)
  THREADS_OMP=(1 2 4 8 16)
  RUNS=5  # repeticiones por configuración para promedio

PARA CADA COMBINACIÓN (N, D, K):
  1. Genera dataset con data_gen.py
  2. Ejecuta bin/knn_seq → captura tiempo
  3. Para cada T en THREADS_OMP: ejecuta bin/knn_omp --threads T → captura tiempo
  4. Ejecuta bin/knn_cuda → captura tiempo_transfer + tiempo_compute
  5. Repite RUNS veces, calcula promedio y desviación estándar
  6. Append a results/benchmark_results.csv

FORMATO CSV:
  n,d,k,impl,threads,run,time_ms,transfer_ms,compute_ms,speedup_vs_seq

TAMBIÉN:
  - Barra de progreso simple con % completado
  - Skip automático si N*D > 5e8 (para evitar OOM en GPU)
  - Log de errores en results/errors.log
  - Al final: imprime resumen de cuántas configuraciones completaron

Escribe SOLO el script bash.
```

---

### Tarea 4.2 — Notebook de Análisis

**Modelo:** `claude-sonnet` / `gpt-4o`

```
PROMPT PARA AGENTE 4.2:

Escribe el código Python completo para `analysis/results.ipynb` como script
.py ejecutable (luego se convierte a notebook). Asume que existe el archivo
results/benchmark_results.csv con columnas:
  n, d, k, impl, threads, run, time_ms, transfer_ms, compute_ms, speedup_vs_seq

GENERA LAS SIGUIENTES VISUALIZACIONES (una por celda/sección):

1. SPEEDUP vs N (fijo D=100, K=5):
   - Líneas para: omp_t1, omp_t4, omp_t8, omp_t16, cuda_compute, cuda_total
   - Escala log en eje X

2. SPEEDUP vs D (fijo N=50000, K=5):
   - Mismas líneas que arriba
   - Muestra dónde CUDA supera a OpenMP

3. HEATMAP de speedup CUDA vs OpenMP:
   - Ejes: N (filas) vs D (columnas)
   - Color: speedup_cuda / speedup_omp_best
   - Anota el punto de cruce (valor ~1.0) con línea de contorno

4. DESGLOSE de tiempo CUDA (N=100000, D=500):
   - Pie chart: transfer_H2D, compute, transfer_D2H
   - Barras apiladas para varios valores de N

5. ESCALABILIDAD FUERTE OpenMP (N=100000, D=100, K=5):
   - Speedup vs num_threads con línea ideal (y=x)
   - Eficiencia E = S/p en eje secundario

Usa matplotlib + seaborn. Estilo: seaborn-v0_8-paper. Paleta: viridis / RdYlGn.
Guarda cada figura en analysis/figures/fig_N.png a 300 DPI.
Escribe SOLO el código Python, con secciones separadas por # %% (Jupyter cells).
```

---

## FASE 5 — Análisis y Reporte (Semana 6-7)

### Tarea 5.1 — Interpretación de Resultados

**Modelo:** `claude-opus` / `gpt-4o` — **MODELO PESADO**
*(Razonamiento analítico profundo, no es generación mecánica)*

```
PROMPT PARA AGENTE 5.1:

Eres un investigador en computación paralela. Analiza los siguientes resultados
de benchmark (sustituye [DATOS] con el contenido real de tu CSV):

[DATOS: pega aquí las primeras 50 filas de benchmark_results.csv]

GENERA un análisis técnico en Markdown con estas secciones:

## Hallazgos Principales
- Top 3 insights más importantes de los datos

## Análisis por Arquitectura
### OpenMP
- ¿A partir de qué N el speedup se estabiliza?
- ¿Hay degradación por false sharing o saturación de memoria?
- Eficiencia real vs teórica (Ley de Amdahl)

### CUDA
- ¿Cuál es el overhead de transferencia como % del tiempo total?
- ¿A partir de qué N*D se vuelve rentable usar GPU?
- ¿El speedup crece linealmente con N o tiene un plateau?

## Punto de Cruce OpenMP vs CUDA
- Define la frontera N_crossover(D) donde CUDA supera a OpenMP
- ¿Depende más de N o de D?

## Bottlenecks Identificados
- Para OpenMP: ¿CPU-bound o memory-bound?
- Para CUDA: ¿Compute-bound o memory bandwidth-bound?

## Recomendaciones Prácticas
- ¿Cuándo usar cada implementación en producción?
- ¿Qué optimizaciones adicionales tendrían mayor impacto?

Sé específico con números de los datos. Evita generalidades.
```

---

### Tarea 5.2 — README del Repositorio

**Modelo:** `gpt-4o-mini` / `claude-haiku`

```
PROMPT PARA AGENTE 5.2:

Escribe el README.md completo para el repositorio GitHub del proyecto
"KNN Paralelo: OpenMP vs CUDA".

ESTRUCTURA DEL REPO (ya existe):
  src/knn_seq.c, src/knn_omp.c, src/knn_cuda.cu, src/kernels.cuh
  scripts/benchmark.sh, scripts/validate.sh
  analysis/results.ipynb
  data_gen.py
  Makefile

SECCIONES DEL README:
1. Título + badges (build, license MIT, CUDA version)
2. Descripción (3 líneas max)
3. Requisitos: GCC ≥11, CUDA ≥12, Python ≥3.10, GPU compute ≥6.0
4. Instalación (4 comandos max)
5. Uso rápido:
   - Generar datos
   - Compilar todo
   - Ejecutar validación
   - Ejecutar benchmark completo
6. Estructura del proyecto (árbol de directorios)
7. Resultados principales (tabla de speedup resumen con valores placeholder)
8. Cómo reproducir (paso a paso en 5 pasos)
9. Licencia MIT

Usa Markdown limpio. Sin emojis excesivos. Tono técnico-profesional.
```

---

### Tarea 5.3 — Informe Técnico Final

**Modelo:** `claude-opus` — **MODELO PESADO OBLIGATORIO**
*(Documento académico de 15-20 páginas, requiere coherencia y profundidad)*

```
PROMPT PARA AGENTE 5.3:

Redacta el informe técnico final del proyecto de Computación Paralela.
Usa los resultados del análisis (Tarea 5.1) y el siguiente contexto:

CONTEXTO DEL PROYECTO:
- Materia: Computación Paralela
- Algoritmo: K-Nearest Neighbors (KNN) clasificación
- Implementaciones: Secuencial (C), OpenMP (CPU), CUDA (GPU)
- Variables evaluadas: N ∈ {1K..500K}, D ∈ {2..1000}, K ∈ {1,3,5,10}
- Hardware: [especifica tu CPU y GPU aquí]
- [RESULTADOS: pega el análisis de la Tarea 5.1]

ESTRUCTURA DEL INFORME:
1. Resumen ejecutivo (media página)
2. Introducción y motivación
3. Fundamento teórico (KNN, OpenMP, CUDA, métricas)
4. Diseño e implementación (decisiones técnicas clave, NO todo el código)
5. Metodología experimental (diseño factorial, variables, control)
6. Resultados (con referencias a figuras fig_1.png..fig_5.png)
7. Discusión (interpretación, limitaciones, amenazas a la validez)
8. Conclusiones y trabajo futuro
9. Referencias

TONO: Académico-técnico. Primera persona del plural ("Se implementó", "Los resultados muestran").
EXTENSIÓN: 2500-3500 palabras.
FORMATO: Markdown con headers ## y tablas Markdown.

Escribe el informe completo, no un esquema.
```

---

*Fin del documento de prompts de orquestación.*
