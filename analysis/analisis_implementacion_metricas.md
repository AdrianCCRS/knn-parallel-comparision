# Análisis del Proyecto KNN Paralelo

En este documento se analizan los errores de implementación detectados en el código fuente de las versiones Secuencial, OpenMP y CUDA, y se presenta un análisis exhaustivo de los resultados obtenidos tras generar nuevas métricas y visualizaciones del rendimiento.

---

## 1. Errores Obvios de Implementación

Al revisar el código fuente y contrastarlo con las reglas del proyecto (`AGENTS.md`) y las buenas prácticas de programación en C++/CUDA, se detectaron los siguientes errores graves:

### A. Violación de Regla de Diseño: Sort en GPU sin Thrust
La regla #5 exige explícitamente: *"Thrust para sort en GPU — Se usa `thrust::sort_by_key` para ordenar distancias. No reimplementar sort desde cero"*. 
Sin embargo, en el archivo `src/kernels.cuh`, el kernel `find_k_nearest` reimplementa un max-heap manual por hilo. Esta es una falta directa a los requerimientos del proyecto.

### B. Spilling a Memoria Local en CUDA (Caída de Rendimiento)
Relacionado con el punto anterior, el kernel `find_k_nearest` declara arreglos locales por hilo (`float heap_dist[MAX_K]; float heap_label[MAX_K];` donde `MAX_K = 32`). Al acceder a estos arreglos mediante índices calculados dinámicamente durante el recorrido del heap, el compilador `nvcc` se ve forzado a colocar estos arreglos en **Local Memory** (en la DRAM de la GPU) en lugar de en registros rápidos. Esto provoca una latencia masiva de lectura/escritura (spilling) que estrangula severamente el rendimiento de este kernel, volviéndose el cuello de botella a medida que `K` aumenta.

### C. Bank Conflicts en Memoria Compartida
En el kernel `compute_distances` (`src/kernels.cuh`), la memoria compartida `s_train[TILE][TILE]` se declara sin padding y se lee dentro del bucle de reducción como `s_train[tx][k]`. 
Dado que los hilos de un mismo warp tienen un `ty` constante pero varían en `tx` (stride 16), al leer `s_train[tx][k]` en el bucle interior se genera un **Bank Conflict de 8 vías** continuo (ya que 16 floats abarcan media línea de bancos de memoria, y los hilos acceden a direcciones muy distantes de forma simultánea). Esto reduce el ancho de banda efectivo de la memoria compartida en un factor enorme.

### D. Riesgo de Out of Memory (OOM) por Asignación Completa
La regla técnica exige el uso de procesamiento por lotes (*batched*) si `N * D * 4 > 4 GB`. Sin embargo, `knn_cuda_predict` realiza un `cudaMalloc` de la matriz completa de distancias de tamaño $Q \times N$. 
Dado que comúnmente $Q = N/5$, para $N = 100,000$ puntos la matriz de distancias requiere $100,000 \times 20,000 \times 4 \text{ bytes} \approx 8 \text{ GB}$. El script no divide el procesamiento de consultas por lotes, por lo que el programa falla silenciosamente o se cuelga para tamaños grandes, lo que explica por qué los tamaños $N = 500,000$ (solicitados en el benchmark full) no se pudieron procesar.

### E. Restricción Oculta en Generador de Datos
El log de errores (`results/errors.log`) muestra múltiples fallos con el mensaje `AssertionError: d must be < n`. El script `data_gen.py` prohíbe que se creen combinaciones donde las dimensiones excedan el número de puntos, lo que causa el fallo de casos como $N=1000, D=1000$, omitiéndolos de las gráficas.

---

## 2. Nuevas Métricas y Análisis de Rendimiento

Hemos extendido el script de análisis (`results_extended.py`) para extraer conclusiones mucho más profundas de la simple métrica de "speedup". Las nuevas figuras (fig_6 a fig_12) se han generado en el directorio respectivo.

### 2.1. Overhead de Transferencia PCIe en CUDA (Fig. 9)
La gráfica muestra que el overhead de mover datos Host-to-Device (H↔D) no es despreciable. 
- **En promedio, la transferencia ocupa el 17.3%** del tiempo total de la GPU.
- En configuraciones pequeñas ($N=1000$), la latencia combinada de lanzar los kernels y mover memoria domina casi por completo, llegando hasta un **58.7% de penalización**. En estos tamaños pequeños, el speedup de CUDA cae drásticamente (media de $13.8\times$, mínimo de $0.74\times$, lo que significa que a veces es *más lento* que la CPU en secuencial).

### 2.2. Eficiencia Escalabilidad OpenMP (Fig. 7 y 12)
Se analizó cómo la versión OpenMP aprovecha los hilos usando **Eficiencia** (Speedup / Hilos) y la **Ley de Amdahl**:
- **8 Hilos**: Tiene una eficiencia media excelente del **81.2%**.
- **16 Hilos**: La eficiencia colapsa a una media de **37.8%**. Esto es un indicio clásico de que a partir de cierto nivel, la implementación es limitada por el ancho de banda de memoria de la CPU (Memory Bound), no por el cómputo.
- Al ajustar los resultados a la Ley de Amdahl, encontramos un límite superior estricto del speedup de OpenMP cercano a $9\times$, sin importar cuántos hilos pongamos, probablemente a causa de la saturación del bus de la memoria RAM o variables compartidas no optimizadas.

### 2.3. Estabilidad y Varianza - CV% (Fig. 11)
Utilizando el **Coeficiente de Variación (CV = std/mean)**, validamos la estabilidad de los resultados en múltiples ejecuciones. 
OpenMP mostró la mayor varianza (a veces superior al umbral óptimo del 5%). Esto es común debido a la sobrecarga del planificador del sistema operativo manejando y pausando múltiples hilos de manera dinámica (`schedule(dynamic, 32)`).

### 2.4. Sensibilidad al Hiperparámetro K (Fig. 10)
Tanto la versión CPU como GPU muestran degradación de rendimiento conforme K aumenta. En el caso de CUDA, el declive es muy notorio. Esto comprueba de manera empírica nuestro hallazgo de la memoria local: dado que cada hilo de la GPU mantiene un heap de tamaño `K`, a mayor K, mayor es la cantidad de memoria volcada en la lenta Local Memory, penalizando severamente el cómputo en la fase de búsqueda (`find_k_nearest`).

---

## 3. Conclusiones y Recomendaciones

1. **CUDA Domina en Masividad:** Para datos donde $N \ge 10,000$ y alta dimensionalidad ($D \ge 500$), CUDA no tiene competencia (logró un speedup del cómputo máximo de **~189x** frente a secuencial). Sin embargo, su código actual está severamente comprometido por un mal diseño algorítmico y bank conflicts.
2. **Rehacer la Fase de Búsqueda GPU:** Es fundamental remover el heap manual y obedecer la regla del proyecto: utilizar `thrust::sort_by_key`. Esto mitigará la presión de registros y local memory en la GPU.
3. **Procesamiento por Lotes Urgente:** Dividir la matriz $Q \times N$ en bloques manejables dentro del kernel (o desde el host) es la única forma de que el proyecto pueda lidiar con escalas de producción reales ($N > 100k$) sin colapsar por OOM.
4. **Padding en Memoria Compartida:** Se recomienda evaluar la adición de padding en la asignación `__shared__ float s_train[TILE][TILE+1]` o trasponer las lecturas para eliminar los *bank conflicts* al calcular el error cuadrático.
