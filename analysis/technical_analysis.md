# Análisis Técnico: KNN Paralelo OpenMP vs CUDA

> Basado en `results/benchmark_results.csv` — 2360 ejecuciones sobre 72 configuraciones (N ∈ {1000, 5000, 10000}, D ∈ {2, 10, 50, 100, 500, 1000}, K ∈ {1, 3, 5, 10}), 5 repeticiones por configuración.

---

## Hallazgos Principales

1. **CUDA domina abrumadoramente en datasets con dimensionalidad significativa.** El speedup promedio de CUDA es 64x contra 3.8x de OpenMP. En N=10000, D=1000, CUDA alcanza 197x de speedup — dos órdenes de magnitud sobre el secuencial.

2. **La dimensionalidad D es el factor determinante para CUDA.** El speedup de CUDA crece de 6.4x (D=2) a 164x (D=1000) promediado sobre N y K. OpenMP apenas escala con D: se mantiene plano entre 3.1x y 4.2x.

3. **OpenMP alcanza saturación temprana.** La eficiencia cae del 96% (2 hilos) al 39% (16 hilos). A partir de 8 hilos, la ganancia marginal es negativa (6.48x → 6.24x a 16 hilos), indicando memory-bound en la arquitectura de doble NUMA del Xeon E5640.

---

## Análisis por Arquitectura

### OpenMP

**Speedup por número de hilos (promedio sobre N, D, K):**

| Hilos | Speedup | Eficiencia |
|-------|---------|------------|
| 1     | 1.00x   | 100.0%     |
| 2     | 1.91x   | 95.7%      |
| 4     | 3.46x   | 86.4%      |
| 8     | 6.48x   | 80.8%      |
| 16    | 6.24x   | 39.1%      |

**¿A partir de qué N se estabiliza el speedup?** El speedup de OpenMP es notablemente estable con respecto a N. Pasa de 2.4x (N=1000) a 4.4x (N=10000). La mejora con N se debe a que el overhead de creación de hilos se amortiza sobre más trabajo. No se observa un plateau marcado en este rango.

**¿Hay degradación por false sharing?** La implementación usa buffers privados por hilo con alineación a línea de caché (64 bytes), lo que mitiga false sharing. La degradación de 8→16 hilos (6.48x → 6.24x) sugiere saturación del ancho de banda de memoria, no false sharing. El Xeon E5640 tiene 8 núcleos físicos con HyperThreading; los 8 hilos adicionales comparten recursos de ejecución sin añadir canales de memoria.

**Eficiencia real vs Ley de Amdahl:** Con 8 hilos y speedup 6.48x, la fracción paralela según Amdahl es f = (1 - 1/6.48) / (1 - 1/8) ≈ 0.968, es decir, 96.8% del código es paralelizable. El 3.2% secuencial explica la pérdida de eficiencia. Con 16 hilos, la fracción secuencial efectiva del hardware (HyperThreading no escala memoria) domina.

### CUDA

**Desglose de tiempo por dimensionalidad (promedio sobre N, K):**

| D   | Transfer (%) | Compute (%) | Speedup Total |
|-----|-------------|-------------|---------------|
| 2   | 4.7%        | 95.3%       | 6.4x          |
| 10  | 8.4%        | 91.6%       | 18.5x         |
| 50  | 16.9%       | 83.1%       | 48.3x         |
| 100 | 22.0%       | 78.0%       | 73.0x         |
| 500 | 31.0%       | 69.0%       | 107.3x        |
| 1000| 22.2%       | 77.8%       | 164.5x        |

**¿Cuál es el overhead de transferencia?** 17.2% promedio global. Crece con D de 4.7% a 31.0%, pero cae a 22.2% en D=1000 porque el cómputo escala con O(N×Q×D) mientras la transferencia escala con O((N+Q)×D). La transferencia es significativa pero no dominante en este rango.

**¿A partir de qué N×D se vuelve rentable la GPU?** El punto de cruce donde CUDA supera al mejor OpenMP ocurre aproximadamente en:
- N=1000, D=50: ratio = 2.03x (CUDA gana)
- N=5000, D=2: ratio = 0.97x (CUDA pierde por poco)
- N=10000, D=2: ratio = 1.61x (CUDA gana para cualquier D)

La frontera de rentabilidad está en N×D ≈ 50,000 operaciones (N=1000, D=50 o N=5000, D=10). Por debajo de este umbral, el overhead de lanzamiento de kernels y transferencias PCIe supera la ganancia del paralelismo masivo.

**¿El speedup crece linealmente con N?** No exactamente. De N=1000 a N=5000, el speedup CUDA salta de 13.8x a 69.4x (5x). De N=5000 a N=10000, sube a 98.3x (1.4x). El crecimiento es superlineal inicialmente (mayor ocupación de SMs, mejor amortización del overhead fijo) y luego se acerca a lineal.

---

## Punto de Cruce OpenMP vs CUDA

**Heatmap: Ratio speedup CUDA / mejor OpenMP (K=5):**

| N      | D=2   | D=10  | D=50  | D=100 | D=500  | D=1000 |
|--------|-------|-------|-------|-------|--------|--------|
| 1000   | 0.51x | 0.73x | 2.03x | 3.19x | 6.28x  | —      |
| 5000   | 0.94x | 2.36x | 5.76x | 10.0x | 15.9x  | 17.2x  |
| 10000  | 1.61x | 4.26x | 9.25x | 14.7x | 20.2x  | 23.0x  |

Casos donde CUDA pierde contra OpenMP: solo 9 de 72 configuraciones, todas con D≤10 y N≤5000. El peor caso es N=1000, D=2, K=10 donde CUDA rinde solo 0.35x del mejor OpenMP.

**¿Depende más de N o de D?** D es el factor dominante. El ratio CUDA/OpenMP crece más rápido al aumentar D que al aumentar N:
- Manteniendo N=5000, pasar D:2→100 multiplica el ratio por ~10.7x
- Manteniendo D=2, pasar N:1000→10000 multiplica el ratio por ~3.2x

Esto es esperable: CUDA paraleliza sobre el producto N×Q×D con miles de threads, mientras OpenMP solo paraleliza sobre Q. A mayor D, más operaciones por thread CUDA que se benefician del ancho de banda de memoria de la GPU.

---

## Bottlenecks Identificados

### OpenMP: Memory-bound

- La eficiencia cae drásticamente a 16 hilos (39%) mientras los 8 hilos mantienen 81%.
- El Xeon E5640 tiene arquitectura NUMA de 2 sockets; accesos cross-NUMA duplican latencia.
- La implementación paraleliza sobre Q (filas de query). Cada hilo lee toda la matriz `train` (N×D floats). Para N=10000, D=1000, son ~40 MB — cabe en L3 cache compartida, pero 16 hilos compitiendo por el mismo bus de memoria saturan el ancho de banda (~32 GB/s para DDR3-1333 en este Xeon).
- Recomendación: paralelizar también sobre el bucle interno de features (D) usando `#pragma omp parallel for collapse(2)` para mejor utilización de cache, o implementar tiling para reducir presión sobre el bus de memoria.

### CUDA: Transfer-bound para D bajo, Compute-bound para D alto

- Para D≤10, la transferencia H↔D representa menos del 9% del tiempo total. El cuello de botella es el lanzamiento de kernels y la baja ocupación de SMs (pocas operaciones por thread).
- Para D=500, la transferencia llega al 31%. Para D=1000, cae a 22% porque el cómputo domina.
- El kernel `compute_distances` con tiling 16×16 en shared memory tiene buena utilización del ancho de banda de memoria global (~80% según el perfil de speedup).
- La selección de top-K con `thrust::sort_by_key` es O(N log N) por query. Para N=10000 esto es aceptable, pero para N=500000 sería un bottleneck. La optimización 3 de la Fase 3 (max-heap en registros) reduciría esto a O(N×K).

---

## Recomendaciones Prácticas

### ¿Cuándo usar cada implementación?

| Condición | Implementación recomendada | Justificación |
|-----------|---------------------------|---------------|
| N×D < 50,000 | OpenMP (8 hilos) | Overhead CUDA supera ganancia |
| N×D < 500,000 | OpenMP (8 hilos) o CUDA | Rendimiento similar; OpenMP más simple |
| N×D > 500,000 | CUDA | GPU gana por margen creciente |
| D < 10, cualquier N | OpenMP si N<5000; CUDA si N≥10000 | Baja utilización de GPU en pocas features |
| Latencia crítica, batch pequeño | OpenMP | Evita latencia de transferencia PCIe |
| Throughput, batch grande | CUDA | Paralelismo masivo compensa transferencia |

### Optimizaciones de mayor impacto

1. **CUDA: Max-heap en registros para top-K** — Reemplazar `thrust::sort_by_key` (O(N log N)) por heap de tamaño K en registros. Para K≤32, cabe en registros. Impacto estimado: 15-25% de reducción en `compute_ms` para N grande.

2. **CUDA: Streams asíncronos para solapar transferencia y cómputo** — Usar `cudaMemcpyAsync` con streams para ocultar latencia de transferencia. Para D≥500, esto podría reducir el tiempo total en 10-20%.

3. **OpenMP: Tiling sobre D con collapse(2)** — Paralelizar sobre Q y D simultáneamente con tiles que quepan en L2 cache. Impacto estimado: mejor escalabilidad a 16 hilos (de 6.24x a ~10x).

4. **Ambos: Usar SIMD (AVX en CPU, half2 en GPU)** — Empaquetar operaciones de distancia para usar instrucciones vectoriales. En CPU, `_mm256_fmadd_ps` para 8 floats simultáneos. En GPU, `half2` para duplicar throughput con precisión reducida.
