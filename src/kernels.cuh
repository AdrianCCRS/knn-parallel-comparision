#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE   16
#define MAX_K  32
/*
 * BATCH_Q: número de queries procesadas por iteración en knn_cuda_predict.
 * d_dist ocupa BATCH_Q * N * 4 bytes en lugar de Q * N * 4.
 * Con BATCH_Q=512 y N=500,000: 512 * 500000 * 4 = ~1 GB (manejable).
 * Reducir si la GPU tiene poca VRAM; aumentar para mayor paralelismo.
 */
#define BATCH_Q 512

#define CHECK_CUDA(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));        \
        exit(1);                                                    \
    }                                                               \
} while (0)

/*
 * compute_distances — tiled kernel, 16x16 threads per block.
 *
 * Grid: 2D  (ceil(N/TILE), ceil(Q_batch/TILE))
 * Block: 2D (TILE, TILE) = (16, 16)    256 threads
 *
 * Each block computes a TILE×TILE submatrix of the Q_batch×N distance matrix.
 * Shared memory: s_train[16][16] + s_query[16][16] = 2 KB per block.
 *
 * Coalesced loads: threadIdx.x maps to the feature dimension (fast index).
 */
__global__ void compute_distances(
    const float * __restrict__ train,
    const float * __restrict__ query,
    float *distances,
    int N, int D, int Q_batch)
{
    __shared__ float s_train[TILE][TILE];
    __shared__ float s_query[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int train_row = blockIdx.x * TILE + tx;
    int query_row = blockIdx.y * TILE + ty;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        printf("[DEBUG compute_distances] grid=(%d,%d) block=(%d,%d) "
               "N=%d D=%d Q_batch=%d\n",
               gridDim.x, gridDim.y, blockDim.x, blockDim.y,
               N, D, Q_batch);

    float dist = 0.0f;

    for (int tile = 0; tile < (D + TILE - 1) / TILE; tile++) {
        int d = tile * TILE + tx;

        if (blockIdx.x * TILE + ty < N && d < D)
            s_train[ty][tx] = train[(blockIdx.x * TILE + ty) * (size_t)D + d];
        else
            s_train[ty][tx] = 0.0f;

        if (blockIdx.y * TILE + ty < Q_batch && d < D)
            s_query[ty][tx] = query[(blockIdx.y * TILE + ty) * (size_t)D + d];
        else
            s_query[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float diff = s_query[ty][k] - s_train[tx][k];
            dist += diff * diff;
        }

        __syncthreads();
    }

    if (train_row < N && query_row < Q_batch)
        distances[(size_t)query_row * N + train_row] = dist;
}

/*
 * find_k_nearest — per-thread max-heap sobre una fila de distancias.
 *
 * Grid: 1D  (ceil(Q_batch / BLOCK_DIM))
 * Block: 1D (BLOCK_DIM)
 *
 * Majority vote O(num_classes) usando histograma en lugar de O(K²).
 * Asume labels enteras en [0, MAX_CLASSES).
 */
#define MAX_CLASSES 64
#define BLOCK_DIM   256

__global__ void find_k_nearest(
    const float *distances,
    const float *labels,
    int N, int Q_batch, int k,
    float *predictions)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q_batch)
        return;

    if (threadIdx.x == 0)
        printf("[DEBUG find_k_nearest] grid=%d block=%d "
               "N=%d Q_batch=%d k=%d\n",
               gridDim.x, blockDim.x,
               N, Q_batch, k);

    float heap_dist[MAX_K];
    float heap_label[MAX_K];
    int heap_size = 0;

    const float *row = distances + (size_t)q * N;

    for (int i = 0; i < N; i++) {
        float d = row[i];

        if (heap_size < k) {
            int idx = heap_size++;
            heap_dist[idx]  = d;
            heap_label[idx] = labels[i];
            while (idx > 0) {
                int p = (idx - 1) / 2;
                if (heap_dist[p] >= heap_dist[idx]) break;
                float td = heap_dist[p]; heap_dist[p] = heap_dist[idx]; heap_dist[idx] = td;
                float tl = heap_label[p]; heap_label[p] = heap_label[idx]; heap_label[idx] = tl;
                idx = p;
            }
        } else if (d < heap_dist[0]) {
            heap_dist[0]  = d;
            heap_label[0] = labels[i];
            int idx = 0;
            for (;;) {
                int largest = idx;
                int left  = 2 * idx + 1;
                int right = 2 * idx + 2;
                if (left  < heap_size && heap_dist[left]  > heap_dist[largest]) largest = left;
                if (right < heap_size && heap_dist[right] > heap_dist[largest]) largest = right;
                if (largest == idx) break;
                float td = heap_dist[idx]; heap_dist[idx] = heap_dist[largest]; heap_dist[largest] = td;
                float tl = heap_label[idx]; heap_label[idx] = heap_label[largest]; heap_label[largest] = tl;
                idx = largest;
            }
        }
    }

    /* Majority vote con histograma O(K + C) en lugar de O(K²) */
    int counts[MAX_CLASSES] = {0};
    for (int a = 0; a < heap_size; a++) {
        int cls = (int)heap_label[a];
        if (cls >= 0 && cls < MAX_CLASSES)
            counts[cls]++;
    }
    float best_label = heap_label[0];
    int best_count = 0;
    for (int c = 0; c < MAX_CLASSES; c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_label = (float)c;
        }
    }
    predictions[q] = best_label;
}

/*
 * knn_cuda_predict — host-side orchestrator con streaming batch.
 *
 * Procesa las Q queries en bloques de BATCH_Q para evitar alocar la
 * matriz de distancias completa Q×N (que crece cuadráticamente con N).
 * Memoria GPU para d_dist: BATCH_Q × N × 4 bytes en lugar de Q × N × 4.
 *
 * Para N=500,000 y BATCH_Q=512: d_dist = ~1 GB (vs 200 GB con Q=100,000).
 *
 * Timing: out_transfer_ms = suma de H→D+D→H de todos los batches,
 *         out_compute_ms  = suma de cómputo de kernels.
 */
static void knn_cuda_predict(
    float *h_train, float *h_labels, int N, int D,
    float *h_query, int Q, int k,
    float *h_predictions,
    float *out_transfer_ms, float *out_compute_ms)
{
    float *d_train, *d_labels, *d_query_batch, *d_dist, *d_pred_batch;
    CHECK_CUDA(cudaMalloc(&d_train,       (size_t)N * D    * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels,      (size_t)N        * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query_batch, (size_t)BATCH_Q * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dist,        (size_t)BATCH_Q * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pred_batch,  (size_t)BATCH_Q    * sizeof(float)));

    cudaEvent_t ev_h2d_s, ev_h2d_e, ev_cmp_s, ev_cmp_e, ev_d2h_s, ev_d2h_e;
    CHECK_CUDA(cudaEventCreate(&ev_h2d_s));
    CHECK_CUDA(cudaEventCreate(&ev_h2d_e));
    CHECK_CUDA(cudaEventCreate(&ev_cmp_s));
    CHECK_CUDA(cudaEventCreate(&ev_cmp_e));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_s));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_e));

    float total_transfer_ms = 0.0f, total_compute_ms = 0.0f;

    /* Transferir train y labels una sola vez (no cambian entre batches) */
    CHECK_CUDA(cudaEventRecord(ev_h2d_s));
    CHECK_CUDA(cudaMemcpy(d_train,  h_train,  (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels, (size_t)N     * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(ev_h2d_e));
    CHECK_CUDA(cudaEventSynchronize(ev_h2d_e));
    float t_tmp;
    CHECK_CUDA(cudaEventElapsedTime(&t_tmp, ev_h2d_s, ev_h2d_e));
    total_transfer_ms += t_tmp;
    fprintf(stderr, "[DEBUG] H2D train+labels: %.3f ms  (train=%zu MB, labels=%zu MB)\n",
            t_tmp,
            (size_t)N * D * sizeof(float) / (1024*1024),
            (size_t)N * sizeof(float) / (1024*1024));

    /* Procesar queries en batches de BATCH_Q */
    int n_batches = (Q + BATCH_Q - 1) / BATCH_Q;
    fprintf(stderr, "[DEBUG] Processing %d queries in %d batches of %d\n",
            Q, n_batches, BATCH_Q);
    for (int q_start = 0; q_start < Q; q_start += BATCH_Q) {
        int q_batch = (q_start + BATCH_Q <= Q) ? BATCH_Q : (Q - q_start);
        int batch_num = q_start / BATCH_Q + 1;
        fprintf(stderr, "[DEBUG] Batch %d/%d: q_start=%d q_batch=%d\n",
                batch_num, n_batches, q_start, q_batch);

        /* H→D: batch de queries */
        CHECK_CUDA(cudaEventRecord(ev_h2d_s));
        CHECK_CUDA(cudaMemcpy(d_query_batch,
                               h_query + (size_t)q_start * D,
                               (size_t)q_batch * D * sizeof(float),
                               cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(ev_h2d_e));
        CHECK_CUDA(cudaEventSynchronize(ev_h2d_e));
        CHECK_CUDA(cudaEventElapsedTime(&t_tmp, ev_h2d_s, ev_h2d_e));
        total_transfer_ms += t_tmp;
        fprintf(stderr, "[DEBUG] Batch %d H2D query: %.3f ms\n", batch_num, t_tmp);

        /* Kernel 1: compute_distances para el batch */
        CHECK_CUDA(cudaEventRecord(ev_cmp_s));
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (q_batch + TILE - 1) / TILE);
        fprintf(stderr, "[DEBUG] Batch %d compute_distances grid=(%d,%d) block=(%d,%d)\n",
                batch_num, grid.x, grid.y, block.x, block.y);
        compute_distances<<<grid, block>>>(d_train, d_query_batch, d_dist, N, D, q_batch);
        CHECK_CUDA(cudaGetLastError());
        fprintf(stderr, "[DEBUG] Batch %d compute_distances launched OK\n", batch_num);

        /* Kernel 2: find_k_nearest para el batch */
        dim3 block2(BLOCK_DIM);
        dim3 grid2((q_batch + BLOCK_DIM - 1) / BLOCK_DIM);
        fprintf(stderr, "[DEBUG] Batch %d find_k_nearest grid=%d block=%d\n",
                batch_num, grid2.x, block2.x);
        find_k_nearest<<<grid2, block2>>>(d_dist, d_labels, N, q_batch, k, d_pred_batch);
        CHECK_CUDA(cudaGetLastError());
        fprintf(stderr, "[DEBUG] Batch %d find_k_nearest launched OK\n", batch_num);

        CHECK_CUDA(cudaDeviceSynchronize());
        fprintf(stderr, "[DEBUG] Batch %d kernels completed (sync OK)\n", batch_num);
        CHECK_CUDA(cudaEventRecord(ev_cmp_e));
        CHECK_CUDA(cudaEventSynchronize(ev_cmp_e));
        CHECK_CUDA(cudaEventElapsedTime(&t_tmp, ev_cmp_s, ev_cmp_e));
        total_compute_ms += t_tmp;
        fprintf(stderr, "[DEBUG] Batch %d compute time: %.3f ms\n", batch_num, t_tmp);

        /* D→H: predicciones del batch */
        CHECK_CUDA(cudaEventRecord(ev_d2h_s));
        CHECK_CUDA(cudaMemcpy(h_predictions + q_start,
                               d_pred_batch,
                               (size_t)q_batch * sizeof(float),
                               cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(ev_d2h_e));
        CHECK_CUDA(cudaEventSynchronize(ev_d2h_e));
        CHECK_CUDA(cudaEventElapsedTime(&t_tmp, ev_d2h_s, ev_d2h_e));
        total_transfer_ms += t_tmp;
        fprintf(stderr, "[DEBUG] Batch %d D2H predictions: %.3f ms\n", batch_num, t_tmp);

        /* Verificar primeras predicciones del batch */
        if (q_start < 10) {
            fprintf(stderr, "[DEBUG] Batch %d first 5 predictions: %.0f %.0f %.0f %.0f %.0f\n",
                    batch_num,
                    h_predictions[q_start],
                    h_predictions[q_start+1],
                    h_predictions[q_start+2],
                    h_predictions[q_start+3],
                    h_predictions[q_start+4]);
        }
    }

    *out_transfer_ms = total_transfer_ms;
    *out_compute_ms  = total_compute_ms;

    CHECK_CUDA(cudaEventDestroy(ev_h2d_s));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_e));
    CHECK_CUDA(cudaEventDestroy(ev_cmp_s));
    CHECK_CUDA(cudaEventDestroy(ev_cmp_e));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_s));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_e));
    CHECK_CUDA(cudaFree(d_train));
    CHECK_CUDA(cudaFree(d_labels));
    CHECK_CUDA(cudaFree(d_query_batch));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_pred_batch));
}

#endif /* KERNELS_CUH */
