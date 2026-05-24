#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 16
#define MAX_K 32

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
 * Grid: 2D  (ceil(N/TILE), ceil(Q/TILE))
 * Block: 2D (TILE, TILE) = (16, 16)    256 threads
 *
 * Each block computes a TILE×TILE submatrix of the Q×N distance matrix.
 * Shared memory: s_train[16][16] + s_query[16][16] = 2 KB per block.
 *
 * Coalesced loads: threadIdx.x maps to the feature dimension (fast index).
 * For each D-tile all 256 threads cooperatively load one train element and
 * one query element; after __syncthreads a reduction loop accumulates the
 * squared difference over the tile.
 */
__global__ void compute_distances(
    const float * __restrict__ train,
    const float * __restrict__ query,
    float *distances,
    int N, int D, int Q)
{
    __shared__ float s_train[TILE][TILE];
    __shared__ float s_query[TILE][TILE];

    int tx = threadIdx.x;                     /* dimension index (0..15) */
    int ty = threadIdx.y;                     /* secondary row   (0..15) */
    int train_row = blockIdx.x * TILE + tx;   /* output row in N        */
    int query_row = blockIdx.y * TILE + ty;   /* output row in Q        */

    float dist = 0.0f;

    for (int tile = 0; tile < (D + TILE - 1) / TILE; tile++) {
        int d = tile * TILE + tx;             /* global feature column  */

        /* Coalesced load: consecutive tx → consecutive addresses */
        if (blockIdx.x * TILE + ty < N && d < D)
            s_train[ty][tx] = train[(blockIdx.x * TILE + ty) * (size_t)D + d];
        else
            s_train[ty][tx] = 0.0f;

        if (blockIdx.y * TILE + ty < Q && d < D)
            s_query[ty][tx] = query[(blockIdx.y * TILE + ty) * (size_t)D + d];
        else
            s_query[ty][tx] = 0.0f;

        __syncthreads();

        /*
         * Reduction over the tile dimension k.
         * Thread (tx,ty) needs s_query[ty][k] and s_train[tx][k] for
         * all k.  These were loaded cooperatively by all threads.
         */
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float diff = s_query[ty][k] - s_train[tx][k];
            dist += diff * diff;
        }

        __syncthreads();
    }

    if (train_row < N && query_row < Q)
        distances[(size_t)query_row * N + train_row] = dist;
}

/*
 * find_k_nearest — per-thread max-heap over one query row.
 *
 * Each thread processes one query point (row of the Q×N distance matrix).
 * Maintains a max-heap of k (dist, label) pairs in registers.
 * After scanning all N train points, a majority vote among the k nearest
 * neighbors determines the prediction.
 *
 * Grid: 1D  (ceil(Q / BLOCK_DIM))
 * Block: 1D (BLOCK_DIM)
 */
__global__ void find_k_nearest(
    const float *distances,
    const float *labels,
    int N, int Q, int k,
    float *predictions)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Q)
        return;

    float heap_dist[MAX_K];
    float heap_label[MAX_K];
    int heap_size = 0;

    const float *row = distances + (size_t)q * N;

    for (int i = 0; i < N; i++) {
        float d = row[i];

        if (heap_size < k) {
            /* push */
            int idx = heap_size++;
            heap_dist[idx] = d;
            heap_label[idx] = labels[i];
            while (idx > 0) {
                int p = (idx - 1) / 2;
                if (heap_dist[p] >= heap_dist[idx]) break;
                float td = heap_dist[p]; heap_dist[p] = heap_dist[idx]; heap_dist[idx] = td;
                float tl = heap_label[p]; heap_label[p] = heap_label[idx]; heap_label[idx] = tl;
                idx = p;
            }
        } else if (d < heap_dist[0]) {
            /* pop root, push new element, sift down */
            heap_dist[0] = d;
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

    /* Majority vote over the k neighbours */
    float best_label = heap_label[0];
    int best_count = 0;
    for (int a = 0; a < heap_size; a++) {
        int count = 0;
        for (int b = 0; b < heap_size; b++)
            if (heap_label[b] == heap_label[a])
                count++;
        if (count > best_count || (count == best_count && heap_label[a] < best_label)) {
            best_count = count;
            best_label = heap_label[a];
        }
    }
    predictions[q] = best_label;
}

/*
 * knn_cuda_predict — host-side orchestrator.
 *
 * 1. Allocate device memory for train, labels, query, distances, predictions.
 * 2. H→D copy: train + labels + query.
 * 3. Launch compute_distances (tiled 2D grid).
 * 4. Launch find_k_nearest (per-query heap + vote).
 * 5. D→H copy: predictions.
 * 6. Timing with cudaEvent_t: out_transfer_ms = H→D + D→H,
 *    out_compute_ms = all kernel execution.
 */
static void knn_cuda_predict(
    float *h_train, float *h_labels, int N, int D,
    float *h_query, int Q, int k,
    float *h_predictions,
    float *out_transfer_ms, float *out_compute_ms)
{
    cudaEvent_t ev_h2d_s, ev_h2d_e, ev_cmp_s, ev_cmp_e, ev_d2h_s, ev_d2h_e;
    CHECK_CUDA(cudaEventCreate(&ev_h2d_s));
    CHECK_CUDA(cudaEventCreate(&ev_h2d_e));
    CHECK_CUDA(cudaEventCreate(&ev_cmp_s));
    CHECK_CUDA(cudaEventCreate(&ev_cmp_e));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_s));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_e));

    float *d_train, *d_labels, *d_query, *d_dist, *d_pred;
    CHECK_CUDA(cudaMalloc(&d_train,  (size_t)N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, (size_t)N     * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_query,  (size_t)Q * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dist,   (size_t)Q * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pred,   (size_t)Q     * sizeof(float)));

    /* H→D */
    CHECK_CUDA(cudaEventRecord(ev_h2d_s));
    CHECK_CUDA(cudaMemcpy(d_train,  h_train,  (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels, (size_t)N     * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_query,  h_query,  (size_t)Q * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(ev_h2d_e));

    /* Compute */
    int blocks_per_dim = 256;
    CHECK_CUDA(cudaEventRecord(ev_cmp_s));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (Q + TILE - 1) / TILE);
    compute_distances<<<grid, block>>>(d_train, d_query, d_dist, N, D, Q);
    CHECK_CUDA(cudaGetLastError());

    dim3 block2(blocks_per_dim);
    dim3 grid2((Q + blocks_per_dim - 1) / blocks_per_dim);
    find_k_nearest<<<grid2, block2>>>(d_dist, d_labels, N, Q, k, d_pred);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_cmp_e));

    /* D→H */
    CHECK_CUDA(cudaEventRecord(ev_d2h_s));
    CHECK_CUDA(cudaMemcpy(h_predictions, d_pred, (size_t)Q * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_d2h_e));

    /* Timing */
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_e));
    float t_h2d, t_cmp, t_d2h;
    CHECK_CUDA(cudaEventElapsedTime(&t_h2d, ev_h2d_s, ev_h2d_e));
    CHECK_CUDA(cudaEventElapsedTime(&t_cmp, ev_cmp_s, ev_cmp_e));
    CHECK_CUDA(cudaEventElapsedTime(&t_d2h, ev_d2h_s, ev_d2h_e));
    *out_transfer_ms = t_h2d + t_d2h;
    *out_compute_ms  = t_cmp;

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(ev_h2d_s));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_e));
    CHECK_CUDA(cudaEventDestroy(ev_cmp_s));
    CHECK_CUDA(cudaEventDestroy(ev_cmp_e));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_s));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_e));
    CHECK_CUDA(cudaFree(d_train));
    CHECK_CUDA(cudaFree(d_labels));
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_pred));
}

#endif /* KERNELS_CUH */
