#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "kernels.cuh"

static float *load_npy(const char *path, int *out_n, int *out_d,
                        float **out_labels)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        exit(1);
    }
    int32_t N, D;
    if (fread(&N, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "error: reading N from '%s'\n", path);
        exit(1);
    }
    if (fread(&D, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "error: reading D from '%s'\n", path);
        exit(1);
    }
    float *data = (float *)malloc((size_t)N * D * sizeof(float));
    if (!data) {
        fprintf(stderr, "error: malloc %d x %d floats\n", N, D);
        exit(1);
    }
    size_t nd = (size_t)N * D;
    if (fread(data, sizeof(float), nd, f) != nd) {
        fprintf(stderr, "error: reading data from '%s'\n", path);
        exit(1);
    }
    if (out_labels) {
        *out_labels = (float *)malloc((size_t)N * sizeof(float));
        if (!*out_labels) {
            fprintf(stderr, "error: malloc %d labels\n", N);
            exit(1);
        }
        size_t nl = (size_t)N;
        if (fread(*out_labels, sizeof(float), nl, f) != nl) {
            fprintf(stderr, "error: reading labels from '%s'\n", path);
            exit(1);
        }
    }
    fclose(f);
    *out_n = (int)N;
    *out_d = (int)D;
    return data;
}

int main(int argc, char **argv)
{
    const char *train_path = NULL;
    const char *query_path = NULL;
    const char *output_path = NULL;
    int k = 5;
    int device = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc)
            train_path = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_path = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = atoi(argv[++i]);
    }
    if (!train_path || !query_path || !output_path) {
        fprintf(stderr, "usage: %s --train <file.npy> --query <file.npy>"
                        " --k <int> --output <file.txt> [--device <int>]\n",
                argv[0]);
        return 1;
    }
    if (k <= 0 || k > MAX_K) {
        fprintf(stderr, "error: k=%d out of range [1, %d] (MAX_K limit)\n",
                k, MAX_K);
        return 1;
    }

    int dev_count;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (device >= dev_count) {
        fprintf(stderr, "error: device %d not available (%d devices)\n",
                device, dev_count);
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "device: %s  mem=%zuMB  cc=%d.%d\n",
            prop.name,
            prop.totalGlobalMem / (1024 * 1024),
            prop.major, prop.minor);

    int N, D, Q, Dq;
    float *labels = NULL;
    float *train = load_npy(train_path, &N, &D, &labels);
    float *query = load_npy(query_path, &Q, &Dq, NULL);
    if (D != Dq) {
        fprintf(stderr, "error: dimension mismatch train=%d query=%d\n", D, Dq);
        return 1;
    }

    fprintf(stderr, "[DEBUG] train=%p (%d x %d = %zu MB) labels=%p (%d = %zu MB) "
                    "query=%p (%d x %d = %zu MB)\n",
            (void*)train, N, D, (size_t)N * D * sizeof(float) / (1024*1024),
            (void*)labels, N, (size_t)N * sizeof(float) / (1024*1024),
            (void*)query, Q, D, (size_t)Q * D * sizeof(float) / (1024*1024));

    size_t alloc_train    = (size_t)N * D * sizeof(float);
    size_t alloc_labels   = (size_t)N * sizeof(float);
    size_t alloc_qbatch   = (size_t)BATCH_Q * D * sizeof(float);
    size_t alloc_dist     = (size_t)BATCH_Q * N * sizeof(float);
    size_t alloc_pred     = (size_t)BATCH_Q * sizeof(float);
    size_t total_gpu_mem  = alloc_train + alloc_labels + alloc_qbatch
                          + alloc_dist + alloc_pred;
    fprintf(stderr, "[DEBUG] GPU memory needed:\n"
                    "  d_train:       %zu MB  (%zu bytes)\n"
                    "  d_labels:      %zu MB  (%zu bytes)\n"
                    "  d_query_batch: %zu MB  (%zu bytes)\n"
                    "  d_dist:        %zu MB  (%zu bytes)\n"
                    "  d_pred_batch:  %zu MB  (%zu bytes)\n"
                    "  TOTAL:         %zu MB  (%zu bytes)\n"
                    "  GPU global:    %zu MB\n",
            alloc_train / (1024*1024), alloc_train,
            alloc_labels / (1024*1024), alloc_labels,
            alloc_qbatch / (1024*1024), alloc_qbatch,
            alloc_dist / (1024*1024), alloc_dist,
            alloc_pred / (1024*1024), alloc_pred,
            total_gpu_mem / (1024*1024), total_gpu_mem,
            prop.totalGlobalMem / (1024*1024));

    if (total_gpu_mem > prop.totalGlobalMem) {
        fprintf(stderr, "ERROR: GPU memory needed (%zu MB) exceeds device memory (%zu MB)\n",
                total_gpu_mem / (1024*1024), prop.totalGlobalMem / (1024*1024));
        return 1;
    }

    float *predictions = (float *)malloc((size_t)Q * sizeof(float));
    float transfer_ms = 0.0f, compute_ms = 0.0f;

    fprintf(stderr, "[DEBUG] Calling knn_cuda_predict: N=%d D=%d Q=%d k=%d BATCH_Q=%d\n",
            N, D, Q, k, BATCH_Q);
    knn_cuda_predict(train, labels, N, D, query, Q, k,
                     predictions, &transfer_ms, &compute_ms);

    FILE *fout = fopen(output_path, "w");
    if (!fout) {
        fprintf(stderr, "error: cannot write '%s'\n", output_path);
        return 1;
    }
    for (int i = 0; i < Q; i++)
        fprintf(fout, "%.0f\n", predictions[i]);
    fclose(fout);

    fprintf(stderr, "N=%d D=%d Q=%d k=%d transfer=%.3fms compute=%.3fms total=%.3fms\n",
            N, D, Q, k, transfer_ms, compute_ms, transfer_ms + compute_ms);

    free(train);
    free(query);
    free(labels);
    free(predictions);
    return 0;
}
