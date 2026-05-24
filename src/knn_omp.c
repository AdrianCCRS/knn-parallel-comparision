#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

typedef struct {
    float dist;
    float label;
} Neighbor;

static void heap_push(Neighbor *heap, int *n, float dist, float label)
{
    int i = (*n)++;
    heap[i].dist = dist;
    heap[i].label = label;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p].dist >= heap[i].dist)
            break;
        Neighbor t = heap[p];
        heap[p] = heap[i];
        heap[i] = t;
        i = p;
    }
}

static void heap_pop(Neighbor *heap, int *n)
{
    heap[0] = heap[--(*n)];
    int i = 0;
    for (;;) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < *n && heap[left].dist > heap[largest].dist)
            largest = left;
        if (right < *n && heap[right].dist > heap[largest].dist)
            largest = right;
        if (largest == i)
            break;
        Neighbor t = heap[i];
        heap[i] = heap[largest];
        heap[largest] = t;
        i = largest;
    }
}

#define MAX_CLASSES 256

/* O(K + C) con histograma en lugar de O(K²). Asume labels enteras en [0, MAX_CLASSES). */
static float majority_vote(Neighbor *heap, int n)
{
    if (n <= 0)
        return 0.0f;
    int counts[MAX_CLASSES] = {0};
    for (int i = 0; i < n; i++) {
        int cls = (int)heap[i].label;
        if (cls >= 0 && cls < MAX_CLASSES)
            counts[cls]++;
    }
    float best_label = heap[0].label;
    int best_cnt = 0;
    for (int c = 0; c < MAX_CLASSES; c++) {
        if (counts[c] > best_cnt) {
            best_cnt = counts[c];
            best_label = (float)c;
        }
    }
    return best_label;
}

/* Tamaño de tile en dimensión D: 64 floats = 256 bytes = 4 líneas de caché.
 * Mejora localidad en L1/L2 cuando D es grande y hay muchos hilos competiendo. */
#define D_TILE 64

void knn_predict(const float *train, const float *labels, int N, int D,
                 const float *query, int Q, int k, float *predictions)
{
    #pragma omp parallel for schedule(dynamic, 32)
    for (int q = 0; q < Q; q++) {
        Neighbor heap[1024];
        int n = 0;
        const float *qrow = query + (size_t)q * D;

        for (int i = 0; i < N; i++) {
            float dist = 0.0f;
            const float *trow = train + (size_t)i * D;

            /* Tiling sobre D para reducir presión en caché compartida */
            for (int d_base = 0; d_base < D; d_base += D_TILE) {
                int d_end = d_base + D_TILE;
                if (d_end > D) d_end = D;
                for (int d = d_base; d < d_end; d++) {
                    float diff = trow[d] - qrow[d];
                    dist += diff * diff;
                }
            }

            if (n < k) {
                heap_push(heap, &n, dist, labels[i]);
            } else if (dist < heap[0].dist) {
                heap_pop(heap, &n);
                heap_push(heap, &n, dist, labels[i]);
            }
        }
        predictions[q] = majority_vote(heap, n);
    }
}

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
    int threads = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0 && i + 1 < argc)
            train_path = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_path = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            threads = atoi(argv[++i]);
    }
    if (!train_path || !query_path || !output_path) {
        fprintf(stderr, "usage: %s --train <file.npy> --query <file.npy>"
                        " --k <int> --output <file.txt> [--threads <int>]\n",
                argv[0]);
        return 1;
    }
    if (k <= 0 || k > 1024) {
        fprintf(stderr, "error: k=%d out of range [1, 1024] (heap buffer limit)\n", k);
        return 1;
    }

    if (threads > 0)
        omp_set_num_threads(threads);

    int N, D, Q, Dq;
    float *labels = NULL;
    float *train = load_npy(train_path, &N, &D, &labels);
    float *query = load_npy(query_path, &Q, &Dq, NULL);
    if (D != Dq) {
        fprintf(stderr, "error: dimension mismatch train=%d query=%d\n", D, Dq);
        return 1;
    }

    float *predictions = (float *)malloc((size_t)Q * sizeof(float));

    double t0 = omp_get_wtime();
    knn_predict(train, labels, N, D, query, Q, k, predictions);
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    FILE *fout = fopen(output_path, "w");
    if (!fout) {
        fprintf(stderr, "error: cannot write '%s'\n", output_path);
        return 1;
    }
    for (int i = 0; i < Q; i++)
        fprintf(fout, "%.0f\n", predictions[i]);
    fclose(fout);

    int actual_threads = omp_get_max_threads();
    fprintf(stderr, "N=%d D=%d Q=%d k=%d threads=%d time=%.6fs per_query=%.6fs\n",
            N, D, Q, k, actual_threads, elapsed, elapsed / Q);

    free(train);
    free(query);
    free(labels);
    free(predictions);
    return 0;
}
