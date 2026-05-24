CC       := gcc
NVCC     := nvcc
CFLAGS   := -O2 -Wall
CUDA_ARCH := sm_70

SRC_DIR := src
BIN_DIR := bin

.PHONY: all seq omp cuda benchmark validate clean

all: seq omp cuda

seq: $(BIN_DIR)/knn_seq

omp: $(BIN_DIR)/knn_omp

cuda: $(BIN_DIR)/knn_cuda

$(BIN_DIR)/knn_seq: $(SRC_DIR)/knn_seq.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/knn_omp: $(SRC_DIR)/knn_omp.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

$(BIN_DIR)/knn_cuda: $(SRC_DIR)/knn_cuda.cu $(SRC_DIR)/kernels.cuh | $(BIN_DIR)
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

benchmark:
	bash scripts/benchmark.sh

validate:
	bash scripts/validate.sh

clean:
	rm -rf $(BIN_DIR)
	rm -f results/benchmark_results.csv results/errors.log
