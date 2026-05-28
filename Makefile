CC       := gcc
NVCC     := nvcc
HOST_CXX := /usr/bin/g++
CFLAGS   := -O3 -Wall -march=native -ffast-math
CUDA_ARCH := sm_52

SRC_DIR := src
BIN_DIR := bin

.PHONY: all seq omp cuda benchmark benchmark-soft validate clean

all: seq omp cuda

seq: $(BIN_DIR)/knn_seq

omp: $(BIN_DIR)/knn_omp

cuda: $(BIN_DIR)/knn_cuda

$(BIN_DIR)/knn_seq: $(SRC_DIR)/knn_seq.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/knn_omp: $(SRC_DIR)/knn_omp.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

$(BIN_DIR)/knn_cuda: $(SRC_DIR)/knn_cuda.cu $(SRC_DIR)/kernels.cuh | $(BIN_DIR)
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -ccbin=$(HOST_CXX) -o $@ $<

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

benchmark:
	bash scripts/benchmark.sh

validate:
	bash scripts/validate.sh

benchmark-soft:
	bash scripts/benchmark.sh --soft

clean:
	rm -rf $(BIN_DIR)
	rm -f results/benchmark_results.csv results/benchmark_results_soft.csv results/errors.log
