#!/usr/bin/env bash
set -euo pipefail

red="\033[0;31m"
green="\033[0;32m"
yellow="\033[0;33m"
reset="\033[0m"

pass()  { echo -e "${green}PASS${reset}: $*"; }
fail()  { echo -e "${red}FAIL${reset}: $*"; }
skip()  { echo -e "${yellow}SKIP${reset}: $*"; }

N=1000
D=10
Q=100
K=5
SEED=42
PREFIX="/tmp/validate_knn_fase2"

echo "--- KNN Validation Suite ---"
echo "N=$N D=$D Q=$Q k=$K seed=$SEED"

echo -n "Generating dataset... "
python data_gen.py --n "$N" --d "$D" --q "$Q" --seed "$SEED" --output "$PREFIX" > /dev/null
echo "done"

echo -n "Running knn_seq... "
./bin/knn_seq --train "${PREFIX}_train.npy" --query "${PREFIX}_query.npy" \
    --k "$K" --output "${PREFIX}_seq.txt" > /dev/null 2>&1
echo "done"

echo -n "Running knn_omp... "
./bin/knn_omp --train "${PREFIX}_train.npy" --query "${PREFIX}_query.npy" \
    --k "$K" --output "${PREFIX}_omp.txt" > /dev/null 2>&1
echo "done"

if diff "${PREFIX}_seq.txt" "${PREFIX}_omp.txt" > /dev/null 2>&1; then
    pass "knn_seq vs knn_omp — predictions identical"
else
    fail "knn_seq vs knn_omp — predictions differ"
    diff "${PREFIX}_seq.txt" "${PREFIX}_omp.txt" | head -20
fi

if [ -x ./bin/knn_cuda ]; then
    echo -n "Running knn_cuda... "
    ./bin/knn_cuda --train "${PREFIX}_train.npy" --query "${PREFIX}_query.npy" \
        --k "$K" --output "${PREFIX}_cuda.txt" > /dev/null 2>&1
    echo "done"
    if diff "${PREFIX}_seq.txt" "${PREFIX}_cuda.txt" > /dev/null 2>&1; then
        pass "knn_seq vs knn_cuda — predictions identical"
    else
        fail "knn_seq vs knn_cuda — predictions differ"
        diff "${PREFIX}_seq.txt" "${PREFIX}_cuda.txt" | head -20
    fi
else
    skip "knn_cuda binary not found"
fi

rm -f "${PREFIX}_"*".txt"
echo "--- Done ---"
