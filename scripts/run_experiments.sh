#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  run_experiments.sh — KNN Paralelo: OpenMP vs CUDA
#
#  Carga módulos, compila, valida y ejecuta el benchmark
#  completo en el cluster Guane.
#
#  Uso:
#    sbatch scripts/run_experiments.sh          # SLURM
#    bash scripts/run_experiments.sh            # interactive
# ============================================================

# --- SLURM header (opcional) -----------------------------------
#SBATCH --job-name=knn-benchmark
#SBATCH --output=knn_%j.out
#SBATCH --error=knn_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_titan

# --- 1. Cargar módulos -----------------------------------------
echo "=== Cargando módulos ==="
module purge
module load cuda/11.8             # NVCC
module load cmake/3.29.3          # opcional, no necesario
module list

# --- 2. Verificar herramientas ---------------------------------
echo "=== Verificando herramientas ==="
command -v gcc  >/dev/null && echo "  gcc:  $(gcc --version | head -1)"
command -v nvcc >/dev/null && echo "  nvcc: $(nvcc --version | tail -1)"
command -v make >/dev/null && echo "  make: $(make --version | head -1)"
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "  ERROR: python no encontrado"; exit 1
fi
$PY --version

# --- 3. Verificar GPU y detectar CUDA arch ----------------------
echo "=== GPU ==="
ARCH="sm_52"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi 2>/dev/null || echo "  nvidia-smi falló"
    CC_INFO=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
    if [ -n "$CC_INFO" ]; then
        CC_MAJOR=$(echo "$CC_INFO" | head -1 | cut -d. -f1 | tr -d ' ')
        CC_MINOR=$(echo "$CC_INFO" | head -1 | cut -d. -f2 | tr -d ' ')
        if [ -n "$CC_MAJOR" ] && [ -n "$CC_MINOR" ]; then
            ARCH="sm_${CC_MAJOR}${CC_MINOR}"
        fi
    fi
fi
echo "  CUDA arch: $ARCH"

# --- 4. Virtual environment + Python packages --------------------
echo "=== Creando virtual environment ==="
VENV_DIR="./.venv"
if [ ! -d "$VENV_DIR" ]; then
    $PY -m venv "$VENV_DIR"
fi
set +u
source "$VENV_DIR/bin/activate"
set -u
pip install --quiet numpy pandas matplotlib seaborn scikit-learn 2>&1 || \
    pip install --quiet numpy pandas matplotlib seaborn scikit-learn 2>&1
echo "  Python: $(which python3)"

# --- 5. Compilar -----------------------------------------------
echo "=== Compilando ==="
make clean 2>/dev/null || true
make seq
make omp
CUDA_ARCH="$ARCH" make cuda 2>&1 || echo "  [WARN] knn_cuda no compiló"
echo "  Binarios:"
ls -lh bin/

# --- 6. Validar correctitud ------------------------------------
echo "=== Validando ==="
make validate 2>&1 || echo "  [WARN] validación falló"

# --- 7. Ejecutar benchmark -------------------------------------
echo "=== Benchmark ==="
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
export KNN_DATA_DIR=./data
export KNN_RESULTS_DIR=./results

mkdir -p "$KNN_DATA_DIR" "$KNN_RESULTS_DIR"
bash scripts/benchmark.sh 2>&1 || echo "  [WARN] benchmark no completó"

# --- 8. Generar figuras ----------------------------------------
echo "=== Análisis ==="
mkdir -p analysis/figures
$PY analysis/results.py 2>&1 || echo "  [WARN] análisis falló"

echo "=== Listo ==="
echo "  Resultados: ${KNN_RESULTS_DIR}/benchmark_results.csv"
echo "  Figuras:    analysis/figures/"
echo "  Log:        ${KNN_RESULTS_DIR}/errors.log"
echo "  Salida:     knn_<jobid>.out"