#!/usr/bin/env bash
#SBATCH --job-name=knn-omp
#SBATCH --output=knn_omp_%j.out
#SBATCH --error=knn_omp_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --partition=GPU

# ============================================================
#  run_omp.sh — Benchmark KNN OpenMP (job SLURM independiente)
#
#  Derivado de run_seq.sh para una comparativa justa: mismos N/D/Q/k,
#  mismos datos (data_gen.py, misma semilla) y misma metodología de medición
#  (calentamiento + mejor-de-N). Añade únicamente el barrido de hilos.
#
#  Uso:
#    sbatch scripts/run_omp.sh           # completo en SLURM
#    sbatch scripts/run_omp.sh --soft    # rápido en SLURM
#    bash   scripts/run_omp.sh           # local
#
#  Variables de entorno (overridables):
#    KNN_RUNS=3      repeticiones cronometradas por config (se reporta el mínimo)
#    KNN_WARMUP=1    ejecuciones de calentamiento (no cronometradas)
#    KNN_TIMEOUT=1800  timeout por ejecución en segundos
#    KNN_SEED=42     semilla del generador de datos (idéntica a seq/cuda)
#
#  Notas de medición:
#    - El tiempo se mide DENTRO del binario (omp_get_wtime, solo el cómputo);
#      la E/S, la generación de datos y las asignaciones quedan EXCLUIDAS.
#    - OMP_NUM_THREADS se fija por ejecución; OMP_PROC_BIND/OMP_PLACES fijan
#      los hilos a cores físicos para reducir varianza y estabilizar el
#      first-touch (placement NUMA) durante el calentamiento.
#    - cpus-per-task=16 debe ser >= al máximo del barrido de hilos.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=""

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "$SLURM_SUBMIT_DIR/Makefile" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    elif [ -f "$SLURM_SUBMIT_DIR/../Makefile" ]; then
        REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
    fi
fi

if [ -z "$REPO_ROOT" ] && [ -f "$SCRIPT_DIR/../Makefile" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [ -z "$REPO_ROOT" ]; then
    echo "ERROR: no se pudo encontrar la raíz del repositorio con Makefile"
    echo "Lanza el job desde la raíz del proyecto:"
    echo "  sbatch scripts/run_omp.sh --soft"
    exit 1
fi

cd "$REPO_ROOT"

# --- Parse flags -----------------------------------------------
SOFT_MODE=0
for arg in "$@"; do
    case "$arg" in
        --soft) SOFT_MODE=1 ;;
        *) echo "Flag desconocido: $arg"; echo "Uso: $0 [--soft]"; exit 1 ;;
    esac
done

# --- Módulos / entorno -----------------------------------------
# El toolchain (gcc, make) lo provee el entorno del nodo (conda/HPC). Se evita
# 'module purge' para no descargar el entorno AdaptiveCpp/conda cargado por defecto.
echo "=== Entorno del nodo ==="
module list 2>/dev/null || true
echo ""

# --- Verificar herramientas ------------------------------------
echo "=== Verificando herramientas ==="
command -v gcc >/dev/null && echo "  gcc: $(gcc --version | head -1)"
if command -v python3 &>/dev/null; then PY=python3
elif command -v python &>/dev/null; then PY=python
else echo "  ERROR: python no encontrado"; exit 1
fi
$PY --version

# --- Entorno Python --------------------------------------------
# data_gen.py solo requiere numpy. Si el entorno activo (p. ej. conda base)
# ya lo provee, se usa directamente; solo si falta se crea un venv local.
echo "=== Entorno Python ==="
if $PY -c "import numpy" >/dev/null 2>&1; then
    echo "  numpy OK ($($PY -c 'import numpy; print(numpy.__version__)')) — usando entorno actual"
else
    echo "  numpy no encontrado: creando venv local"
    VENV_DIR="./.venv-run-omp"
    [ ! -d "$VENV_DIR" ] && $PY -m venv "$VENV_DIR"
    set +u; source "$VENV_DIR/bin/activate"; set -u
    PY=python
    pip install --quiet numpy 2>&1 || true
    if ! $PY -c "import numpy" >/dev/null 2>&1; then
        echo "  ERROR: numpy no disponible y no se pudo instalar (¿nodo sin internet?)."
        echo "         Activa un entorno con numpy antes de lanzar, o preinstala el venv desde el login."
        exit 1
    fi
fi

# --- Compilar --------------------------------------------------
# Se compila también el secuencial para que 'make validate' compare seq vs omp.
echo "=== Compilando ==="
make clean 2>/dev/null || true
make seq omp
echo "  Binario: $(ls -lh bin/knn_omp)"

# --- Validar ---------------------------------------------------
echo "=== Validando ==="
make validate 2>&1 || echo "  [WARN] validación falló"

# --- Configuración de experimentos -----------------------------
KNN_DATA_DIR="${KNN_DATA_DIR:-./data}"
KNN_RESULTS_DIR="${KNN_RESULTS_DIR:-./results}"
mkdir -p "$KNN_DATA_DIR" "$KNN_RESULTS_DIR"

KNN_RUNS="${KNN_RUNS:-3}"
KNN_WARMUP="${KNN_WARMUP:-1}"
KNN_TIMEOUT="${KNN_TIMEOUT:-1800}"
SEED="${KNN_SEED:-42}"

BINARY="./bin/knn_omp"

# Fijado de hilos a cores físicos (estabilidad + first-touch correcto).
export OMP_PROC_BIND=close
export OMP_PLACES=cores

if [ "$SOFT_MODE" -eq 1 ]; then
    N_VALUES=(1000 5000)
    D_VALUES=(32 64)
    Q_VALUES=(100 500)
    K_VALUES=(5)
    THREADS=(1 2 4)
else
    N_VALUES=(1000 5000 10000 50000 100000)
    D_VALUES=(32 128 512)
    Q_VALUES=(1000 10000)
    K_VALUES=(1 5 15)
    THREADS=(1 2 4 8 16)
fi

RESULTS_FILE="$KNN_RESULTS_DIR/omp_results$([ "$SOFT_MODE" -eq 1 ] && echo '_soft' || echo '').csv"
echo "impl,N,D,Q,k,threads,time_s" > "$RESULTS_FILE"

ERRF="$(mktemp)"
trap 'rm -f "$ERRF"' EXIT

extract_time_s() { sed -n 's/.*time=\([0-9.]*\)s.*/\1/p' "$1"; }

# ensure_dataset N D Q -> define TRAIN_FILE y QUERY_FILE (genera si faltan).
ensure_dataset() {
    local N="$1" D="$2" Q="$3"
    TRAIN_FILE="$KNN_DATA_DIR/train_N${N}_D${D}_s${SEED}.npy"
    QUERY_FILE="$KNN_DATA_DIR/query_N${N}_D${D}_Q${Q}_s${SEED}.npy"
    if [ -f "$TRAIN_FILE" ] && [ -f "$QUERY_FILE" ]; then return 0; fi
    echo "  Generando datos N=$N D=$D Q=$Q ..."
    local tmp; tmp="$(mktemp -u "$KNN_DATA_DIR/.gen_XXXXXX")"
    $PY data_gen.py --n "$N" --d "$D" --q "$Q" --seed "$SEED" --output "$tmp" >/dev/null
    mv -f "${tmp}_train.npy" "$TRAIN_FILE"
    mv -f "${tmp}_query.npy" "$QUERY_FILE"
}

# measure_omp threads train query k out -> define RESULT_TIME (mínimo en s).
measure_omp() {
    local threads="$1" train="$2" query="$3" k="$4" out="$5"
    local best="" t rc w r
    export OMP_NUM_THREADS="$threads"
    for ((w = 0; w < KNN_WARMUP; w++)); do
        timeout "$KNN_TIMEOUT" "$BINARY" \
            --train "$train" --query "$query" --k "$k" --threads "$threads" --output "$out" \
            >/dev/null 2>/dev/null || true
    done
    for ((r = 1; r <= KNN_RUNS; r++)); do
        rc=0
        timeout "$KNN_TIMEOUT" "$BINARY" \
            --train "$train" --query "$query" --k "$k" --threads "$threads" --output "$out" \
            2>"$ERRF" || rc=$?
        if [ "$rc" -ne 0 ]; then echo "    [T=$threads run $r] rc=$rc (timeout/fallo)" >&2; continue; fi
        t=$(extract_time_s "$ERRF")
        if [ -z "$t" ]; then echo "    [T=$threads run $r] parse-fail" >&2; continue; fi
        echo "    [T=$threads run $r] ${t}s" >&2
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    RESULT_TIME="$best"
}

echo "=== Benchmark OMP (SOFT=$SOFT_MODE, RUNS=$KNN_RUNS, WARMUP=$KNN_WARMUP, THREADS=${THREADS[*]}) ==="
echo "    Resultados → $RESULTS_FILE"

# Un timeout/fallo aislado no debe abortar todo el barrido.
set +e

# --- Loop de experimentos --------------------------------------
for N in "${N_VALUES[@]}"; do
for D in "${D_VALUES[@]}"; do
    (( D >= N )) && continue   # data_gen.py requiere d < n
for Q in "${Q_VALUES[@]}"; do
for K in "${K_VALUES[@]}"; do

    ensure_dataset "$N" "$D" "$Q"

    echo "  N=$N D=$D Q=$Q k=$K"
    for T in "${THREADS[@]}"; do
        OUT_FILE="$KNN_RESULTS_DIR/omp_pred_N${N}_D${D}_Q${Q}_k${K}_T${T}.txt"
        measure_omp "$T" "$TRAIN_FILE" "$QUERY_FILE" "$K" "$OUT_FILE"
        echo "    T=$T mejor=${RESULT_TIME:-NA}s"
        echo "omp,$N,$D,$Q,$K,$T,${RESULT_TIME:-NA}" >> "$RESULTS_FILE"
    done

done; done; done; done

echo ""
echo "=== OMP completado ==="
echo "    Resultados: $RESULTS_FILE"
echo "    Salida:     knn_omp_<jobid>.out"
