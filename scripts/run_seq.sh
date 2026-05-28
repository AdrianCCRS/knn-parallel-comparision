#!/usr/bin/env bash
#SBATCH --job-name=knn-seq
#SBATCH --output=knn_seq_%j.out
#SBATCH --error=knn_seq_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --partition=GPU

# ============================================================
#  run_seq.sh — Benchmark KNN Secuencial (job SLURM independiente)
#
#  Uso:
#    sbatch scripts/run_seq.sh           # completo en SLURM
#    sbatch scripts/run_seq.sh --soft    # rápido en SLURM
#    bash   scripts/run_seq.sh           # local
#
#  Variables de entorno (overridables):
#    KNN_RUNS=3      repeticiones cronometradas por config (se reporta el mínimo)
#    KNN_WARMUP=1    ejecuciones de calentamiento (no cronometradas)
#    KNN_TIMEOUT=1800  timeout por ejecución en segundos
#    KNN_SEED=42     semilla del generador de datos (idéntica para seq/omp/cuda)
#
#  Notas de medición:
#    - El tiempo se mide DENTRO del binario (solo el cómputo KNN); la E/S,
#      la generación de datos y las asignaciones de memoria quedan EXCLUIDAS.
#    - Se usa data_gen.py (numpy) para que seq/omp/cuda corran sobre datos
#      byte-idénticos (misma semilla) y la generación no sea el cuello de botella.
#    - Calentamiento + mejor-de-N elimina el sesgo de caché fría / page-faults.
#    - El proceso se fija a un core (taskset) para reducir la varianza por migración.
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
    echo "  sbatch scripts/run_seq.sh --soft"
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
    VENV_DIR="./.venv-run-seq"
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
echo "=== Compilando ==="
make clean 2>/dev/null || true
make seq
echo "  Binario: $(ls -lh bin/knn_seq)"

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

BINARY="./bin/knn_seq"

# Fijar el proceso a un core para una medición estable (si taskset existe).
TASKSET=""
if command -v taskset >/dev/null 2>&1; then TASKSET="taskset -c 0"; fi

if [ "$SOFT_MODE" -eq 1 ]; then
    N_VALUES=(1000 5000)
    D_VALUES=(32 64)
    Q_VALUES=(100 500)
    K_VALUES=(5)
else
    N_VALUES=(10000 100000 1000000)
    D_VALUES=(32 128 512)
    Q_VALUES=(1000 10000)
    K_VALUES=(1 5 15)
fi

RESULTS_FILE="$KNN_RESULTS_DIR/seq_results$([ "$SOFT_MODE" -eq 1 ] && echo '_soft' || echo '').csv"
echo "impl,N,D,Q,k,time_s" > "$RESULTS_FILE"

ERRF="$(mktemp)"
trap 'rm -f "$ERRF"' EXIT

extract_time_s() { sed -n 's/.*time=\([0-9.]*\)s.*/\1/p' "$1"; }

# ensure_dataset N D Q -> define TRAIN_FILE y QUERY_FILE (genera si faltan).
# train depende de (N,D,seed); query depende de (N,D,Q,seed). Generación atómica.
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

# measure_seq train query k out -> define RESULT_TIME (mínimo en s, o vacío).
measure_seq() {
    local train="$1" query="$2" k="$3" out="$4"
    local best="" t rc w r
    for ((w = 0; w < KNN_WARMUP; w++)); do
        timeout "$KNN_TIMEOUT" $TASKSET "$BINARY" \
            --train "$train" --query "$query" --k "$k" --output "$out" \
            >/dev/null 2>/dev/null || true
    done
    for ((r = 1; r <= KNN_RUNS; r++)); do
        rc=0
        timeout "$KNN_TIMEOUT" $TASKSET "$BINARY" \
            --train "$train" --query "$query" --k "$k" --output "$out" \
            2>"$ERRF" || rc=$?
        if [ "$rc" -ne 0 ]; then echo "    [run $r] rc=$rc (timeout/fallo)" >&2; continue; fi
        t=$(extract_time_s "$ERRF")
        if [ -z "$t" ]; then echo "    [run $r] parse-fail" >&2; continue; fi
        echo "    [run $r] ${t}s" >&2
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    RESULT_TIME="$best"
}

echo "=== Benchmark SEQ (SOFT=$SOFT_MODE, RUNS=$KNN_RUNS, WARMUP=$KNN_WARMUP) ==="
echo "    Resultados → $RESULTS_FILE"

# Desactivar 'exit on error' durante el barrido: un timeout/fallo aislado no
# debe abortar todo el benchmark (se registra NA y se continúa).
set +e

# --- Loop de experimentos --------------------------------------
for N in "${N_VALUES[@]}"; do
for D in "${D_VALUES[@]}"; do
    (( D >= N )) && continue   # data_gen.py requiere d < n
for Q in "${Q_VALUES[@]}"; do
for K in "${K_VALUES[@]}"; do

    ensure_dataset "$N" "$D" "$Q"
    OUT_FILE="$KNN_RESULTS_DIR/seq_pred_N${N}_D${D}_Q${Q}_k${K}.txt"

    echo "  N=$N D=$D Q=$Q k=$K"
    measure_seq "$TRAIN_FILE" "$QUERY_FILE" "$K" "$OUT_FILE"
    echo "    mejor=${RESULT_TIME:-NA}s"
    echo "seq,$N,$D,$Q,$K,${RESULT_TIME:-NA}" >> "$RESULTS_FILE"

done; done; done; done

echo ""
echo "=== SEQ completado ==="
echo "    Resultados: $RESULTS_FILE"
echo "    Salida:     knn_seq_<jobid>.out"
