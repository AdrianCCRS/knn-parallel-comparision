#!/usr/bin/env bash
#SBATCH --job-name=knn-cuda
#SBATCH --output=knn_cuda_%j.out
#SBATCH --error=knn_cuda_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU

# ============================================================
#  run_cuda.sh — Benchmark KNN CUDA (job SLURM independiente)
#
#  Derivado de run_seq.sh para una comparativa justa: mismos N/D/Q/k,
#  mismos datos (data_gen.py, misma semilla) y misma metodología de medición
#  (calentamiento + mejor-de-N). Añade únicamente la gestión de la GPU.
#
#  Uso:
#    sbatch scripts/run_cuda.sh           # completo en SLURM
#    sbatch scripts/run_cuda.sh --soft    # rápido en SLURM
#    bash   scripts/run_cuda.sh           # local
#
#  Variables de entorno (overridables):
#    KNN_RUNS=3      repeticiones cronometradas por config (se reporta el mínimo)
#    KNN_WARMUP=1    ejecuciones de calentamiento (no cronometradas)
#    KNN_TIMEOUT=1800  timeout por ejecución en segundos
#    KNN_SEED=42     semilla del generador de datos (idéntica a seq/omp)
#    KNN_DEVICE=0    índice de GPU a usar
#
#  Notas de medición:
#    - El tiempo se mide DENTRO del binario con cudaEvents: transfer_ms (H<->D)
#      y compute_ms (kernels). time_s = (transfer+compute)/1000 para que sea
#      comparable con seq/omp; transfer_s y compute_s se reportan por separado.
#    - El calentamiento es clave en GPU: amortiza la inicialización del contexto
#      CUDA y estabiliza los relojes antes de cronometrar.
#    - La E/S a disco y la asignación de memoria host quedan EXCLUIDAS.
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
    echo "  sbatch scripts/run_cuda.sh --soft"
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
# CUDA 12.0 (nvcc), gcc y make los provee el entorno del nodo (conda/HPC):
# NO se requiere 'module load' y se evita 'module purge' para no descargar
# el entorno AdaptiveCpp/conda cargado por defecto.
echo "=== Entorno del nodo ==="
module list 2>/dev/null || true
echo ""

# --- Verificar herramientas ------------------------------------
echo "=== Verificando herramientas ==="
command -v gcc  >/dev/null && echo "  gcc:  $(gcc --version | head -1)"
command -v nvcc >/dev/null && echo "  nvcc: $(nvcc --version | tail -1)"
if command -v python3 &>/dev/null; then PY=python3
elif command -v python &>/dev/null; then PY=python
else echo "  ERROR: python no encontrado"; exit 1
fi
$PY --version

# --- GPU: selección y detección de arquitectura ----------------
KNN_DEVICE="${KNN_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$KNN_DEVICE}"

echo "=== GPU ==="
ARCH="sm_80"   # A100 = compute capability 8.0; autodetección abajo lo confirma/ajusta
GPU_MEM_MB=0
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
    MEM_INFO=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
    [ -n "$MEM_INFO" ] && GPU_MEM_MB="$MEM_INFO"
fi
echo "  CUDA arch: $ARCH   VRAM: ${GPU_MEM_MB}MB   device: $KNN_DEVICE"

# --- Entorno Python --------------------------------------------
# data_gen.py solo requiere numpy. Si el entorno activo (p. ej. conda base)
# ya lo provee, se usa directamente; solo si falta se crea un venv local.
echo "=== Entorno Python ==="
if $PY -c "import numpy" >/dev/null 2>&1; then
    echo "  numpy OK ($($PY -c 'import numpy; print(numpy.__version__)')) — usando entorno actual"
else
    echo "  numpy no encontrado: creando venv local"
    VENV_DIR="./.venv-run-cuda"
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
# Se compila también el secuencial para que 'make validate' compare seq vs cuda.
echo "=== Compilando ==="
make clean 2>/dev/null || true
make seq
CUDA_ARCH="$ARCH" make cuda
echo "  Binario: $(ls -lh bin/knn_cuda)"

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

BINARY="./bin/knn_cuda"

# BATCH_Q en kernels.cuh (memoria de d_dist = BATCH_Q * N * 4 bytes).
BATCH_Q=512

if [ "$SOFT_MODE" -eq 1 ]; then
    N_VALUES=(1000 5000)
    D_VALUES=(32 64)
    Q_VALUES=(100 500)
    K_VALUES=(5)
else
    N_VALUES=(1000 5000 10000 50000 100000)
    D_VALUES=(32 128 512)
    Q_VALUES=(1000 10000)
    K_VALUES=(1 5 15)
fi

RESULTS_FILE="$KNN_RESULTS_DIR/cuda_results$([ "$SOFT_MODE" -eq 1 ] && echo '_soft' || echo '').csv"
echo "impl,N,D,Q,k,time_s,transfer_s,compute_s" > "$RESULTS_FILE"

ERRF="$(mktemp)"
trap 'rm -f "$ERRF"' EXIT

extract_total_ms()    { sed -n 's/.*total=\([0-9.]*\)ms.*/\1/p' "$1"; }
extract_transfer_ms() { sed -n 's/.*transfer=\([0-9.]*\)ms.*/\1/p' "$1"; }
extract_compute_ms()  { sed -n 's/.*compute=\([0-9.]*\)ms.*/\1/p' "$1"; }

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

# gpu_fits N D -> 0 si la huella estimada cabe en VRAM (margen 85%), 1 si no.
# Huella ~ d_train(N*D) + d_labels(N) + d_query(BATCH_Q*D)
#          + d_dist(BATCH_Q*N) + d_pred(BATCH_Q), en floats (4 bytes).
gpu_fits() {
    local N="$1" D="$2"
    [ "$GPU_MEM_MB" -le 0 ] && return 0   # sin dato de VRAM: no filtrar
    local need_mb
    need_mb=$(awk "BEGIN{ b=($N*$D + $N + $BATCH_Q*$D + $BATCH_Q*$N + $BATCH_Q)*4; printf \"%d\", b/1048576 }")
    local budget=$(( GPU_MEM_MB * 85 / 100 ))
    if [ "$need_mb" -gt "$budget" ]; then
        echo "  [skip-oom] N=$N D=$D: ~${need_mb}MB > ${budget}MB (85% de ${GPU_MEM_MB}MB)"
        return 1
    fi
    return 0
}

# measure_cuda train query k out -> define RESULT_TOTAL_S/TRANSFER_S/COMPUTE_S.
# Selecciona el mejor por total y conserva su desglose transfer/compute.
measure_cuda() {
    local train="$1" query="$2" k="$3" out="$4"
    local best_total="" best_tr="" best_cm=""
    local total tr cm rc w r
    for ((w = 0; w < KNN_WARMUP; w++)); do
        timeout "$KNN_TIMEOUT" "$BINARY" \
            --train "$train" --query "$query" --k "$k" --device "$KNN_DEVICE" --output "$out" \
            >/dev/null 2>/dev/null || true
    done
    for ((r = 1; r <= KNN_RUNS; r++)); do
        rc=0
        timeout "$KNN_TIMEOUT" "$BINARY" \
            --train "$train" --query "$query" --k "$k" --device "$KNN_DEVICE" --output "$out" \
            2>"$ERRF" || rc=$?
        if [ "$rc" -ne 0 ]; then echo "    [run $r] rc=$rc (timeout/fallo)" >&2; continue; fi
        total=$(extract_total_ms "$ERRF")
        tr=$(extract_transfer_ms "$ERRF")
        cm=$(extract_compute_ms "$ERRF")
        if [ -z "$total" ] || [ -z "$tr" ] || [ -z "$cm" ]; then
            echo "    [run $r] parse-fail" >&2; continue
        fi
        echo "    [run $r] total=${total}ms (transfer=${tr} compute=${cm})" >&2
        if [ -z "$best_total" ] || awk "BEGIN{exit !($total < $best_total)}"; then
            best_total="$total"; best_tr="$tr"; best_cm="$cm"
        fi
    done
    if [ -z "$best_total" ]; then
        RESULT_TOTAL_S=""; RESULT_TRANSFER_S=""; RESULT_COMPUTE_S=""
    else
        RESULT_TOTAL_S=$(awk "BEGIN{printf \"%.6f\", $best_total/1000}")
        RESULT_TRANSFER_S=$(awk "BEGIN{printf \"%.6f\", $best_tr/1000}")
        RESULT_COMPUTE_S=$(awk "BEGIN{printf \"%.6f\", $best_cm/1000}")
    fi
}

echo "=== Benchmark CUDA (SOFT=$SOFT_MODE, RUNS=$KNN_RUNS, WARMUP=$KNN_WARMUP) ==="
echo "    Resultados → $RESULTS_FILE"

# Un timeout/fallo aislado no debe abortar todo el barrido.
set +e

# --- Loop de experimentos --------------------------------------
for N in "${N_VALUES[@]}"; do
for D in "${D_VALUES[@]}"; do
    (( D >= N )) && continue   # data_gen.py requiere d < n
    if ! gpu_fits "$N" "$D"; then
        for Q in "${Q_VALUES[@]}"; do for K in "${K_VALUES[@]}"; do
            echo "cuda,$N,$D,$Q,$K,NA,NA,NA" >> "$RESULTS_FILE"
        done; done
        continue
    fi
for Q in "${Q_VALUES[@]}"; do
for K in "${K_VALUES[@]}"; do

    ensure_dataset "$N" "$D" "$Q"
    OUT_FILE="$KNN_RESULTS_DIR/cuda_pred_N${N}_D${D}_Q${Q}_k${K}.txt"

    echo "  N=$N D=$D Q=$Q k=$K"
    measure_cuda "$TRAIN_FILE" "$QUERY_FILE" "$K" "$OUT_FILE"
    echo "    mejor total=${RESULT_TOTAL_S:-NA}s (transfer=${RESULT_TRANSFER_S:-NA} compute=${RESULT_COMPUTE_S:-NA})"
    echo "cuda,$N,$D,$Q,$K,${RESULT_TOTAL_S:-NA},${RESULT_TRANSFER_S:-NA},${RESULT_COMPUTE_S:-NA}" >> "$RESULTS_FILE"

done; done; done; done

echo ""
echo "=== CUDA completado ==="
echo "    Resultados: $RESULTS_FILE"
echo "    Salida:     knn_cuda_<jobid>.out"
