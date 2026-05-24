#!/usr/bin/env bash
set -o pipefail

SOFT_MODE=0
for arg in "$@"; do
    case "$arg" in
        --soft) SOFT_MODE=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: $0 [--soft]"; exit 1 ;;
    esac
done

DATA_DIR="${KNN_DATA_DIR:-./data}"
RES_DIR="${KNN_RESULTS_DIR:-./results}"
mkdir -p "$DATA_DIR" "$RES_DIR"

if [ "$SOFT_MODE" -eq 1 ]; then
    CSV="${RES_DIR}/benchmark_results_soft.csv"
    N_VALUES=(1000 5000 10000)
    D_VALUES=(2 10 50 100)
    K_VALUES=(1 3 10)
    THREADS_OMP=(1 2 4 8)
    RUNS=3
    echo "[soft mode] quick benchmark for visualization"
else
    CSV="${RES_DIR}/benchmark_results.csv"
    N_VALUES=(1000 5000 10000 50000 100000 500000)
    D_VALUES=(2 10 50 100 500 1000)
    K_VALUES=(1 3 5 10)
    THREADS_OMP=(1 2 4 8 16)
    RUNS=5
fi

ERRLOG="${RES_DIR}/errors.log"
> "$ERRLOG"

echo "n,d,k,impl,threads,run,time_ms,transfer_ms,compute_ms,speedup_vs_seq" > "$CSV"
TMPDIR="/tmp/knn_bench"
mkdir -p "$TMPDIR"

HAVE_CUDA=0
[ -x ./bin/knn_cuda ] && HAVE_CUDA=1

log_err() { echo "[$(date '+%H:%M:%S')] $*" >> "$ERRLOG"; }

extract_time_s() {
    sed -n 's/.*time=\([0-9.]*\)s.*/\1/p' "$1"
}
extract_transfer_ms() {
    sed -n 's/.*transfer=\([0-9.]*\)ms.*/\1/p' "$1"
}
extract_compute_ms() {
    sed -n 's/.*compute=\([0-9.]*\)ms.*/\1/p' "$1"
}

# Timeout máximo por ejecución individual (segundos). Ajustar según hardware.
SEQ_TIMEOUT=600   # 10 min para secuencial (configs grandes pueden ser lentas)
OMP_TIMEOUT=120   # 2 min para OpenMP
CUDA_TIMEOUT=60   # 1 min para CUDA

run_seq() {
    local out="$1" err="$2" train="$3" query="$4" k="$5"
    timeout "$SEQ_TIMEOUT" ./bin/knn_seq --train "$train" --query "$query" --k "$k" --output "$out" 2>"$err"
}

run_omp() {
    local threads="$1" out="$2" err="$3" train="$4" query="$5" k="$6"
    timeout "$OMP_TIMEOUT" ./bin/knn_omp --train "$train" --query "$query" --k "$k" --threads "$threads" --output "$out" 2>"$err"
}

run_cuda() {
    local out="$1" err="$2" train="$3" query="$4" k="$5"
    timeout "$CUDA_TIMEOUT" ./bin/knn_cuda --train "$train" --query "$query" --k "$k" --output "$out" 2>"$err"
}

total_configs=0
for N in "${N_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        (( N * D > 500000000 )) && continue
        for K in "${K_VALUES[@]}"; do
            total_configs=$((total_configs + 1))
        done
    done
done
runs_per_config=$((RUNS * (1 + ${#THREADS_OMP[@]} + HAVE_CUDA)))
total_runs=$((total_configs * runs_per_config))

completed=0

progress() {
    local pct=$((completed * 100 / total_runs))
    local bar=""
    local n=$((pct / 5))
    for ((i = 0; i < n; i++)); do bar="${bar}#"; done
    for ((i = n; i < 20; i++)); do bar="${bar}-"; done
    printf "\r[%s] %3d%%  (%d / %d)" "$bar" "$pct" "$completed" "$total_runs"
}

echo "Benchmark: $total_configs configs x $RUNS runs x ~$((runs_per_config / RUNS)) impl = ~$total_runs executions"
echo ""

for N in "${N_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        (( N * D > 500000000 )) && continue
        Q=$(( N / 5 > 100 ? N / 5 : 100 ))
        # Guard CUDA: d_dist = Q*N*4 bytes. Saltar si supera 6 GB de VRAM.
        dist_mb=$(( Q * N * 4 / 1024 / 1024 ))
        if (( HAVE_CUDA && dist_mb > 6000 )); then
            echo "  [skip-cuda-oom] N=$N D=$D: d_dist=${dist_mb}MB > 6000MB"
            log_err "skip-cuda-oom N=$N D=$D dist_mb=$dist_mb"
        fi

        echo "--- N=$N D=$D Q=$Q ---"

        for K in "${K_VALUES[@]}"; do
            prefix="${TMPDIR}/bench_n${N}_d${D}_k${K}"
            train="${prefix}_train.npy"
            query="${prefix}_query.npy"

            python3 data_gen.py --n "$N" --d "$D" --q "$Q" --seed 42 \
                --output "$prefix" >/dev/null 2>>"$ERRLOG" || {
                log_err "data_gen failed N=$N D=$D K=$K"
                continue
            }

            for run in $(seq 1 $RUNS); do
                errf="${TMPDIR}/err_${N}_${D}_${K}_${run}.txt"
                outf="${TMPDIR}/out_${N}_${D}_${K}_${run}.txt"

                # --- Sequential ---
                run_seq "$outf" "$errf" "$train" "$query" "$K"
                seq_rc=$?
                if [ $seq_rc -eq 124 ]; then
                    log_err "TIMEOUT seq N=$N D=$D K=$K run=$run (>${SEQ_TIMEOUT}s)"
                    tseq=""
                elif [ $seq_rc -eq 0 ]; then
                    ts=$(extract_time_s "$errf")
                    if [ -n "$ts" ]; then
                        tseq=$(awk "BEGIN {printf \"%.6f\", $ts * 1000}")
                        echo "$N,$D,$K,seq,1,$run,$tseq,0,$tseq,1.0" >> "$CSV"
                    else
                        log_err "parse seq N=$N D=$D K=$K run=$run"
                        tseq=""
                    fi
                else
                    log_err "seq failed N=$N D=$D K=$K run=$run (rc=$seq_rc)"
                    tseq=""
                fi
                completed=$((completed + 1))

                # --- OpenMP ---
                for T in "${THREADS_OMP[@]}"; do
                    run_omp "$T" "$outf" "$errf" "$train" "$query" "$K"
                    omp_rc=$?
                    if [ $omp_rc -eq 124 ]; then
                        log_err "TIMEOUT omp N=$N D=$D K=$K T=$T run=$run (>${OMP_TIMEOUT}s)"
                    elif [ $omp_rc -eq 0 ]; then
                        tomp=$(extract_time_s "$errf")
                        if [ -n "$tomp" ] && [ -n "$tseq" ]; then
                            tomp_ms=$(awk "BEGIN {printf \"%.6f\", $tomp * 1000}")
                            sp=$(awk "BEGIN {printf \"%.6f\", $tseq / $tomp_ms}")
                            echo "$N,$D,$K,omp,$T,$run,$tomp_ms,0,$tomp_ms,$sp" >> "$CSV"
                        else
                            log_err "parse omp N=$N D=$D K=$K T=$T run=$run"
                        fi
                    else
                        log_err "omp failed N=$N D=$D K=$K T=$T run=$run (rc=$omp_rc)"
                    fi
                    completed=$((completed + 1))
                done

                # --- CUDA ---
                if [ "$HAVE_CUDA" -eq 1 ] && (( dist_mb <= 6000 )); then
                    run_cuda "$outf" "$errf" "$train" "$query" "$K"
                    cuda_rc=$?
                    if [ $cuda_rc -eq 124 ]; then
                        log_err "TIMEOUT cuda N=$N D=$D K=$K run=$run (>${CUDA_TIMEOUT}s)"
                    elif [ $cuda_rc -eq 0 ]; then
                        tfr=$(extract_transfer_ms "$errf")
                        tcm=$(extract_compute_ms "$errf")
                        if [ -n "$tfr" ] && [ -n "$tcm" ] && [ -n "$tseq" ]; then
                            ttotal=$(awk "BEGIN {printf \"%.6f\", $tfr + $tcm}")
                            sp=$(awk "BEGIN {printf \"%.6f\", $tseq / ($tfr + $tcm)}")
                            echo "$N,$D,$K,cuda,1,$run,$ttotal,$tfr,$tcm,$sp" >> "$CSV"
                        else
                            log_err "parse cuda N=$N D=$D K=$K run=$run"
                        fi
                    else
                        log_err "cuda failed N=$N D=$D K=$K run=$run (rc=$cuda_rc)"
                    fi
                    completed=$((completed + 1))
                fi
                progress
            done
            rm -f "$train" "$query" "${prefix}"_*.npy "${prefix}"_*.csv
        done
    done
done

echo ""
echo "Done.  CSV: $CSV"
echo "Errors: $ERRLOG"
wc -l < "$CSV" | xargs echo "Rows:"
if [ -s "$ERRLOG" ]; then
    echo "Warnings/errors: $(wc -l < "$ERRLOG")"
fi
