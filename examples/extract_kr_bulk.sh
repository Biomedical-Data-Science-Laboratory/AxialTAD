#!/bin/bash
# Bulk KR extraction: 4 cell lines × 22 chromosomes = 88 dense .txt matrices.
# Run from inside the `axialtad` conda env (see README for setup).

set -uo pipefail

: "${AXIALTAD_ROOT:?Please set AXIALTAD_ROOT to your working directory}"
ROOT="${AXIALTAD_ROOT}"
HIC_DIR=$ROOT/data_4dn
OUT_DIR=$ROOT/matrix_kr_25kb
LOG_DIR=$ROOT/logs/extract
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/../scripts/extract_kr_matrix.py"
PARALLEL=4
RES=25000

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Build the job list as TSV: cell<TAB>chr_label<TAB>hic_path
JOBS_FILE=$(mktemp)
trap "rm -f $JOBS_FILE" EXIT

for cell_acc in "GM12878:4DNFI1UEG1HD" "HeLa-S3:4DNFICEGAHRC" "IMR-90:4DNFIH7TH4MF" "K562:4DNFITUOMFUQ"; do
    cell="${cell_acc%%:*}"
    acc="${cell_acc##*:}"
    hic="$HIC_DIR/$acc.hic"
    if [[ ! -s "$hic" ]]; then
        echo "MISSING-HIC $cell : $hic"
        continue
    fi
    for n in {1..22}; do
        printf "%s\t%s\t%s\n" "$cell" "chr$n" "$hic" >> "$JOBS_FILE"
    done
done

n_jobs=$(wc -l < "$JOBS_FILE")
echo "Total jobs: $n_jobs"

export OUT_DIR LOG_DIR SCRIPT RES

worker() {
    local cell="$1" chr_label="$2" hic="$3"
    local chr_in_hic="${chr_label#chr}"
    local out="$OUT_DIR/${cell}_${chr_label}_25kb.txt"
    local log="$LOG_DIR/${cell}_${chr_label}.log"
    if [[ -s "$out" ]]; then
        echo "SKIP ${cell}_${chr_label}"
        return 0
    fi
    if python "$SCRIPT" "$hic" "$chr_in_hic" "$out" --resolution "$RES" >"$log" 2>&1; then
        local sz
        sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [[ "$sz" -lt 1000 ]]; then
            echo "FAIL ${cell}_${chr_label} (small: $sz)"
            return 1
        fi
        echo "OK   ${cell}_${chr_label} ($sz)"
    else
        echo "FAIL ${cell}_${chr_label} (see $log)"
        return 1
    fi
}
export -f worker

awk -F'\t' '{print $0}' "$JOBS_FILE" \
  | xargs -d'\n' -n1 -P"$PARALLEL" -I{} bash -c '
        IFS=$'"'"'\t'"'"' read -r c lbl h <<< "$1"
        worker "$c" "$lbl" "$h"
    ' _ {}

echo ""
echo "=== summary ==="
n_done=$(ls "$OUT_DIR"/*.txt 2>/dev/null | wc -l)
echo "txt files produced: $n_done / 88"
