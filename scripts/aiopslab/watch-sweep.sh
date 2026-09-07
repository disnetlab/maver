#!/usr/bin/env bash
# Watch one AIOpsLab sweep: the grade of every problem so far, and the sweep's own log.
#
#   scripts/aiopslab/watch-sweep.sh /var/lib/aiopslab/results/sweeps/<sweep-id> [seconds]
#
# It observes and nothing more. It does not kill a slow problem — the sweep's own
# --problem-timeout is the policy boundary — and it does not restart a dead one, because
# resuming is `aiopslab-driver sweep` with the same --sweep-dir, which skips every
# problem AIOpsLab has already graded.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'usage: %s <sweep-dir> [interval-seconds]\n' "$0" >&2
  exit 2
fi

sweep_dir="$1"
interval_seconds="${2:-60}"
status_file="${sweep_dir}/status.tsv"
log_file="${sweep_dir}/sweep.log"
metadata_file="${sweep_dir}/metadata.env"
log_lines="${AIOPSLAB_WATCH_LOG_LINES:-12}"
row_lines="${AIOPSLAB_WATCH_ROW_LINES:-10}"

if [[ ! -d "${sweep_dir}" ]]; then
  printf 'no such sweep directory: %s\n' "${sweep_dir}" >&2
  exit 2
fi

tally() {
  # By header name rather than by position, so adding a column does not silently move
  # the count to another field.
  awk -F '\t' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "grade") column = i; next }
    column { counts[$column]++; total++ }
    END {
      printf "%d graded", total
      for (grade in counts) printf "  %s=%d", grade, counts[grade]
      printf "\n"
    }' "${status_file}"
}

rows() {
  local rendered
  if command -v column >/dev/null 2>&1; then
    rendered="$(column -ts $'\t' "${status_file}")"
  else
    rendered="$(cat "${status_file}")"
  fi
  # The header stays on screen once a sweep has more rows than fit.
  printf '%s\n' "${rendered}" | head -n 1
  printf '%s\n' "${rendered}" | tail -n +2 | tail -n "${row_lines}"
}

while true; do
  printf '\n===== %s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${sweep_dir}"
  if [[ -f "${status_file}" ]]; then
    tally
    rows
  else
    printf 'no problem has finished yet\n'
  fi
  if [[ -f "${log_file}" ]]; then
    printf -- '----- sweep.log\n'
    tail -n "${log_lines}" "${log_file}"
  fi

  if [[ -f "${metadata_file}" ]] && grep -q '^completed_utc=' "${metadata_file}"; then
    printf 'sweep complete\n'
    exit 0
  fi
  sleep "${interval_seconds}"
done
