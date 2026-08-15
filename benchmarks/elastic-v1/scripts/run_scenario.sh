#!/usr/bin/env bash
# Run one scenario under stock or v1.
# Usage:
#   VARIANT=v1 SCENARIO=s2 ./scripts/run_scenario.sh
#   VARIANT=stock SCENARIO=s1 ./scripts/run_scenario.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

VARIANT="${VARIANT:-v1}"       # stock | v1
SCENARIO="${SCENARIO:-s2}"     # s1 | s2 | s3 | noisy
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_BASE}/${VARIANT}/${SCENARIO}/${RUN_ID}"
mkdir -p "${OUT_DIR}"
export OUT_DIR

need_bin "${SM_BURN}"
need_bin "${LIBVGPU}"
command -v nvidia-smi >/dev/null || die "nvidia-smi required"
[[ "${VARIANT}" == "stock" || "${VARIANT}" == "v1" ]] || die "VARIANT must be stock|v1"
[[ -x "${FEEDBACK_LITE}" ]] || [[ "${VARIANT}" == "stock" ]] || die "feedback-lite missing"

# Isolate this run's hook tree.
HOOK_BASE="${OUT_DIR}/hook"
mkdir -p "${HOOK_BASE}/containers"
export HOOK_BASE

echo "== elastic-v1 harness =="
echo " variant=${VARIANT} scenario=${SCENARIO} floor=${FLOOR_SM} duration=${DURATION_S}s"
echo " out=${OUT_DIR}"
echo " libvgpu=${LIBVGPU}"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill "${p}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

NVPID="$(sample_nvsmi)"
PIDS+=("${NVPID}")

FB_PID=""
if [[ "${VARIANT}" == "v1" ]]; then
  need_bin "${FEEDBACK_LITE}"
  FB_PID="$(start_feedback_lite)"
  PIDS+=("${FB_PID}")
  # Let feedback-lite start before workers create caches.
  sleep 0.3
fi

SIGNAL_FILE="${OUT_DIR}/wake_owner"
rm -f "${SIGNAL_FILE}"

case "${SCENARIO}" in
  s1)
    # Both busy: floor guarantees, little/no headroom benefit expected.
    PIDS+=("$(run_worker A busy)")
    PIDS+=("$(run_worker B busy)")
    ;;
  s2)
    # A idle / B busy: lending benefit on v1.
    PIDS+=("$(run_worker A idle)")
    PIDS+=("$(run_worker B busy)")
    ;;
  s3)
    # A stays idle until SIGNAL_FILE only (do not use --idle-sec auto-wake:
    # that raced the reclaim clock and made latency look wrongly ~0).
    : "${OWNER_IDLE_SEC:=8}"
    PIDS+=("$(run_worker A owner --idle-sec=999999 --signal-file="${SIGNAL_FILE}")")
    PIDS+=("$(run_worker B busy)")
    (
      # Wait until A should be IDLE and B may have borrowed headroom.
      sleep "${OWNER_IDLE_SEC}"
      sleep 3
      date +%s%3N >"${OUT_DIR}/wake_unix_ms.txt"
      : >"${SIGNAL_FILE}"
      echo "signaled owner wake at $(cat "${OUT_DIR}/wake_unix_ms.txt")"
    ) &
    PIDS+=("$!")
    ;;
  noisy)
    PIDS+=("$(run_worker A noisy --noisy-on=0.25 --noisy-off=0.75)")
    PIDS+=("$(run_worker B busy)")
    ;;
  *)
    die "unknown SCENARIO=${SCENARIO}"
    ;;
esac

# Wait for workers (sm_burn) roughly duration + margin.
sleep $((DURATION_S + 3))
cleanup
trap - EXIT

# Record meta
cat >"${OUT_DIR}/meta.json" <<EOF
{
  "variant": "${VARIANT}",
  "scenario": "${SCENARIO}",
  "floor_sm": ${FLOOR_SM},
  "duration_s": ${DURATION_S},
  "libvgpu": "${LIBVGPU}",
  "run_id": "${RUN_ID}"
}
EOF

python3 "${SCRIPT_DIR}/summarize.py" "${OUT_DIR}" | tee "${OUT_DIR}/summary.json"
echo "done: ${OUT_DIR}/summary.json"
