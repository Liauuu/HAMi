#!/usr/bin/env bash
# Common env for elastic-v1 harness workers.
# Usage: source scripts/common.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAMI_ROOT="$(cd "${ROOT}/../.." && pwd)"
CORE_ROOT="$(cd "${HAMI_ROOT}/../HAMi-core" && pwd)"

: "${FLOOR_SM:=40}"
: "${MEM_LIMIT:=2g}"
: "${DURATION_S:=35}"
: "${LIBVGPU:=${CORE_ROOT}/build/libvgpu.so}"
: "${HOOK_BASE:=${ROOT}/.run/hook}"
: "${OUT_BASE:=${ROOT}/.run/out}"
: "${SM_BURN:=${ROOT}/bin/sm_burn}"
: "${FEEDBACK_LITE:=${ROOT}/bin/feedback-lite}"

mkdir -p "${HOOK_BASE}/containers" "${OUT_BASE}" "${ROOT}/bin"

die() { echo "ERROR: $*" >&2; exit 1; }

need_bin() {
  # .so files are often 0644 (not +x); treat readable files as present too.
  [[ -x "$1" || -f "$1" ]] || die "missing $1 (run: make -C ${ROOT} build)"
}

prepare_worker_dir() {
  local pod="$1" ctr="$2"
  local dir="${HOOK_BASE}/containers/${pod}_${ctr}"
  mkdir -p "${dir}"
  # Fresh cache per run so SM limit / ABI layout matches the loaded libvgpu.
  rm -f "${dir}"/*.cache "${dir}"/cudevshr.cache 2>/dev/null || true
  echo "${dir}"
}

# Run one LD_PRELOAD worker. Args after name/mode go to sm_burn.
run_worker() {
  local name="$1" mode="$2"
  shift 2
  local pod="bench-${name}"
  local ctr="${name}"
  local dir
  dir="$(prepare_worker_dir "${pod}" "${ctr}")"
  local log="${OUT_DIR}/${name}.jsonl"

  (
    export CONTAINER_VGPU_MOUNT="${HOOK_BASE}"
    export HOOK_PATH="${HOOK_BASE}"
    export POD_UID="${pod}"
    export CONTAINER_NAME="${ctr}"
    export CUDA_DEVICE_MEMORY_LIMIT="${MEM_LIMIT}"
    export CUDA_DEVICE_SM_LIMIT="${FLOOR_SM}"
    export GPU_CORE_UTILIZATION_POLICY="${GPU_CORE_UTILIZATION_POLICY:-force}"
    export LD_PRELOAD="${LIBVGPU}"
    # libvgpu's dlsym() rewrites cu* lookups via this path (default
    # /usr/local/vgpu/libvgpu.so is usually missing on bare VM).
    export CUDA_REDIRECT="${LIBVGPU}"
    # Prefer per-container cache under hook path (set by libvgpu via POD_UID).
    unset CUDA_DEVICE_MEMORY_SHARED_CACHE || true
    exec "${SM_BURN}" --name="${name}" --mode="${mode}" --duration="${DURATION_S}" "$@"
  ) >"${log}" 2>"${OUT_DIR}/${name}.stderr" &
  echo $!
}

start_feedback_lite() {
  local metrics="${OUT_DIR}/policy.jsonl"
  export HOOK_PATH="${HOOK_BASE}"
  export HAMI_COMPUTE_LIGHT_TICK_MS="${HAMI_COMPUTE_LIGHT_TICK_MS:-200}"
  export HAMI_COMPUTE_HEAVY_TICK_MS="${HAMI_COMPUTE_HEAVY_TICK_MS:-2000}"
  export HAMI_COMPUTE_IDLE_CANDIDATE_MS="${HAMI_COMPUTE_IDLE_CANDIDATE_MS:-1000}"
  export HAMI_COMPUTE_IDLE_MS="${HAMI_COMPUTE_IDLE_MS:-2000}"
  export HAMI_COMPUTE_BURST_CAP="${HAMI_COMPUTE_BURST_CAP:-85}"
  export HAMI_COMPUTE_GRANT_STEP="${HAMI_COMPUTE_GRANT_STEP:-10}"
  "${FEEDBACK_LITE}" -hook-path="${HOOK_BASE}" -metrics-log="${metrics}" \
    >"${OUT_DIR}/feedback-lite.stdout" 2>"${OUT_DIR}/feedback-lite.stderr" &
  echo $!
}

sample_nvsmi() {
  local out="${OUT_DIR}/nvsmi_util.csv"
  echo "t_unix_ms,gpu_util" >"${out}"
  (
    while true; do
      local ts util
      ts="$(date +%s%3N)"
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
      echo "${ts},${util:-NA}" >>"${out}"
      sleep 0.2
    done
  ) &
  echo $!
}
