#!/usr/bin/env bash
# Run stock then v1 for one scenario (or all) and print compare JSON.
# IMPORTANT: point LIBVGPU at the matching build for each half, e.g.:
#   LIBVGPU=/path/stock/libvgpu.so VARIANT=stock ...
# This script expects you to export STOCK_LIBVGPU and V1_LIBVGPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TARGET="${1:-all}"
SCENARIOS=(s1 s2 s3 noisy)
if [[ "${TARGET}" != "all" ]]; then
  SCENARIOS=("${TARGET}")
fi

: "${STOCK_LIBVGPU:=${LIBVGPU}}"
: "${V1_LIBVGPU:=${LIBVGPU}}"

COMPARE_ROOT="${OUT_BASE}/compare-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${COMPARE_ROOT}"
echo "compare root: ${COMPARE_ROOT}"
echo " stock lib: ${STOCK_LIBVGPU}"
echo " v1 lib:    ${V1_LIBVGPU}"

for sc in "${SCENARIOS[@]}"; do
  echo "==== ${sc} stock ===="
  LIBVGPU="${STOCK_LIBVGPU}" VARIANT=stock SCENARIO="${sc}" RUN_ID="stock-${sc}" \
    OUT_BASE="${COMPARE_ROOT}" bash "${SCRIPT_DIR}/run_scenario.sh"
  STOCK_DIR="$(ls -d "${COMPARE_ROOT}/stock/${sc}"/* | tail -n1)"

  echo "==== ${sc} v1 ===="
  LIBVGPU="${V1_LIBVGPU}" VARIANT=v1 SCENARIO="${sc}" RUN_ID="v1-${sc}" \
    OUT_BASE="${COMPARE_ROOT}" bash "${SCRIPT_DIR}/run_scenario.sh"
  V1_DIR="$(ls -d "${COMPARE_ROOT}/v1/${sc}"/* | tail -n1)"

  python3 "${SCRIPT_DIR}/compare_pair.py" "${STOCK_DIR}" "${V1_DIR}" \
    | tee "${COMPARE_ROOT}/${sc}_compare.json"
done

echo "all results under ${COMPARE_ROOT}"
