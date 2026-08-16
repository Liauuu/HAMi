#!/usr/bin/env bash
# S2 보정(floor 10/20/40, stock) + 본게임(stock 10 vs v1 10 -> 대여 시 20)
set -euo pipefail

HARNESS="/home/Ubuntu/hami-bench/HAMi/benchmarks/elastic-v1"
STOCK_LIBVGPU="/home/Ubuntu/hami-bench/libvgpu-stock.so"
V1_LIBVGPU="/home/Ubuntu/hami-bench/libvgpu-v1.so"
DURATION_S=35

export GPU_CORE_UTILIZATION_POLICY=force
export DURATION_S
cd "${HARNESS}"
[[ -x ./scripts/run_scenario.sh ]] || { echo "ERROR: HARNESS path wrong: ${HARNESS}"; exit 1; }
[[ -f "${STOCK_LIBVGPU}" ]] || { echo "ERROR: missing STOCK_LIBVGPU=${STOCK_LIBVGPU}"; exit 1; }
[[ -f "${V1_LIBVGPU}" ]] || { echo "ERROR: missing V1_LIBVGPU=${V1_LIBVGPU}"; exit 1; }

make build
OUT_ROOT="${HARNESS}/.run/out"
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/s2-rerun-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "log=${LOG}"

run_s2() {
  local variant="$1" floor="$2" lib="$3"
  echo
  echo "======== VARIANT=${variant} FLOOR_SM=${floor} lib=${lib} ========"
  VARIANT="${variant}" SCENARIO=s2 FLOOR_SM="${floor}" LIBVGPU="${lib}" \
    ./scripts/run_scenario.sh
}

latest_summary() {
  local variant="$1" floor="$2"
  python3 - "${OUT_ROOT}" "${variant}" "${floor}" <<'PY'
import json, sys
from pathlib import Path
root, variant, floor = Path(sys.argv[1]), sys.argv[2], str(sys.argv[3])
cands = []
base = root / variant / "s2"
if not base.is_dir():
    sys.exit(f"no runs at {base}")
for d in base.iterdir():
    s = d / "summary.json"
    m = d / "meta.json"
    if not s.exists() or not m.exists():
        continue
    meta = json.loads(m.read_text())
    if str(meta.get("floor_sm")) != floor:
        continue
    cands.append((s.stat().st_mtime, s, meta))
if not cands:
    sys.exit(f"no summary for variant={variant} floor={floor}")
cands.sort()
print(cands[-1][1])
PY
}

print_b() {
  local label="$1" summary="$2"
  python3 - "${label}" "${summary}" <<'PY'
import json, sys
label, path = sys.argv[1], sys.argv[2]
d = json.loads(open(path).read())
b = (d.get("workers") or {}).get("B") or {}
pol = ((d.get("policy") or {}).get("containers") or {}).get("B") or {}
nvs = d.get("nvsmi") or {}
print(
    f"{label}\n"
    f"  summary={path}\n"
    f"  B_median_iters_per_s={b.get('median_busy_iters_per_s')}\n"
    f"  B_total_iters={b.get('total_iters')}\n"
    f"  B_median_dynamic={pol.get('median_dynamic')}\n"
    f"  B_max_dynamic={pol.get('max_dynamic')}\n"
    f"  nvsmi_median_gpu_util={nvs.get('median_gpu_util')}"
)
PY
}

echo "########## 1단계 보정: stock S2, floor 10 / 20 / 40 ##########"
for floor in 10 20 40; do
  run_s2 stock "${floor}" "${STOCK_LIBVGPU}"
done

echo
echo "########## 1단계 결과 (B iters가 floor 따라 줄어야 한도가 먹은 것) ##########"
S10=$(latest_summary stock 10)
S20=$(latest_summary stock 20)
S40=$(latest_summary stock 40)
print_b "stock floor=10" "${S10}"
print_b "stock floor=20" "${S20}"
print_b "stock floor=40" "${S40}"

python3 - "${S10}" "${S20}" "${S40}" <<'PY'
import json, sys
def med(p):
    d = json.loads(open(p).read())
    return (d.get("workers") or {}).get("B", {}).get("median_busy_iters_per_s")
m10, m20, m40 = med(sys.argv[1]), med(sys.argv[2]), med(sys.argv[3])
print(f"\ncalib B median: 10={m10}  20={m20}  40={m40}")
if not all(isinstance(x, (int, float)) and x > 0 for x in (m10, m20, m40)):
    print("STOP: 숫자 없음. summary/B ticks를 확인하세요. 2단계 건너뛰는 걸 권장.")
    raise SystemExit(2)
if m10 >= m40 * 0.90:
    print("STOP: floor 10이 40과 비슷함 (차이 <10%). 이 워크로드는 한도에 안 걸림.")
    print("      2단계는 의미 없음. sm_burn을 무겁게 하기 전에는 본게임 하지 마세요.")
    raise SystemExit(3)
print("OK: floor가 낮을수록 느림 -> 한도가 병목. 2단계 진행.")
PY

echo
echo "########## 2단계 본게임: v1 S2 floor=10 (idle 대여 -> dynamic≈20) ##########"
echo "stock floor=10은 1단계 런을 재사용합니다."
run_s2 v1 10 "${V1_LIBVGPU}"
V10=$(latest_summary v1 10)

echo
echo "########## 2단계 결과 (10 vs 대여 20) ##########"
print_b "stock floor=10 (보정 재사용)" "${S10}"
print_b "v1    floor=10 (expect B dynamic≈20)" "${V10}"

python3 - "${S10}" "${V10}" <<'PY'
import json, sys
def load(p):
    return json.loads(open(p).read())
stock, v1 = load(sys.argv[1]), load(sys.argv[2])
sb = stock["workers"]["B"]["median_busy_iters_per_s"]
vb = v1["workers"]["B"]["median_busy_iters_per_s"]
dyn = ((v1.get("policy") or {}).get("containers") or {}).get("B") or {}
ratio = (vb / sb) if sb else None
print(f"\nB median iters/s  stock={sb}  v1={vb}  speedup={ratio:.3f}" if ratio else "speedup n/a")
print(f"v1 B median_dynamic={dyn.get('median_dynamic')} max_dynamic={dyn.get('max_dynamic')}")
if dyn.get("median_dynamic") in (None, 0, 10.0):
    print("WARN: v1이 20 근처로 안 올렸으면 대여가 안 된 것 (A가 IDLE인지 policy.jsonl 확인).")
if ratio and ratio > 1.05:
    print("PASS: v1이 stock 대비 확실히 빠름.")
elif ratio and ratio > 1.0:
    print("WEAK: 약간만 빠름. 런을 한 번 더 하거나 duration을 늘려 보세요.")
else:
    print("FAIL: 대여는 됐을 수 있어도 throughput 승리는 아님.")
PY

echo
echo "done. full log: ${LOG}"
