# elastic-v1 step-8 harness (순정 vs v1)

Standalone GPU-VM harness for comparing **stock (`main`)** vs **v1 (`feat/dynamic-sm-limit`)**.

Does **not** create a C++ monitor. Uses `libvgpu.so` + Go `feedback-lite` (same policy as vGPUmonitor light tick: **reclaim immediate, grant stepped, candidate excluded from headroom**). libvgpu watcher still **re-reads effective limit every tick**.

## What it runs

| Scenario | A | B | What you look for |
|----------|---|---|-------------------|
| **s1** | busy | busy | Both near floor; stock ≈ v1; low floor violation |
| **s2** | idle | busy | v1 B throughput / dynamic > stock |
| **s3** | idle→busy | busy | v1 reclaim latency small after A wakes |
| **noisy** | short wake/sleep | busy | Oscillation not insane (`state_flips`, dynamic swing) |

Metrics (per run `summary.json`):

- Per-worker busy `iters_per_s` (throughput)
- Policy snapshots: `dynamic`, `floor`, `state`, shared-region `sm_util`
- `reclaim_latency_ms` (s3): wake signal → B `dynamic` back to floor
- Whole-GPU util via `nvidia-smi` (coarse)

## Layout

```
benchmarks/elastic-v1/
  workload/sm_burn.cu          # controllable CUDA load
  cmd/feedback-lite/           # K8s-free policy loop (writes dynamic_sm_limit)
  scripts/run_scenario.sh      # one variant × one scenario
  scripts/run_compare.sh       # stock then v1
  scripts/summarize.py
  scripts/compare_pair.py
```

## Build (GPU VM)

```bash
# 1) matching libvgpu for the variant under test
cd /path/to/HAMi-core
git checkout main                 # or feat/dynamic-sm-limit for v1
./build.sh
export LIBVGPU=$PWD/build/libvgpu.so

# 2) harness binaries (from HAMi repo)
cd /path/to/HAMi/benchmarks/elastic-v1
make build
```

For a fair A/B you need **two libvgpu builds** (stock tree vs v1 tree). Point `LIBVGPU` at the one under test each time.

## Run

```bash
# single shot
VARIANT=v1 SCENARIO=s2 DURATION_S=35 FLOOR_SM=40 LIBVGPU=/path/to/libvgpu.so \
  ./scripts/run_scenario.sh

# stock vs v1 for one scenario (rebuild/swap LIBVGPU between sides yourself,
# or run twice with different LIBVGPU)
LIBVGPU=/path/to/stock/libvgpu.so VARIANT=stock SCENARIO=s2 ./scripts/run_scenario.sh
LIBVGPU=/path/to/v1/libvgpu.so    VARIANT=v1    SCENARIO=s2 ./scripts/run_scenario.sh

# helper that runs stock then v1 with the *current* LIBVGPU — swap the .so between calls
./scripts/run_compare.sh s2
./scripts/run_compare.sh all
```

Env knobs (optional):

- `FLOOR_SM` (default 40) — both workers’ `CUDA_DEVICE_SM_LIMIT`
- `DURATION_S` (default 35)
- `HAMI_COMPUTE_*` — same as production monitor (idle ms, burst cap, grant step, light/heavy tick)
- `GPU_CORE_UTILIZATION_POLICY=force` (default in harness) so limits stay enforced

## Local dry-run (no GPU)

```bash
make dry-run          # builds feedback-lite + summarizes sample JSON
go test ./cmd/vGPUmonitor/ ./pkg/monitor/nvidia/
```

## Pass / fail reading

- **s1**: B median busy iters/s close; `dynamic_below_floor_ratio` near 0 on both
- **s2**: `B_throughput_speedup_v1_over_stock` > 1; v1 `B median_dynamic` > floor (`sm_util_above_floor` on B is expected lending, not a breach)
- **s3**: v1 `reclaim_latency_ms` small (order of light tick + watcher); after reclaim both near floor. Owner wakes **only** via signal file (not `--idle-sec` auto-wake)
- **noisy**: prefer fewer wild `state_flips` / dynamic thrash on v1 with default hysteresis

## Troubleshooting s2 (A idle / B busy, but no lending)

If `summary.json` shows both containers `state=3` (IDLE), `last_launch_ns=0`,
while `nvidia-smi` util is high:

**Root cause:** cudart resolves `cuLaunchKernel` via `dlsym(libcuda)` and
caches the *real* driver pointer. LD_PRELOAD alone does not see those launches,
so `mark_compute_active` never runs.

**Harness fix:** `sm_burn` calls Driver API `cuLaunchKernel` itself (PLT →
libvgpu). `run_worker` also sets `CUDA_REDIRECT=$LIBVGPU`.

```bash
make build-sm-burn
# quick hook smoke (stderr should stay quiet; last_launch must become non-zero):
HOOK_BASE=$(pwd)/.run/hook
mkdir -p "$HOOK_BASE/containers/bench-B_B"
CONTAINER_VGPU_MOUNT=$HOOK_BASE HOOK_PATH=$HOOK_BASE \
  POD_UID=bench-B CONTAINER_NAME=B \
  CUDA_DEVICE_MEMORY_LIMIT=2g CUDA_DEVICE_SM_LIMIT=40 \
  GPU_CORE_UTILIZATION_POLICY=force \
  LD_PRELOAD=$LIBVGPU CUDA_REDIRECT=$LIBVGPU \
  ./bin/sm_burn --name=B --mode=busy --duration=2
# while it runs in another shell, or after: check usage.cache via feedback-lite
VARIANT=v1 SCENARIO=s2 ./scripts/run_scenario.sh
```

Expect mid-run policy: A `state=3` `dynamic=40`; B `state=1` `last_launch_ns>0` `dynamic>40`.

`make clean` only removes `bin/`; use `make clean-all` to wipe `.run` logs.

Do **not** hand-edit HAMi-core `memory.c` includes for this harness; rebuild from a
clean cmake tree (`./build.sh`) if `libvgpu` fails to compile.
