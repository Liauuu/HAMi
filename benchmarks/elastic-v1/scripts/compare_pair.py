#!/usr/bin/env python3
"""Compare stock vs v1 summary.json directories for one scenario."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        # regenerate
        import subprocess

        subprocess.check_call(
            [sys.executable, str(Path(__file__).with_name("summarize.py")), str(run_dir)],
            stdout=open(p, "w"),
        )
    return json.loads(p.read_text())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_pair.py STOCK_DIR V1_DIR", file=sys.stderr)
        return 2
    stock = load_summary(Path(sys.argv[1]))
    v1 = load_summary(Path(sys.argv[2]))

    def ips(s, who):
        return (s.get("workers") or {}).get(who, {}).get("median_busy_iters_per_s")

    def dyn(s, who):
        return ((s.get("policy") or {}).get("containers") or {}).get(who, {}).get("median_dynamic")

    def dyn_below(s, who):
        return ((s.get("policy") or {}).get("containers") or {}).get(who, {}).get(
            "dynamic_below_floor_ratio"
        )

    def util_above(s, who):
        # Not a floor violation — on s2 this is the lending signal for B.
        return ((s.get("policy") or {}).get("containers") or {}).get(who, {}).get(
            "sm_util_above_floor_ratio"
        )

    out = {
        "scenario": (v1.get("meta") or {}).get("scenario") or (stock.get("meta") or {}).get("scenario"),
        "B_throughput": {"stock": ips(stock, "B"), "v1": ips(v1, "B")},
        "A_throughput": {"stock": ips(stock, "A"), "v1": ips(v1, "A")},
        "B_median_dynamic": {"stock": dyn(stock, "B"), "v1": dyn(v1, "B")},
        "floor_breach_dynamic_below_ratio": {
            "stock": {"A": dyn_below(stock, "A"), "B": dyn_below(stock, "B")},
            "v1": {"A": dyn_below(v1, "A"), "B": dyn_below(v1, "B")},
        },
        "B_sm_util_above_floor_ratio": {"stock": util_above(stock, "B"), "v1": util_above(v1, "B")},
        "reclaim_latency_ms": {
            "stock": stock.get("reclaim_latency_ms"),
            "v1": v1.get("reclaim_latency_ms"),
        },
        "gpu_util_median": {
            "stock": (stock.get("nvsmi") or {}).get("median_gpu_util"),
            "v1": (v1.get("nvsmi") or {}).get("median_gpu_util"),
        },
        "pass_hints": {
            "s1_both_busy_similar": "B throughput stock≈v1; dynamic_below_floor≈0",
            "s2_idle_lend": "v1 B throughput and/or dynamic > stock",
            "s3_reclaim": "v1 reclaim_latency_ms small; A gets floor back",
            "noisy": "check state_flips / dynamic oscillation not extreme",
        },
    }

    b_s, b_v = ips(stock, "B"), ips(v1, "B")
    if b_s and b_v and b_s > 0:
        out["B_throughput_speedup_v1_over_stock"] = b_v / b_s

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
