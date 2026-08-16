#!/usr/bin/env python3
"""Summarize one elastic-v1 harness run directory into summary.json fields on stdout."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def worker_stats(path: Path) -> dict:
    rows = load_jsonl(path)
    ticks = [r for r in rows if r.get("event") == "tick"]
    modes = [r for r in rows if r.get("event") == "mode"]
    busy_ips = [float(r["iters_per_s"]) for r in ticks if r.get("mode") == "busy"]
    return {
        "ticks": len(ticks),
        "mode_changes": len(modes),
        "median_busy_iters_per_s": median(busy_ips),
        "mean_busy_iters_per_s": (sum(busy_ips) / len(busy_ips)) if busy_ips else float("nan"),
        "total_iters": next((r.get("iters") for r in reversed(rows) if r.get("event") == "done"), None),
    }


def policy_stats(path: Path, floor_hint: float | None = None) -> dict:
    rows = load_jsonl(path)
    if not rows:
        return {"samples": 0}

    by_name: dict[str, list[dict]] = {}
    for r in rows:
        for c in r.get("containers") or []:
            by_name.setdefault(c.get("name") or "?", []).append(c)

    per = {}
    for name, samples in by_name.items():
        floors = [float(s.get("floor") or 0) for s in samples]
        dyns = [float(s.get("dynamic") or 0) for s in samples]
        utils = [float(s.get("sm_util") or 0) for s in samples]
        states = [int(s.get("state") or 0) for s in samples]
        floor = median([f for f in floors if f > 0]) if any(floors) else (floor_hint or float("nan"))
        # Floor violation: shared-region sm_util above floor while we care about guarantee.
        # Using util vs floor is noisy; also track dynamic < floor (should never happen).
        dyn_below = sum(1 for d, f in zip(dyns, floors) if f > 0 and d > 0 and d + 1e-9 < f)
        util_above = sum(1 for u, f in zip(utils, floors) if f > 0 and u > f + 2)
        # Oscillation proxy: dynamic peak-to-peak and state flips.
        state_flips = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
        per[name] = {
            "samples": len(samples),
            "median_dynamic": median(dyns),
            "max_dynamic": max(dyns) if dyns else float("nan"),
            "median_sm_util": median(utils),
            "dynamic_below_floor_ratio": dyn_below / len(samples) if samples else float("nan"),
            "sm_util_above_floor_ratio": util_above / len(samples) if samples else float("nan"),
            "state_flips": state_flips,
            "median_floor": floor,
        }
    return {"samples": len(rows), "containers": per}


def reclaim_latency_ms(out_dir: Path) -> float | None:
    wake_path = out_dir / "wake_unix_ms.txt"
    policy_path = out_dir / "policy.jsonl"
    if not wake_path.exists() or not policy_path.exists():
        return None
    try:
        wake_ms = int(wake_path.read_text().strip())
    except ValueError:
        return None
    # Reclaim: after wake, borrower B dynamic returns to near floor.
    for r in load_jsonl(policy_path):
        t = int(r.get("t_unix_ms") or 0)
        if t < wake_ms:
            continue
        for c in r.get("containers") or []:
            if c.get("name") != "B":
                continue
            floor = float(c.get("floor") or 0)
            dyn = float(c.get("dynamic") or 0)
            if floor > 0 and dyn <= floor + 1:
                return float(t - wake_ms)
    return None


def nvsmi_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    utils = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            utils.append(float(parts[1]))
        except ValueError:
            continue
    return {
        "median_gpu_util": median(utils),
        "mean_gpu_util": (sum(utils) / len(utils)) if utils else float("nan"),
        "samples": len(utils),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    meta = {}
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    summary = {
        "meta": meta,
        "workers": {
            "A": worker_stats(out_dir / "A.jsonl"),
            "B": worker_stats(out_dir / "B.jsonl"),
        },
        "policy": policy_stats(out_dir / "policy.jsonl", floor_hint=meta.get("floor_sm")),
        "nvsmi": nvsmi_stats(out_dir / "nvsmi_util.csv"),
        "reclaim_latency_ms": reclaim_latency_ms(out_dir),
    }
    # Replace NaN for JSON
    def clean(o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    print(json.dumps(clean(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
