/*
Copyright 2024 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    10|Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nvidia

import (
	"time"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
)

// Compute-state values mirrored from HAMi-core COMPUTE_STATE_*.
const (
	ComputeStateUnset         int32 = 0
	ComputeStateActive        int32 = 1
	ComputeStateIdleCandidate int32 = 2
	ComputeStateIdle          int32 = 3
)

const DefaultCardCoreCapacity uint64 = 100

// ComputePolicyConfig holds elastic SM headroom knobs.
// Reclaim is always immediate; grant may be stepped. IDLE_CANDIDATE never lends.
type ComputePolicyConfig struct {
	IdleCandidateNs uint64
	IdleThresholdNs uint64
	BurstCap        uint64
	GrantStep       uint64
	Capacity        uint64
}

type gpuMember struct {
	c      *ContainerUsage
	devIdx int
}

// MonotonicNowNs matches libvgpu last_launch_ns clock (CLOCK_MONOTONIC_COARSE).
func MonotonicNowNs() uint64 {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC_COARSE, &ts); err != nil {
		if err2 := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts); err2 != nil {
			return 0
		}
	}
	return uint64(ts.Sec)*uint64(time.Second) + uint64(ts.Nsec)
}

// UpdateComputeState applies hysteresis idle transitions for one container.
func UpdateComputeState(c *ContainerUsage, nowNs, candidateNs, thresholdNs uint64) {
	if c == nil || c.Info == nil {
		return
	}
	if thresholdNs < candidateNs {
		thresholdNs = candidateNs
	}

	last := c.Info.GetLastLaunchNs()
	var idleFor uint64
	switch {
	case last == 0:
		idleFor = thresholdNs
	case nowNs >= last:
		idleFor = nowNs - last
	default:
		idleFor = 0
	}

	var next int32
	switch {
	case idleFor < candidateNs:
		next = ComputeStateActive
	case idleFor < thresholdNs:
		next = ComputeStateIdleCandidate
	default:
		next = ComputeStateIdle
	}

	cur := c.Info.GetComputeState()
	if cur == next {
		return
	}
	if cur != ComputeStateUnset {
		klog.V(5).Infof("compute_state %s/%s %d -> %d (idleFor=%dns last=%d)",
			c.PodUID, c.ContainerName, cur, next, idleFor, last)
	}
	c.Info.SetComputeState(next)
}

func floorOf(m gpuMember) uint64 {
	floor := m.c.Info.GetFloorSmLimit(m.devIdx)
	if floor == 0 {
		floor = m.c.Info.GetDeviceSmLimit(m.devIdx)
	}
	return floor
}

// ApplyDynamicLimitFor writes dynamic_sm_limit for one device slot.
func ApplyDynamicLimitFor(c *ContainerUsage, devIdx int, target, step uint64) {
	ApplyDynamicLimit(gpuMember{c: c, devIdx: devIdx}, target, step)
}

// ApplyDynamicLimit writes dynamic_sm_limit: immediate on decrease, stepped on increase.
func ApplyDynamicLimit(m gpuMember, target, step uint64) {
	floor := floorOf(m)
	if target < floor {
		target = floor
	}
	cur := m.c.Info.GetDynamicSmLimit(m.devIdx)
	if cur == 0 {
		cur = floor
	}
	next := target
	if target < cur {
		next = target
	} else if target > cur {
		if step == 0 {
			next = target
		} else {
			stepped := cur + step
			if stepped < target {
				next = stepped
			} else {
				next = target
			}
		}
	}
	if m.c.Info.GetDynamicSmLimit(m.devIdx) != next {
		klog.V(5).Infof("dynamic_sm_limit %s/%s dev=%d %d -> %d (target=%d floor=%d)",
			m.c.PodUID, m.c.ContainerName, m.devIdx,
			m.c.Info.GetDynamicSmLimit(m.devIdx), next, target, floor)
		m.c.Info.SetDynamicSmLimit(m.devIdx, next)
	}
}

func clampTargetsToCapacity(members []gpuMember, targets map[gpuMember]uint64, capacity uint64) {
	var sum uint64
	for _, m := range members {
		sum += targets[m]
	}
	if sum <= capacity {
		return
	}
	overflow := sum - capacity

	type surplusEntry struct {
		m       gpuMember
		surplus uint64
	}
	var entries []surplusEntry
	var totalSurplus uint64
	for _, m := range members {
		floor := floorOf(m)
		if targets[m] > floor {
			s := targets[m] - floor
			entries = append(entries, surplusEntry{m: m, surplus: s})
			totalSurplus += s
		}
	}
	if totalSurplus == 0 || overflow == 0 {
		return
	}

	var reduced uint64
	for i := range entries {
		cut := overflow * entries[i].surplus / totalSurplus
		if cut > entries[i].surplus {
			cut = entries[i].surplus
		}
		targets[entries[i].m] -= cut
		reduced += cut
	}
	for reduced < overflow {
		progress := false
		for i := range entries {
			m := entries[i].m
			floor := floorOf(m)
			if targets[m] > floor {
				targets[m]--
				reduced++
				progress = true
				if reduced >= overflow {
					break
				}
			}
		}
		if !progress {
			break
		}
	}
}

// RedistributeHeadroom lends IDLE floors to ACTIVE containers on the same GPU.
// IDLE_CANDIDATE does not contribute headroom.
func RedistributeHeadroom(containers map[string]*ContainerUsage, cap, step, capacity uint64) {
	if capacity == 0 {
		capacity = DefaultCardCoreCapacity
	}
	groups := map[string][]gpuMember{}
	for _, c := range containers {
		for i := range c.Info.DeviceMax() {
			if !c.Info.IsValidUUID(i) {
				continue
			}
			uuid := c.Info.DeviceUUID(i)
			if uuid == "" {
				continue
			}
			groups[uuid] = append(groups[uuid], gpuMember{c: c, devIdx: i})
		}
	}

	for uuid, members := range groups {
		var headroom uint64
		var actives []gpuMember
		targets := make(map[gpuMember]uint64, len(members))

		for _, m := range members {
			floor := floorOf(m)
			state := m.c.Info.GetComputeState()
			switch state {
			case ComputeStateIdle:
				headroom += floor
				targets[m] = floor
			case ComputeStateActive:
				actives = append(actives, m)
			default:
				targets[m] = floor
			}
		}

		var share uint64
		if n := uint64(len(actives)); n > 0 {
			share = headroom / n
		}

		for _, m := range actives {
			floor := floorOf(m)
			target := floor + share
			if cap > 0 && target > cap {
				target = cap
			}
			if target < floor {
				target = floor
			}
			targets[m] = target
		}

		clampTargetsToCapacity(members, targets, capacity)

		klog.V(5).Infof("headroom uuid=%s idleHeadroom=%d actives=%d share=%d burstCap=%d",
			uuid, headroom, len(actives), share, cap)

		for _, m := range members {
			ApplyDynamicLimit(m, targets[m], step)
		}
	}
}

// ObserveComputePolicy runs hysteresis state updates then headroom redistribution.
func ObserveComputePolicy(containers map[string]*ContainerUsage, cfg ComputePolicyConfig) {
	nowNs := MonotonicNowNs()
	cand := cfg.IdleCandidateNs
	thr := cfg.IdleThresholdNs
	for _, c := range containers {
		UpdateComputeState(c, nowNs, cand, thr)
	}
	RedistributeHeadroom(containers, cfg.BurstCap, cfg.GrantStep, cfg.Capacity)
}
