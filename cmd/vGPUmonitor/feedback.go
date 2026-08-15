/*
Copyright 2024 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"

	"github.com/Project-HAMi/HAMi/pkg/monitor/nvidia"
)

var errTemporaryClosed = errors.New("temporary closed")

// Compute-state values mirrored from HAMi-core COMPUTE_STATE_*.
const (
	computeStateUnset         int32 = 0
	computeStateActive        int32 = 1
	computeStateIdleCandidate int32 = 2
	computeStateIdle          int32 = 3
)

const (
	defaultIdleCandidateMs = 1000
	defaultIdleMs          = 2000
	defaultBurstCap        = 85
	defaultGrantStep       = 10
	cardCoreCapacity       = 100
)

//type hostGPUPid struct {
//	hostGPUPid int
//	mtime      uint64
//}

type UtilizationPerDevice []int

var (
	idleCandidateNs uint64
	idleThresholdNs uint64
	burstCap        uint64
	grantStep       uint64
)

func init() {
	idleCandidateNs = uint64(envDurationMs("HAMI_COMPUTE_IDLE_CANDIDATE_MS", defaultIdleCandidateMs)) * uint64(time.Millisecond)
	idleThresholdNs = uint64(envDurationMs("HAMI_COMPUTE_IDLE_MS", defaultIdleMs)) * uint64(time.Millisecond)
	if idleThresholdNs < idleCandidateNs {
		klog.Warningf("HAMI_COMPUTE_IDLE_MS < HAMI_COMPUTE_IDLE_CANDIDATE_MS; raising idle to candidate (%d ms)", idleCandidateNs/uint64(time.Millisecond))
		idleThresholdNs = idleCandidateNs
	}
	burstCap = uint64(envDurationMs("HAMI_COMPUTE_BURST_CAP", defaultBurstCap))
	if burstCap == 0 || burstCap > cardCoreCapacity {
		burstCap = defaultBurstCap
	}
	grantStep = uint64(envDurationMs("HAMI_COMPUTE_GRANT_STEP", defaultGrantStep))
}

func envDurationMs(name string, def int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < 0 {
		klog.Warningf("invalid %s=%q, using default %d", name, raw, def)
		return def
	}
	return v
}

func monotonicNowNs() uint64 {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC_COARSE, &ts); err != nil {
		if err2 := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts); err2 != nil {
			return 0
		}
	}
	return uint64(ts.Sec)*uint64(time.Second) + uint64(ts.Nsec)
}

func CheckBlocking(utSwitchOn map[string]UtilizationPerDevice, p int, c *nvidia.ContainerUsage) bool {
	for i := range c.Info.DeviceMax() {
		uuid := c.Info.DeviceUUID(i)
		_, ok := utSwitchOn[uuid]
		if ok {
			for i := range min(p, len(utSwitchOn[uuid])) {
				if utSwitchOn[uuid][i] > 0 {
					return true
				}
			}
		}
	}
	return false
}

// Check whether task with higher priority use GPU or there are other tasks with the same priority.
func CheckPriority(utSwitchOn map[string]UtilizationPerDevice, p int, c *nvidia.ContainerUsage) bool {
	for i := range c.Info.DeviceMax() {
		uuid := c.Info.DeviceUUID(i)
		_, ok := utSwitchOn[uuid]
		if ok {
			for i := range min(p, len(utSwitchOn[uuid])) {
				if utSwitchOn[uuid][i] > 0 {
					return true
				}
			}
			if p >= 0 && p < len(utSwitchOn[uuid]) && utSwitchOn[uuid][p] > 1 {
				return true
			}
		}
	}
	return false
}

// updateComputeState applies hysteresis idle transitions only.
func updateComputeState(c *nvidia.ContainerUsage, nowNs uint64) {
	updateComputeStateWith(c, nowNs, idleCandidateNs, idleThresholdNs)
}

func updateComputeStateWith(c *nvidia.ContainerUsage, nowNs, candidateNs, thresholdNs uint64) {
	if thresholdNs < candidateNs {
		thresholdNs = candidateNs
	}

	last := c.Info.GetLastLaunchNs()
	var idleFor uint64
	switch {
	case last == 0:
		// Never launched (or legacy cache): treat as fully idle for classification.
		idleFor = thresholdNs
	case nowNs >= last:
		idleFor = nowNs - last
	default:
		// Clock quirk / cross-process mismatch: prefer ACTIVE.
		idleFor = 0
	}

	var next int32
	switch {
	case idleFor < candidateNs:
		next = computeStateActive
	case idleFor < thresholdNs:
		next = computeStateIdleCandidate
	default:
		next = computeStateIdle
	}

	cur := c.Info.GetComputeState()
	if cur == next {
		return
	}
	if cur != computeStateUnset {
		klog.V(5).Infof("compute_state %s/%s %d -> %d (idleFor=%dns last=%d)",
			c.PodUID, c.ContainerName, cur, next, idleFor, last)
	}
	c.Info.SetComputeState(next)
}

type gpuMember struct {
	c      *nvidia.ContainerUsage
	devIdx int
}

func floorOf(m gpuMember) uint64 {
	floor := m.c.Info.GetFloorSmLimit(m.devIdx)
	if floor == 0 {
		floor = m.c.Info.GetDeviceSmLimit(m.devIdx)
	}
	return floor
}

func applyDynamicLimit(m gpuMember, target uint64) {
	applyDynamicLimitWith(m, target, grantStep)
}

func applyDynamicLimitWith(m gpuMember, target, step uint64) {
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
		// Immediate reclaim when reducing (owner woke / lost headroom).
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
	// Fix integer remainder: shave 1 from largest surplus holders until within capacity.
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

// redistributeHeadroom lends IDLE floors to ACTIVE containers on the same GPU.
// IDLE_CANDIDATE does not contribute headroom. Reclaim is immediate via applyDynamicLimit.
func redistributeHeadroom(containers map[string]*nvidia.ContainerUsage) {
	redistributeHeadroomWith(containers, burstCap, grantStep, cardCoreCapacity)
}

func redistributeHeadroomWith(containers map[string]*nvidia.ContainerUsage, cap, step, capacity uint64) {
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
			case computeStateIdle:
				headroom += floor
				targets[m] = floor
			case computeStateActive:
				actives = append(actives, m)
			default:
				// UNSET / IDLE_CANDIDATE: keep floor, no lending, no borrowing.
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
			applyDynamicLimitWith(m, targets[m], step)
		}
	}
}

func Observe(lister *nvidia.ContainerLister) {
	utSwitchOn := map[string]UtilizationPerDevice{}
	containers := lister.ListContainers()
	nowNs := monotonicNowNs()

	for _, c := range containers {
		recentKernel := c.Info.GetRecentKernel()
		if recentKernel > 0 {
			recentKernel--
			if recentKernel > 0 {
				for i := range c.Info.DeviceMax() {
					//for _, devuuid := range val.sr.uuids {
					// Null device condition
					if !c.Info.IsValidUUID(i) {
						continue
					}
					uuid := c.Info.DeviceUUID(i)
					p := c.Info.GetPriority()
					if p < 0 {
						continue
					}
					for p >= len(utSwitchOn[uuid]) {
						utSwitchOn[uuid] = append(utSwitchOn[uuid], 0)
					}
					utSwitchOn[uuid][p]++
				}
			}
			c.Info.SetRecentKernel(recentKernel)
		}
	}
	for idx, c := range containers {
		updateComputeState(c, nowNs)

		priority := c.Info.GetPriority()
		recentKernel := c.Info.GetRecentKernel()
		utilizationSwitch := c.Info.GetUtilizationSwitch()
		if CheckBlocking(utSwitchOn, priority, c) {
			if recentKernel >= 0 {
				klog.V(5).Infof("utSwitchon=%v", utSwitchOn)
				klog.V(5).Infof("Setting Blocking to on %v", idx)
				c.Info.SetRecentKernel(-1)
			}
		} else {
			if recentKernel < 0 {
				klog.V(5).Infof("utSwitchon=%v", utSwitchOn)
				klog.V(5).Infof("Setting Blocking to off %v", idx)
				c.Info.SetRecentKernel(0)
			}
		}
		if CheckPriority(utSwitchOn, priority, c) {
			if utilizationSwitch != 1 {
				klog.V(5).Infof("utSwitchon=%v", utSwitchOn)
				klog.V(5).Infof("Setting UtilizationSwitch to on %v", idx)
				c.Info.SetUtilizationSwitch(1)
			}
		} else {
			if utilizationSwitch != 0 {
				klog.V(5).Infof("utSwitchon=%v", utSwitchOn)
				klog.V(5).Infof("Setting UtilizationSwitch to off %v", idx)
				c.Info.SetUtilizationSwitch(0)
			}
		}
	}

	// After states settle: lend IDLE headroom to ACTIVE on the same GPU.
	redistributeHeadroom(containers)
}

func watchAndFeedback(ctx context.Context, lister *nvidia.ContainerLister, migLockSignal <-chan bool) error {
	klog.Info("Starting watchAndFeedback")
	if nvret := nvml.Init(); nvret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %s", nvml.ErrorString(nvret))
	}
	defer nvml.Shutdown()

	ticker := time.NewTicker(time.Second * 5)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			klog.Info("Shutting down watchAndFeedback")
			return nil
		case signal := <-migLockSignal:
			if signal {
				klog.Info("Received MIG apply lock file")
				return errTemporaryClosed
			}

		case <-ticker.C:
			if err := lister.Update(); err != nil {
				klog.Errorf("Failed to update container list: %v", err)
				continue
			}
			Observe(lister)
		}
	}
}
