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
	"k8s.io/klog/v2"

	"github.com/Project-HAMi/HAMi/pkg/monitor/nvidia"
)

var errTemporaryClosed = errors.New("temporary closed")

// Compute-state aliases for existing call sites / tests.
const (
	computeStateUnset         = nvidia.ComputeStateUnset
	computeStateActive        = nvidia.ComputeStateActive
	computeStateIdleCandidate = nvidia.ComputeStateIdleCandidate
	computeStateIdle          = nvidia.ComputeStateIdle
)

const (
	defaultIdleCandidateMs = 1000
	defaultIdleMs          = 2000
	defaultBurstCap        = 85
	defaultGrantStep       = 10
	cardCoreCapacity       = 100

	// Light tick: compute_state + dynamic_sm_limit only (shared-memory R/W).
	// Keep this in the 100–200ms band; do not pull heavy work onto it.
	defaultLightTickMs = 200
	minLightTickMs     = 50
	// Heavy tick: container-dir/pod refresh (+ legacy priority/recent_kernel).
	// NVML Init stays once at start; metrics NVML stays on its own path.
	defaultHeavyTickMs = 5000
	minHeavyTickMs     = 1000
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
	lightTick       time.Duration
	heavyTick       time.Duration
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

	lightTick = clampTickMs("HAMI_COMPUTE_LIGHT_TICK_MS", defaultLightTickMs, minLightTickMs, 0)
	heavyTick = clampTickMs("HAMI_COMPUTE_HEAVY_TICK_MS", defaultHeavyTickMs, minHeavyTickMs, 0)
	if lightTick >= heavyTick {
		klog.Warningf("HAMI_COMPUTE_LIGHT_TICK_MS (%v) >= HEAVY (%v); light path still skips Update()", lightTick, heavyTick)
	}
}

// clampTickMs parses an env duration in milliseconds and clamps to [minMs, maxMs]
// (maxMs<=0 means no upper bound).
func clampTickMs(name string, def, minMs, maxMs int) time.Duration {
	v := envDurationMs(name, def)
	if v < minMs {
		klog.Warningf("invalid %s=%d (< %d), using %d", name, v, minMs, minMs)
		v = minMs
	}
	if maxMs > 0 && v > maxMs {
		klog.Warningf("invalid %s=%d (> %d), using %d", name, v, maxMs, maxMs)
		v = maxMs
	}
	return time.Duration(v) * time.Millisecond
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

func monotonicNowNs() uint64 { return nvidia.MonotonicNowNs() }

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
	nvidia.UpdateComputeState(c, nowNs, candidateNs, thresholdNs)
}

type gpuMember struct {
	c      *nvidia.ContainerUsage
	devIdx int
}

func applyDynamicLimitWith(m gpuMember, target, step uint64) {
	nvidia.ApplyDynamicLimitFor(m.c, m.devIdx, target, step)
}

func redistributeHeadroom(containers map[string]*nvidia.ContainerUsage) {
	redistributeHeadroomWith(containers, burstCap, grantStep, cardCoreCapacity)
}

func redistributeHeadroomWith(containers map[string]*nvidia.ContainerUsage, cap, step, capacity uint64) {
	nvidia.RedistributeHeadroom(containers, cap, step, capacity)
}

// Observe runs the full feedback pass (legacy priority + compute elastic).
// Prefer observeComputePolicy on the light tick so recent_kernel's per-tick
// decrement keeps its historical ~5s timescale.
func Observe(lister *nvidia.ContainerLister) {
	containers := lister.ListContainers()
	observePriorityFeedback(containers)
	observeComputePolicy(containers)
}

// observeComputePolicy: light work only — hysteresis state + dynamic limits.
// Candidate never contributes headroom; reclaim is immediate, grant is stepped.
func observeComputePolicy(containers map[string]*nvidia.ContainerUsage) {
	nvidia.ObserveComputePolicy(containers, nvidia.ComputePolicyConfig{
		IdleCandidateNs: idleCandidateNs,
		IdleThresholdNs: idleThresholdNs,
		BurstCap:        burstCap,
		GrantStep:       grantStep,
		Capacity:        cardCoreCapacity,
	})
}

// observePriorityFeedback: legacy recent_kernel / utilization_switch path.
// recent_kernel is decremented once per call; keep this on the heavy tick.
func observePriorityFeedback(containers map[string]*nvidia.ContainerUsage) {
	utSwitchOn := map[string]UtilizationPerDevice{}

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
}

func watchAndFeedback(ctx context.Context, lister *nvidia.ContainerLister, migLockSignal <-chan bool) error {
	klog.Infof("Starting watchAndFeedback (light=%v heavy=%v; no C++ monitor)", lightTick, heavyTick)
	if nvret := nvml.Init(); nvret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %s", nvml.ErrorString(nvret))
	}
	defer nvml.Shutdown()

	// Prime container map so the first light tick is not a no-op.
	if err := lister.Update(); err != nil {
		klog.Errorf("Failed initial container list update: %v", err)
	}

	lightTicker := time.NewTicker(lightTick)
	defer lightTicker.Stop()
	heavyTicker := time.NewTicker(heavyTick)
	defer heavyTicker.Stop()

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

		case <-lightTicker.C:
			// Shared-memory only: state transitions + dynamic writes.
			// Watcher re-reads effective limit every libvgpu tick independently.
			observeComputePolicy(lister.ListContainers())

		case <-heavyTicker.C:
			if err := lister.Update(); err != nil {
				klog.Errorf("Failed to update container list: %v", err)
				continue
			}
			Observe(lister)
		}
	}
}
