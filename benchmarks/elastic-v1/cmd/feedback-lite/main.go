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

// feedback-lite: K8s-free elastic compute policy loop for the elastic-v1 harness.
// Same policy as vGPUmonitor light tick (reclaim immediate, grant stepped,
// candidate excluded). Does not create a C++ monitor. Does not sample NVML.
// Watcher limit re-read remains in libvgpu; this process only writes dynamic_sm_limit.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"k8s.io/klog/v2"

	"github.com/Project-HAMi/HAMi/pkg/monitor/nvidia"
)

type snapContainer struct {
	Name    string `json:"name"`
	State   int32  `json:"state"`
	Floor   uint64 `json:"floor"`
	Dynamic uint64 `json:"dynamic"`
	SmUtil  uint64 `json:"sm_util"`
	UUID    string `json:"uuid,omitempty"`
}

type snapLine struct {
	Event      string          `json:"event"`
	TUnixMs    int64           `json:"t_unix_ms"`
	Containers []snapContainer `json:"containers"`
}

func envMs(name string, def int) time.Duration {
	raw := os.Getenv(name)
	if raw == "" {
		return time.Duration(def) * time.Millisecond
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < 0 {
		klog.Warningf("invalid %s=%q, using %d", name, raw, def)
		return time.Duration(def) * time.Millisecond
	}
	return time.Duration(v) * time.Millisecond
}

func envU64(name string, def uint64) uint64 {
	raw := os.Getenv(name)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		return def
	}
	return v
}

func refreshContainers(hookPath string, cur map[string]*nvidia.ContainerUsage) map[string]*nvidia.ContainerUsage {
	next, err := nvidia.ScanHookContainers(hookPath)
	if err != nil {
		klog.Errorf("scan: %v", err)
		return cur
	}
	for name, old := range cur {
		if _, ok := next[name]; !ok {
			old.Close()
		}
	}
	for name, neu := range next {
		if old, ok := cur[name]; ok {
			// Remap each heavy tick so we do not keep a stale fd mapping forever.
			old.Close()
			_ = neu
		}
		_ = name
	}
	return next
}

func main() {
	hookPath := flag.String("hook-path", os.Getenv("HOOK_PATH"), "HOOK_PATH / CONTAINER_VGPU_MOUNT root")
	metricsPath := flag.String("metrics-log", "", "append JSON snapshots here (optional)")
	flag.Parse()
	if *hookPath == "" {
		fmt.Fprintln(os.Stderr, "hook-path / HOOK_PATH required")
		os.Exit(2)
	}

	light := envMs("HAMI_COMPUTE_LIGHT_TICK_MS", 200)
	heavy := envMs("HAMI_COMPUTE_HEAVY_TICK_MS", 2000)
	cfg := nvidia.ComputePolicyConfig{
		IdleCandidateNs: uint64(envMs("HAMI_COMPUTE_IDLE_CANDIDATE_MS", 1000)),
		IdleThresholdNs: uint64(envMs("HAMI_COMPUTE_IDLE_MS", 2000)),
		BurstCap:        envU64("HAMI_COMPUTE_BURST_CAP", 85),
		GrantStep:       envU64("HAMI_COMPUTE_GRANT_STEP", 10),
		Capacity:        nvidia.DefaultCardCoreCapacity,
	}
	if cfg.IdleThresholdNs < cfg.IdleCandidateNs {
		cfg.IdleThresholdNs = cfg.IdleCandidateNs
	}

	klog.Infof("feedback-lite hook=%s light=%v heavy=%v cap=%d step=%d (no C++ monitor)",
		*hookPath, light, heavy, cfg.BurstCap, cfg.GrantStep)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var metricsFile *os.File
	if *metricsPath != "" {
		var err error
		metricsFile, err = os.OpenFile(*metricsPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			klog.Fatalf("open metrics: %v", err)
		}
		defer metricsFile.Close()
	}

	containers := refreshContainers(*hookPath, nil)

	lightTicker := time.NewTicker(light)
	defer lightTicker.Stop()
	heavyTicker := time.NewTicker(heavy)
	defer heavyTicker.Stop()
	snapTicker := time.NewTicker(200 * time.Millisecond)
	defer snapTicker.Stop()

	writeSnap := func() {
		if len(containers) == 0 {
			return
		}
		line := snapLine{Event: "policy", TUnixMs: time.Now().UnixMilli()}
		for _, c := range containers {
			sc := snapContainer{
				Name:  c.ContainerName,
				State: c.Info.GetComputeState(),
			}
			for i := range c.Info.DeviceMax() {
				if !c.Info.IsValidUUID(i) {
					continue
				}
				sc.UUID = c.Info.DeviceUUID(i)
				floor := c.Info.GetFloorSmLimit(i)
				if floor == 0 {
					floor = c.Info.GetDeviceSmLimit(i)
				}
				sc.Floor = floor
				sc.Dynamic = c.Info.GetDynamicSmLimit(i)
				sc.SmUtil = c.Info.DeviceSmUtil(i)
				break
			}
			line.Containers = append(line.Containers, sc)
		}
		b, _ := json.Marshal(line)
		b = append(b, '\n')
		if metricsFile != nil {
			_, _ = metricsFile.Write(b)
		} else {
			_, _ = os.Stdout.Write(b)
		}
	}

	for {
		select {
		case <-ctx.Done():
			for _, c := range containers {
				c.Close()
			}
			klog.Info("feedback-lite exit")
			return
		case <-heavyTicker.C:
			containers = refreshContainers(*hookPath, containers)
		case <-lightTicker.C:
			if len(containers) == 0 {
				continue
			}
			nvidia.ObserveComputePolicy(containers, cfg)
		case <-snapTicker.C:
			writeSnap()
		}
	}
}
