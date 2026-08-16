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
	"testing"
	"time"

	"github.com/Project-HAMi/HAMi/pkg/monitor/nvidia"
)

// stubDeviceMax mirrors the production Spec.DeviceMax, which reports the constant
// maxDevices (16) rather than the live device count.
const stubDeviceMax = 16

// stubInfo mocks nvidia.UsageInfo. The container's real device UUIDs occupy the
// leading slots of a fixed-size table; DeviceMax still reports the constant slot
// count and the trailing slots read back invalid, so the check functions walk the
// same 16 slots they do in production.
type stubInfo struct {
	priority   int
	uuids      []string
	total      []uint64
	limit      []uint64
	ctxSize    []uint64
	modSize    []uint64
	bufSize    []uint64
	smUtil     []uint64
	lastKernel int64
}

func slot(v []uint64, i int) uint64 {
	if i >= 0 && i < len(v) {
		return v[i]
	}
	return 0
}

func (s *stubInfo) DeviceMax() int { return stubDeviceMax }
func (s *stubInfo) DeviceNum() int { return len(s.uuids) }
func (s *stubInfo) DeviceUUID(i int) string {
	if i < len(s.uuids) {
		return s.uuids[i]
	}
	return ""
}
func (s *stubInfo) DeviceMemoryContextSize(i int) uint64 { return slot(s.ctxSize, i) }
func (s *stubInfo) DeviceMemoryModuleSize(i int) uint64  { return slot(s.modSize, i) }
func (s *stubInfo) DeviceMemoryBufferSize(i int) uint64  { return slot(s.bufSize, i) }
func (s *stubInfo) DeviceMemoryOffset(int) uint64        { return 0 }
func (s *stubInfo) DeviceMemoryTotal(i int) uint64       { return slot(s.total, i) }
func (s *stubInfo) DeviceSmUtil(i int) uint64            { return slot(s.smUtil, i) }
func (s *stubInfo) SetDeviceSmLimit(uint64)              {}
func (s *stubInfo) IsValidUUID(i int) bool {
	return i >= 0 && i < len(s.uuids) && len(s.uuids[i]) > 0 && s.uuids[i][0] != 0
}
func (s *stubInfo) DeviceMemoryLimit(i int) uint64 { return slot(s.limit, i) }
func (s *stubInfo) SetDeviceMemoryLimit(uint64)    {}
func (s *stubInfo) LastKernelTime() int64          { return s.lastKernel }
func (s *stubInfo) GetPriority() int               { return s.priority }
func (s *stubInfo) GetRecentKernel() int32         { return 1 }
func (s *stubInfo) SetRecentKernel(int32)          {}
func (s *stubInfo) GetUtilizationSwitch() int32    { return 0 }
func (s *stubInfo) SetUtilizationSwitch(int32)     {}

func TestCheckFunctionsHighPriority(t *testing.T) {
	sw := map[string]UtilizationPerDevice{"gpu-0": {0, 1}}
	c := &nvidia.ContainerUsage{Info: &stubInfo{priority: 3, uuids: []string{"gpu-0"}}}
	if !CheckBlocking(sw, 3, c) {
		t.Error("CheckBlocking: expected true")
	}
	if !CheckPriority(sw, 3, c) {
		t.Error("CheckPriority: expected true")
	}
	sw2 := map[string]UtilizationPerDevice{"gpu-0": {0, 0}}
	if CheckBlocking(sw2, 2, c) {
		t.Error("CheckBlocking: expected false")
	}
}

// TestCheckBlocking_MultiDevice verifies that CheckBlocking inspects every device
// the container uses, not just the first one that appears in the switch map. The
// cases cover several device counts, all-clear, contention isolated to the last
// device, and UUIDs missing from the switch map. In every case contention (when
// present) sits at an index below the priority, so CheckPriority agrees and is
// asserted for parity.
func TestCheckBlocking_MultiDevice(t *testing.T) {
	tests := []struct {
		name     string
		priority int
		uuids    []string
		sw       map[string]UtilizationPerDevice
		want     bool
	}{
		{
			name:     "two devices, contention on the second",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1"},
			sw:       map[string]UtilizationPerDevice{"gpu-0": {0, 0}, "gpu-1": {1, 0}},
			want:     true,
		},
		{
			name:     "three devices, all clear",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1", "gpu-2"},
			sw:       map[string]UtilizationPerDevice{"gpu-0": {0, 0}, "gpu-1": {0, 0}, "gpu-2": {0, 0}},
			want:     false,
		},
		{
			name:     "four devices, contention only on the last",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1", "gpu-2", "gpu-3"},
			sw:       map[string]UtilizationPerDevice{"gpu-0": {0, 0}, "gpu-1": {0, 0}, "gpu-2": {0, 0}, "gpu-3": {1, 0}},
			want:     true,
		},
		{
			name:     "some device UUIDs missing from switch map, present one contended",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1", "gpu-2"},
			sw:       map[string]UtilizationPerDevice{"gpu-1": {1, 0}},
			want:     true,
		},
		{
			name:     "none of the container UUIDs are in the switch map",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1"},
			sw:       map[string]UtilizationPerDevice{"gpu-9": {1, 0}},
			want:     false,
		},
		{
			name:     "empty switch map",
			priority: 1,
			uuids:    []string{"gpu-0", "gpu-1"},
			sw:       map[string]UtilizationPerDevice{},
			want:     false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := &nvidia.ContainerUsage{Info: &stubInfo{priority: test.priority, uuids: test.uuids}}
			if got := CheckBlocking(test.sw, test.priority, c); got != test.want {
				t.Errorf("CheckBlocking: want %v, got %v", test.want, got)
			}
			// The sibling CheckPriority scans all devices too and, for these
			// inputs, agrees with CheckBlocking.
			if got := CheckPriority(test.sw, test.priority, c); got != test.want {
				t.Errorf("CheckPriority: want %v, got %v", test.want, got)
			}
		})
	}
}

func (s *stubInfo) GetComputeState() int32        { return 0 }
func (s *stubInfo) SetComputeState(int32)         {}
func (s *stubInfo) GetLastLaunchNs() uint64       { return 0 }
func (s *stubInfo) GetDeviceSmLimit(int) uint64   { return 0 }
func (s *stubInfo) GetDynamicSmLimit(int) uint64  { return 0 }
func (s *stubInfo) SetDynamicSmLimit(int, uint64) {}
func (s *stubInfo) GetFloorSmLimit(int) uint64    { return 0 }

func TestClampTickMs(t *testing.T) {
	if got := clampTickMs("HAMI_COMPUTE_LIGHT_TICK_MS_UNSET_TEST", 200, 50, 0); got != 200*time.Millisecond {
		t.Fatalf("default light tick=%v, want 200ms", got)
	}
	t.Setenv("HAMI_COMPUTE_LIGHT_TICK_MS_CLAMP_LOW", "10")
	if got := clampTickMs("HAMI_COMPUTE_LIGHT_TICK_MS_CLAMP_LOW", 200, 50, 0); got != 50*time.Millisecond {
		t.Fatalf("below-min clamp=%v, want 50ms", got)
	}
	t.Setenv("HAMI_COMPUTE_HEAVY_TICK_MS_CLAMP_HIGH", "99999")
	if got := clampTickMs("HAMI_COMPUTE_HEAVY_TICK_MS_CLAMP_HIGH", 5000, 1000, 10000); got != 10*time.Second {
		t.Fatalf("above-max clamp=%v, want 10s", got)
	}
}

func TestObservePriorityFeedback_DecrementsRecentKernel(t *testing.T) {
	info := &priorityStub{uuids: []string{"gpu-0"}, recentKernel: 2}
	c := &nvidia.ContainerUsage{PodUID: "x", ContainerName: "x", Info: info}
	observePriorityFeedback(map[string]*nvidia.ContainerUsage{"x": c})
	if info.setRecentKernelN == 0 || info.recentKernel != 1 {
		t.Fatalf("heavy path should decrement recent_kernel to 1, got calls=%d value=%d",
			info.setRecentKernelN, info.recentKernel)
	}
}

// priorityStub is a UsageInfo that records recent_kernel writes.
type priorityStub struct {
	uuids            []string
	recentKernel     int32
	setRecentKernelN int
}

func (s *priorityStub) DeviceMax() int { return stubDeviceMax }
func (s *priorityStub) DeviceNum() int { return len(s.uuids) }
func (s *priorityStub) DeviceUUID(i int) string {
	if i < len(s.uuids) {
		return s.uuids[i]
	}
	return ""
}
func (s *priorityStub) DeviceMemoryContextSize(int) uint64 { return 0 }
func (s *priorityStub) DeviceMemoryModuleSize(int) uint64  { return 0 }
func (s *priorityStub) DeviceMemoryBufferSize(int) uint64  { return 0 }
func (s *priorityStub) DeviceMemoryOffset(int) uint64      { return 0 }
func (s *priorityStub) DeviceMemoryTotal(int) uint64       { return 0 }
func (s *priorityStub) DeviceSmUtil(int) uint64            { return 0 }
func (s *priorityStub) SetDeviceSmLimit(uint64)            {}
func (s *priorityStub) IsValidUUID(i int) bool             { return i < len(s.uuids) }
func (s *priorityStub) DeviceMemoryLimit(int) uint64       { return 0 }
func (s *priorityStub) SetDeviceMemoryLimit(uint64)        {}
func (s *priorityStub) LastKernelTime() int64              { return 0 }
func (s *priorityStub) GetPriority() int                   { return 0 }
func (s *priorityStub) GetRecentKernel() int32             { return s.recentKernel }
func (s *priorityStub) SetRecentKernel(v int32) {
	s.setRecentKernelN++
	s.recentKernel = v
}
func (s *priorityStub) GetUtilizationSwitch() int32   { return 0 }
func (s *priorityStub) SetUtilizationSwitch(int32)    {}
func (s *priorityStub) GetComputeState() int32        { return 0 }
func (s *priorityStub) SetComputeState(int32)         {}
func (s *priorityStub) GetLastLaunchNs() uint64       { return 0 }
func (s *priorityStub) GetDeviceSmLimit(int) uint64   { return 0 }
func (s *priorityStub) GetDynamicSmLimit(int) uint64  { return 0 }
func (s *priorityStub) SetDynamicSmLimit(int, uint64) {}
func (s *priorityStub) GetFloorSmLimit(int) uint64    { return 0 }
