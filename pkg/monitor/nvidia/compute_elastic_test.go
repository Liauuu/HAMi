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

package nvidia

import (
	"testing"
	"time"
)

const stubDeviceMax = 16

type stubInfo struct {
	priority         int
	uuids            []string
	computeState     int32
	lastLaunchNs     uint64
	setStateCalls    int
	lastSetStateTo   int32
	recentKernel     int32
	setRecentKernelN int
	floorSmLimit     [stubDeviceMax]uint64
	smLimit          [stubDeviceMax]uint64
	dynamicSmLimit   [stubDeviceMax]uint64
}

func (s *stubInfo) DeviceMax() int { return stubDeviceMax }
func (s *stubInfo) DeviceNum() int { return len(s.uuids) }
func (s *stubInfo) DeviceUUID(i int) string {
	if i < len(s.uuids) {
		return s.uuids[i]
	}
	return ""
}
func (s *stubInfo) DeviceMemoryContextSize(int) uint64 { return 0 }
func (s *stubInfo) DeviceMemoryModuleSize(int) uint64  { return 0 }
func (s *stubInfo) DeviceMemoryBufferSize(int) uint64  { return 0 }
func (s *stubInfo) DeviceMemoryOffset(int) uint64      { return 0 }
func (s *stubInfo) DeviceMemoryTotal(int) uint64       { return 0 }
func (s *stubInfo) DeviceSmUtil(int) uint64            { return 0 }
func (s *stubInfo) SetDeviceSmLimit(uint64)            {}
func (s *stubInfo) IsValidUUID(i int) bool             { return i < len(s.uuids) }
func (s *stubInfo) DeviceMemoryLimit(int) uint64       { return 0 }
func (s *stubInfo) SetDeviceMemoryLimit(uint64)        {}
func (s *stubInfo) LastKernelTime() int64              { return 0 }
func (s *stubInfo) GetPriority() int                   { return s.priority }
func (s *stubInfo) GetRecentKernel() int32             { return s.recentKernel }
func (s *stubInfo) SetRecentKernel(v int32) {
	s.setRecentKernelN++
	s.recentKernel = v
}
func (s *stubInfo) GetUtilizationSwitch() int32 { return 0 }
func (s *stubInfo) SetUtilizationSwitch(int32)  {}
func (s *stubInfo) GetComputeState() int32      { return s.computeState }
func (s *stubInfo) SetComputeState(v int32) {
	s.setStateCalls++
	s.lastSetStateTo = v
	s.computeState = v
}
func (s *stubInfo) GetLastLaunchNs() uint64 { return s.lastLaunchNs }
func (s *stubInfo) GetDeviceSmLimit(idx int) uint64 {
	if idx < 0 || idx >= stubDeviceMax {
		return 0
	}
	return s.smLimit[idx]
}
func (s *stubInfo) GetFloorSmLimit(idx int) uint64 {
	if idx < 0 || idx >= stubDeviceMax {
		return 0
	}
	return s.floorSmLimit[idx]
}
func (s *stubInfo) GetDynamicSmLimit(idx int) uint64 {
	if idx < 0 || idx >= stubDeviceMax {
		return 0
	}
	return s.dynamicSmLimit[idx]
}
func (s *stubInfo) SetDynamicSmLimit(idx int, v uint64) {
	if idx < 0 || idx >= stubDeviceMax {
		return
	}
	s.dynamicSmLimit[idx] = v
}

func newGPUContainer(name, uuid string, state int32, floor, dynamic uint64) (*ContainerUsage, *stubInfo) {
	info := &stubInfo{
		uuids:        []string{uuid},
		computeState: state,
	}
	info.floorSmLimit[0] = floor
	info.smLimit[0] = floor
	info.dynamicSmLimit[0] = dynamic
	return &ContainerUsage{PodUID: name, ContainerName: name, Info: info}, info
}

func TestUpdateComputeState_HysteresisTable(t *testing.T) {
	const (
		candidateNs = uint64(1000 * time.Millisecond)
		idleNs      = uint64(2000 * time.Millisecond)
		now         = uint64(10_000_000_000)
	)

	tests := []struct {
		name     string
		last     uint64
		cur      int32
		want     int32
		wantSets int
	}{
		{
			name:     "recent launch stays/becomes ACTIVE",
			last:     now - 100*uint64(time.Millisecond),
			cur:      ComputeStateIdle,
			want:     ComputeStateActive,
			wantSets: 1,
		},
		{
			name:     "between candidate and idle -> IDLE_CANDIDATE",
			last:     now - 1500*uint64(time.Millisecond),
			cur:      ComputeStateActive,
			want:     ComputeStateIdleCandidate,
			wantSets: 1,
		},
		{
			name:     "past idle threshold -> IDLE",
			last:     now - 2500*uint64(time.Millisecond),
			cur:      ComputeStateIdleCandidate,
			want:     ComputeStateIdle,
			wantSets: 1,
		},
		{
			name:     "exactly candidate boundary is IDLE_CANDIDATE (idleFor < candidate is ACTIVE)",
			last:     now - candidateNs,
			cur:      ComputeStateActive,
			want:     ComputeStateIdleCandidate,
			wantSets: 1,
		},
		{
			name:     "exactly idle boundary is IDLE",
			last:     now - idleNs,
			cur:      ComputeStateIdleCandidate,
			want:     ComputeStateIdle,
			wantSets: 1,
		},
		{
			name:     "never launched (last=0) classified IDLE",
			last:     0,
			cur:      ComputeStateActive,
			want:     ComputeStateIdle,
			wantSets: 1,
		},
		{
			name:     "clock skew now<last prefers ACTIVE",
			last:     now + 5*uint64(time.Millisecond),
			cur:      ComputeStateIdle,
			want:     ComputeStateActive,
			wantSets: 1,
		},
		{
			name:     "already ACTIVE and still fresh does not rewrite",
			last:     now - 10*uint64(time.Millisecond),
			cur:      ComputeStateActive,
			want:     ComputeStateActive,
			wantSets: 0,
		},
		{
			name:     "already IDLE and still idle does not rewrite",
			last:     now - 5*uint64(time.Second),
			cur:      ComputeStateIdle,
			want:     ComputeStateIdle,
			wantSets: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			info := &stubInfo{
				computeState: tc.cur,
				lastLaunchNs: tc.last,
			}
			c := &ContainerUsage{Info: info}
			UpdateComputeState(c, now, candidateNs, idleNs)

			if info.computeState != tc.want {
				t.Fatalf("compute_state=%d, want %d", info.computeState, tc.want)
			}
			if info.setStateCalls != tc.wantSets {
				t.Fatalf("SetComputeState calls=%d, want %d (lastSet=%d)",
					info.setStateCalls, tc.wantSets, info.lastSetStateTo)
			}
			if tc.wantSets > 0 && info.lastSetStateTo != tc.want {
				t.Fatalf("last SetComputeState arg=%d, want %d", info.lastSetStateTo, tc.want)
			}
		})
	}
}

func TestUpdateComputeState_ActiveCandidateIdleActiveReplay(t *testing.T) {
	const (
		candidateNs = uint64(1000 * time.Millisecond)
		idleNs      = uint64(2000 * time.Millisecond)
		now         = uint64(20_000_000_000)
	)

	info := &stubInfo{
		computeState: ComputeStateUnset,
		lastLaunchNs: now,
	}
	c := &ContainerUsage{PodUID: "pod", ContainerName: "ctr", Info: info}

	UpdateComputeState(c, now, candidateNs, idleNs)
	if info.computeState != ComputeStateActive {
		t.Fatalf("after fresh launch: state=%d, want ACTIVE(%d)", info.computeState, ComputeStateActive)
	}

	info.lastLaunchNs = now - 1500*uint64(time.Millisecond)
	UpdateComputeState(c, now, candidateNs, idleNs)
	if info.computeState != ComputeStateIdleCandidate {
		t.Fatalf("after 1.5s idle: state=%d, want IDLE_CANDIDATE(%d)", info.computeState, ComputeStateIdleCandidate)
	}

	info.lastLaunchNs = now - 2500*uint64(time.Millisecond)
	UpdateComputeState(c, now, candidateNs, idleNs)
	if info.computeState != ComputeStateIdle {
		t.Fatalf("after 2.5s idle: state=%d, want IDLE(%d)", info.computeState, ComputeStateIdle)
	}

	info.lastLaunchNs = now
	UpdateComputeState(c, now, candidateNs, idleNs)
	if info.computeState != ComputeStateActive {
		t.Fatalf("after relaunch: state=%d, want ACTIVE(%d)", info.computeState, ComputeStateActive)
	}
}

func TestUpdateComputeState_ThresholdSanity(t *testing.T) {
	const now = uint64(5_000_000_000)
	info := &stubInfo{
		computeState: ComputeStateActive,
		lastLaunchNs: now - 1500*uint64(time.Millisecond),
	}
	c := &ContainerUsage{Info: info}

	UpdateComputeState(c, now, 2000*uint64(time.Millisecond), 1000*uint64(time.Millisecond))
	if info.computeState != ComputeStateActive {
		t.Fatalf("with clamped thresholds, 1.5s idle should stay ACTIVE, got %d", info.computeState)
	}
}

func TestRedistributeHeadroom_IdleLendsToActiveWithCap(t *testing.T) {
	a, aInfo := newGPUContainer("A", "gpu-0", ComputeStateIdle, 40, 40)
	b, bInfo := newGPUContainer("B", "gpu-0", ComputeStateActive, 40, 40)

	RedistributeHeadroom(map[string]*ContainerUsage{"A": a, "B": b}, 85, 0, 100)

	if aInfo.dynamicSmLimit[0] != 40 {
		t.Fatalf("IDLE A dynamic=%d, want floor 40", aInfo.dynamicSmLimit[0])
	}
	if bInfo.dynamicSmLimit[0] != 60 {
		t.Fatalf("ACTIVE B dynamic=%d, want 60 after Σ<=100 clamp (raw would be 80)", bInfo.dynamicSmLimit[0])
	}
	if aInfo.dynamicSmLimit[0]+bInfo.dynamicSmLimit[0] > 100 {
		t.Fatalf("sum=%d exceeds capacity", aInfo.dynamicSmLimit[0]+bInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_CandidateDoesNotContribute(t *testing.T) {
	a, aInfo := newGPUContainer("A", "gpu-0", ComputeStateIdleCandidate, 40, 40)
	b, bInfo := newGPUContainer("B", "gpu-0", ComputeStateActive, 60, 60)

	RedistributeHeadroom(map[string]*ContainerUsage{"A": a, "B": b}, 85, 0, 100)

	if aInfo.dynamicSmLimit[0] != 40 {
		t.Fatalf("CANDIDATE A dynamic=%d, want floor 40", aInfo.dynamicSmLimit[0])
	}
	if bInfo.dynamicSmLimit[0] != 60 {
		t.Fatalf("ACTIVE B dynamic=%d, want unchanged floor 60 (no idle headroom)", bInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_BothActiveNoBorrow(t *testing.T) {
	a, aInfo := newGPUContainer("A", "gpu-0", ComputeStateActive, 40, 40)
	b, bInfo := newGPUContainer("B", "gpu-0", ComputeStateActive, 60, 60)

	RedistributeHeadroom(map[string]*ContainerUsage{"A": a, "B": b}, 85, 0, 100)

	if aInfo.dynamicSmLimit[0] != 40 || bInfo.dynamicSmLimit[0] != 60 {
		t.Fatalf("both ACTIVE should keep floors, got A=%d B=%d", aInfo.dynamicSmLimit[0], bInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_EqualShareAmongActivesThenClamp(t *testing.T) {
	idle, idleInfo := newGPUContainer("idle", "gpu-0", ComputeStateIdle, 40, 40)
	a1, a1Info := newGPUContainer("a1", "gpu-0", ComputeStateActive, 30, 30)
	a2, a2Info := newGPUContainer("a2", "gpu-0", ComputeStateActive, 30, 30)

	RedistributeHeadroom(map[string]*ContainerUsage{
		"idle": idle, "a1": a1, "a2": a2,
	}, 85, 0, 100)

	if idleInfo.dynamicSmLimit[0] != 40 {
		t.Fatalf("idle dynamic=%d, want 40", idleInfo.dynamicSmLimit[0])
	}
	if a1Info.dynamicSmLimit[0] != 30 || a2Info.dynamicSmLimit[0] != 30 {
		t.Fatalf("actives should clamp back to floors when card already full, got %d/%d",
			a1Info.dynamicSmLimit[0], a2Info.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_BurstCapAppliedBeforeClamp(t *testing.T) {
	idle, idleInfo := newGPUContainer("idle", "gpu-0", ComputeStateIdle, 50, 50)
	active, activeInfo := newGPUContainer("busy", "gpu-0", ComputeStateActive, 20, 20)

	RedistributeHeadroom(map[string]*ContainerUsage{
		"idle": idle, "busy": active,
	}, 65, 0, 100)

	if idleInfo.dynamicSmLimit[0] != 50 {
		t.Fatalf("idle=%d, want 50", idleInfo.dynamicSmLimit[0])
	}
	if activeInfo.dynamicSmLimit[0] != 50 {
		t.Fatalf("active=%d, want 50 after cap+clamp", activeInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_ImmediateReclaim(t *testing.T) {
	a, aInfo := newGPUContainer("A", "gpu-0", ComputeStateActive, 50, 50)
	b, bInfo := newGPUContainer("B", "gpu-0", ComputeStateActive, 50, 80)

	RedistributeHeadroom(map[string]*ContainerUsage{"A": a, "B": b}, 85, 10, 100)

	if aInfo.dynamicSmLimit[0] != 50 {
		t.Fatalf("A=%d, want 50", aInfo.dynamicSmLimit[0])
	}
	if bInfo.dynamicSmLimit[0] != 50 {
		t.Fatalf("B reclaim should be immediate to floor 50, got %d (grantStep must not delay reductions)", bInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_MildGrant(t *testing.T) {
	idle, _ := newGPUContainer("idle", "gpu-0", ComputeStateIdle, 40, 40)
	busy, busyInfo := newGPUContainer("busy", "gpu-0", ComputeStateActive, 40, 40)

	RedistributeHeadroom(map[string]*ContainerUsage{"idle": idle, "busy": busy}, 85, 10, 100)

	if busyInfo.dynamicSmLimit[0] != 50 {
		t.Fatalf("mild grant: busy=%d, want 50 (floor 40 + step 10 toward target 60)", busyInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_AllIdleNoActiveNoPanic(t *testing.T) {
	a, aInfo := newGPUContainer("A", "gpu-0", ComputeStateIdle, 40, 40)
	b, bInfo := newGPUContainer("B", "gpu-0", ComputeStateIdle, 60, 60)

	RedistributeHeadroom(map[string]*ContainerUsage{"A": a, "B": b}, 85, 0, 100)

	if aInfo.dynamicSmLimit[0] != 40 || bInfo.dynamicSmLimit[0] != 60 {
		t.Fatalf("all IDLE should keep floors, got %d/%d", aInfo.dynamicSmLimit[0], bInfo.dynamicSmLimit[0])
	}
}

func TestRedistributeHeadroom_DifferentGPUsIsolated(t *testing.T) {
	idle0, _ := newGPUContainer("idle0", "gpu-0", ComputeStateIdle, 40, 40)
	busy0, busy0Info := newGPUContainer("busy0", "gpu-0", ComputeStateActive, 40, 40)
	busy1, busy1Info := newGPUContainer("busy1", "gpu-1", ComputeStateActive, 40, 40)

	RedistributeHeadroom(map[string]*ContainerUsage{
		"idle0": idle0, "busy0": busy0, "busy1": busy1,
	}, 85, 0, 100)

	if busy0Info.dynamicSmLimit[0] != 60 {
		t.Fatalf("gpu-0 busy=%d, want 60", busy0Info.dynamicSmLimit[0])
	}
	if busy1Info.dynamicSmLimit[0] != 40 {
		t.Fatalf("gpu-1 busy should stay floor 40, got %d", busy1Info.dynamicSmLimit[0])
	}
}

func TestApplyDynamicLimit_NeverBelowFloor(t *testing.T) {
	c, info := newGPUContainer("x", "gpu-0", ComputeStateActive, 40, 40)
	ApplyDynamicLimitFor(c, 0, 10, 0)
	if info.dynamicSmLimit[0] != 40 {
		t.Fatalf("dynamic=%d, want floor 40", info.dynamicSmLimit[0])
	}
}

func TestObserveComputePolicy_DoesNotTouchRecentKernel(t *testing.T) {
	a, aInfo := newGPUContainer("idle", "gpu-0", ComputeStateIdle, 40, 40)
	b, bInfo := newGPUContainer("busy", "gpu-0", ComputeStateActive, 40, 40)
	aInfo.lastLaunchNs = 0
	aInfo.recentKernel = 2
	bInfo.lastLaunchNs = MonotonicNowNs()
	bInfo.recentKernel = 2

	ObserveComputePolicy(map[string]*ContainerUsage{"idle": a, "busy": b}, ComputePolicyConfig{
		IdleCandidateNs: uint64(time.Second),
		IdleThresholdNs: 2 * uint64(time.Second),
		BurstCap:        85,
		GrantStep:       0,
		Capacity:        100,
	})

	if aInfo.setRecentKernelN != 0 || bInfo.setRecentKernelN != 0 {
		t.Fatalf("light path must not SetRecentKernel (idle=%d busy=%d)",
			aInfo.setRecentKernelN, bInfo.setRecentKernelN)
	}
	if aInfo.recentKernel != 2 || bInfo.recentKernel != 2 {
		t.Fatalf("recent_kernel mutated on light path: idle=%d busy=%d", aInfo.recentKernel, bInfo.recentKernel)
	}
	if bInfo.dynamicSmLimit[0] <= 40 {
		t.Fatalf("ACTIVE busy should receive headroom, dynamic=%d", bInfo.dynamicSmLimit[0])
	}
	if aInfo.dynamicSmLimit[0] != 40 {
		t.Fatalf("IDLE should stay at floor, dynamic=%d", aInfo.dynamicSmLimit[0])
	}
}
