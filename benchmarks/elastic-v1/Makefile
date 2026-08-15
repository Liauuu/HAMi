# elastic-v1 step-8 harness
ROOT := $(abspath .)
HAMI_ROOT := $(abspath ../..)
BIN := $(ROOT)/bin

.PHONY: build build-sm-burn build-feedback-lite clean dry-run help

help:
	@echo "Targets:"
	@echo "  make build              # sm_burn + feedback-lite"
	@echo "  make dry-run            # no GPU: build + summarize self-check"
	@echo "On GPU VM (after building matching libvgpu):"
	@echo "  VARIANT=v1 SCENARIO=s2 ./scripts/run_scenario.sh"
	@echo "  ./scripts/run_compare.sh s2"

build: build-sm-burn build-feedback-lite

build-sm-burn: $(BIN)/sm_burn

$(BIN)/sm_burn: workload/sm_burn.cu
	mkdir -p $(BIN)
	nvcc -O2 -o $@ $<

build-feedback-lite: $(BIN)/feedback-lite

$(BIN)/feedback-lite: cmd/feedback-lite/main.go
	mkdir -p $(BIN)
	cd $(HAMI_ROOT) && go build -o $(BIN)/feedback-lite ./benchmarks/elastic-v1/cmd/feedback-lite

clean:
	rm -rf $(BIN) .run

dry-run: build-feedback-lite
	python3 scripts/summarize.py scripts/testdata/sample_run > /tmp/elastic-summary.json
	@echo "dry-run ok: /tmp/elastic-summary.json"
	@head -n 20 /tmp/elastic-summary.json
