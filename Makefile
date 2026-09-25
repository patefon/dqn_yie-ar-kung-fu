# Yie Ar Kung-Fu RL -- developer tasks.
#
# Training runs under WSL2 (stable-retro has no native Windows build), but the
# source lives on the Windows filesystem so you can edit it from either side.
# The venv deliberately lives on the WSL-native filesystem: a venv on /mnt/c is
# several times slower to import from.

# Name of the training run. train/watch/eval all key off this, so the viewer
# always points at the run you are actually training.
NAME   ?= run1
RUN    ?= runs/$(NAME)
CONFIG ?= configs/default.yaml
PORT   ?= 8080
DEMO_PORT ?= 8081
# Viewer envs are extra emulator processes alongside the training envs.
# Lower this if the viewer is competing with training for CPU.
WATCH_ENVS ?= 6
# Extra flags forwarded to watch/demo, e.g. make watch NAME=run1 ARGS=--endless
ARGS ?=

VENV ?= $(HOME)/.venvs/kungfu-rl
PY   := $(VENV)/bin/python
UV   := $(HOME)/.local/bin/uv

.PHONY: help setup install rom calibrate atlas check train go smoke eval watch demo test lint fmt config tb clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install the project (editable, CUDA torch)
	$(UV) venv --python 3.12 $(VENV)
	$(UV) pip install --python $(PY) -e ".[dev,baselines]" --torch-backend=cu126

rom: ## Register the ROM in roms/ as a stable-retro integration + savestate
	$(PY) -m kungfu.emulator.integration

calibrate: ## Harvest HUD digit glyphs (then label out/glyphs/labels.json)
	$(PY) -m tools.calibrate_vision --command dump --frames 300
	$(PY) -m tools.calibrate_vision --command harvest --frames 4000

atlas: ## Build the digit atlas from the labelled glyphs
	$(PY) -m tools.calibrate_vision --command build-atlas

check: ## Verify the vision pipeline can read the HUD
	$(PY) -m tools.calibrate_vision --command check --frames 2000

setup: rom calibrate atlas check ## Full first-time setup after dropping in a ROM

smoke: ## 40k-step training run to prove the pipeline end to end (~3 min)
	$(PY) -m kungfu.train --config configs/smoke.yaml

train: ## Full training run: make train NAME=run1 [ARGS=--resume runs/old/final.pt]
	$(PY) -m kungfu.train --config $(CONFIG) --run-name $(NAME) $(ARGS)

go: ## Train AND watch in one command: make go NAME=run1 [ARGS=--resume ...]
	@mkdir -p $(dir $(RUN))
	@echo "viewer      -> http://localhost:$(PORT)"
	@echo "viewer log  -> $(RUN)-viewer.log"
	@echo "tensorboard -> make tb   (separate terminal)"
	@echo "Ctrl+C stops both."
	@echo ""
	@$(PY) -m tools.watch --run $(RUN) --config $(CONFIG) \
	        --envs $(WATCH_ENVS) --cols 3 --port $(PORT) > $(RUN)-viewer.log 2>&1 & \
	  VIEWER=$$!; \
	  trap 'kill $$VIEWER 2>/dev/null; echo; echo "viewer stopped"' EXIT INT TERM; \
	  $(PY) -m kungfu.train --config $(CONFIG) --run-name $(NAME) $(ARGS)

eval: ## Evaluate the finished run: make eval NAME=run1 [ARGS=--endless]
	$(PY) -m kungfu.evaluate --checkpoint $(RUN)/final.pt --episodes 10 --video out/best.mp4 $(ARGS)

watch: ## Live mosaic against an existing run: make watch NAME=run1 [ARGS=--endless]
	$(PY) -m tools.watch --run $(RUN) --envs $(WATCH_ENVS) --cols 3 --port $(PORT) $(ARGS)

demo: ## Record MP4 (sound + overlays) and watch it live: make demo NAME=run1 [ARGS=--endless]
	@echo "live -> http://localhost:$(DEMO_PORT)"
	$(PY) -m tools.demo --checkpoint $(RUN)/final.pt --out out/$(NAME)-demo.mp4 \
	    --endless --episodes 1 --live --realtime --port $(DEMO_PORT) $(ARGS)

tb: ## Launch TensorBoard on the runs directory
	$(PY) -m tensorboard.main --logdir runs --bind_all

test: ## Run the test suite (no ROM required)
	$(PY) -m pytest -q

lint: ## Lint
	$(PY) -m ruff check src tools tests

fmt: ## Auto-format and fix
	$(PY) -m ruff check --fix src tools tests
	$(PY) -m ruff format src tools tests

config: ## Regenerate configs/default.yaml from the code defaults
	$(PY) -c "from kungfu.config import Config; Config().dump('configs/default.yaml')"

clean: ## Remove caches and build artefacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.py[co]' -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
