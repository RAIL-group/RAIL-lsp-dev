MAKEFLAGS += --no-print-directory
## ==== Core Arguments and Parameters ====
MAJOR ?= 0
MINOR ?= 1
VERSION = $(MAJOR).$(MINOR)
APP_NAME ?= rail-plan-uncertain
NUM_BUILD_CORES ?= $(shell grep -c ^processor /proc/cpuinfo)

# Handle Optional GPU
USE_GPU ?= true
ifeq ($(USE_GPU),true)
	DOCKER_GPU_ARG = --gpus all
endif

# Paths and Key File Names
EXPERIMENT_NAME ?= dbg
RAIL_SIM_VERSION ?= 1.0.0
RAIL_SIM_DIR ?= $(shell pwd)/resources/unity/
RAIL_SIM_BASENAME ?= rail_sim

# Docker args
DISPLAY ?= :0.0
DATA_BASE_DIR ?= $(shell pwd)/data
RESOURCES_BASE_DIR ?= $(shell pwd)/resources
XPASSTHROUGH ?= false
DOCKER_FILE_DIR = "."
DOCKERFILE = ${DOCKER_FILE_DIR}/Dockerfile
IMAGE_NAME = ${APP_NAME}
DOCKER_CORE_VOLUMES = \
	--env XPASSTHROUGH=$(XPASSTHROUGH) \
	--env DISPLAY=$(DISPLAY) \
	--env USE_GPU=$(USE_GPU) \
	$(DOCKER_GPU_ARG) \
	--volume="$(RAIL_SIM_DIR)/v$(RAIL_SIM_VERSION):/unity/:ro" \
	--volume="$(DATA_BASE_DIR):/data/:rw" \
	--volume="$(RESOURCES_BASE_DIR):/resources/:rw" \
	--volume="$(RESOURCES_BASE_DIR)/notebooks:/notebooks/:rw" \
	--volume="$(RESOURCES_BASE_DIR)/ai2thor:/root/.ai2thor/:rw" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
DOCKER_BASE = docker run --platform linux/amd64 --init --ipc=host --rm \
	$(DOCKER_ARGS) $(DOCKER_CORE_VOLUMES) \
	${IMAGE_NAME}:${VERSION}
DOCKER_PYTHON = $(DOCKER_BASE) uv run python
DOCKER_JUPYTER = $(DOCKER_BASE) uv run --with jupyter jupyter

## ==== Helpers for setting up the environment ====
define arg_check_unity
	@echo "DEPRICATION: arg_check_unity no longer required"
endef

define xhost_activate
	@echo "Enabling local xhost sharing:"
	@echo "  Display: $(DISPLAY)"
	@-DISPLAY=$(DISPLAY) xhost  +
	@-xhost  +
endef

arg-check-unity:
	$(call arg_check_unity)
arg-check-data:
	@[ "${DATA_BASE_DIR}" ] && true || \
		( echo "ERROR: Environment variable 'DATA_BASE_DIR' must be set." 1>&2; exit 1 )
xhost-activate:
	$(call xhost_activate)

unity-simulator-full-name = $(RAIL_SIM_DIR)/v$(RAIL_SIM_VERSION)/$(RAIL_SIM_BASENAME).x86_64
.PHONY: unity-simulator
unity-simulator: $(unity-simulator-full-name)
UNITY_BASENAME ?= $(RAIL_SIM_BASENAME)
$(unity-simulator-full-name):
	@echo "Downloading the Unity simulator"
	@mkdir -p $(RAIL_SIM_DIR)
	cd $(RAIL_SIM_DIR) \
		&& curl -OL https://github.com/RAIL-group/RAIL-group-simulator/releases/download/v$(RAIL_SIM_VERSION)/rail_sim.zip \
		&& unzip rail_sim.zip \
		&& rm rail_sim.zip
	@echo "Unity simulator downloaded and unpacked."

## ==== Build targets ====
.PHONY: build
build: $(unity-simulator-full-name)
	@echo "Building the Docker container"
	@DOCKER_CLI_HINTS=false docker build -t ${IMAGE_NAME}:${VERSION} \
		--platform linux/amd64 \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		-f ./${DOCKERFILE} .

## ==== Running tests & cleanup ====
.PHONY: test
test: DOCKER_ARGS ?= -it
test: PYTEST_FILTER ?= "py"
test: build
	@$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/test_logs
	@$(DOCKER_PYTHON) \
		-m pytest -vk $(PYTEST_FILTER) \
		-rsx \
		--full-trace \
		--ignore-glob=**/pybind11/* \
		--ignore-glob=**/scratch/* \
		--html=/data/test_logs/report.html \
		--xpassthrough=$(XPASSTHROUGH) \
		--unity-path=/unity/$(UNITY_BASENAME).x86_64 \
		$(TEST_ADDITIONAL_ARGS) \
		/modules/

flake8: DOCKER_ARGS = -it --workdir /modules
flake8: build
	@echo "Running flake8 format checker..."
	@$(DOCKER_BASE) flake8
	@echo "... no formatting issues discovered."

## ==== Other Targets ====
.PHONY: kill
kill:
	@echo "Closing all running docker containers:"
	@docker kill $(shell docker ps -q --filter ancestor=${IMAGE_NAME}:${VERSION})

.PHONY: shell
shell: DOCKER_ARGS ?= -it
shell:
	@$(DOCKER_BASE) bash

.PHONY: notebook
notebook: DOCKER_ARGS=-it -p 8888:8888
notebook: build
	@$(DOCKER_JUPYTER) notebook \
		--notebook-dir=/notebooks \
		--no-browser --allow-root \
		--ip 0.0.0.0 \
		--NotebookApp.token='' --NotebookApp.password=''

tensorboard:
	@docker run -it \
		-p 0.0.0.0:6006:6006 \
		$(DOCKER_CORE_VOLUMES) \
		$(IMAGE_NAME):$(VERSION) tensorboard \
		--logdir /data --host 0.0.0.0

# ==== Includes ====
include modules/lsp/Makefile.mk
include modules/procthor/Makefile.mk
include modules/taskplan/Makefile.mk
include modules/object_search/Makefile.mk
