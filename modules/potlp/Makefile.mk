POTLP_BASENAME = potlp
POTLP_UNITY_BASENAME ?= $(UNITY_BASENAME)

POTLP_CORE_ARGS ?= --unity_path /unity/$(POTLP_UNITY_BASENAME).x86_64 \
		--save_dir /data/$(POTLP_BASENAME)/$(EXPERIMENT_NAME) \
		--limit_frontiers 7 \
		--iterations 40000 \

POTLP_SEED_START = 2000
POTLP_NUM_EXPERIMENTS = 100
define potlp_get_seeds
	$(shell seq $(POTLP_SEED_START) $$(($(POTLP_SEED_START)+$(POTLP_NUM_EXPERIMENTS) - 1)))
endef

all-targets = $(foreach seed, $(call potlp_get_seeds), \
					$(DATA_BASE_DIR)/$(POTLP_BASENAME)/$(EXPERIMENT_NAME)/run_$(seed).txt)

.PHONY: run-potlp
run-potlp: $(all-targets)
$(all-targets): seed = $(shell echo $@ | grep -oE 'run_[0-9]+' | cut -d'_' -f2)
$(all-targets): map_type = $(ENV)
$(all-targets):
	@echo $(network_file)
	$(call xhost_activate)
	@echo "Current experiment: Planner = $(planner) | Map = $(map_type) | Seed = $(seed) | Num robots = $(num_robots)"
	@mkdir -p $(DATA_BASE_DIR)/$(POTLP_BASENAME)/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m modules.potlp.potlp.scripts.simulation_$(planner) \
		$(POTLP_CORE_ARGS) \
		--seed $(seed) \
		--num_robots $(num_robots) \
		--map_type $(map_type) \
		--network_file $(network_file)

.PHONY: potlp-maze
potlp-maze: $(lsp-maze-train-file)
potlp-maze:
	@$(MAKE) run-potlp ENV=maze \
		NETWORK_FILE=/data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt

.PHONY: potlp-office
potlp-office: $(lsp-office-train-file)
potlp-office:
	@$(MAKE) run-potlp ENV=office2 \
		NETWORK_FILE=/data/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt

# Visualize different planner for different seed and number of robots in office environment
.PHONY: visualize
visualize: DOCKER_ARGS ?= -it
visualize: xhost-activate arg-check-data
	@mkdir -p $(DATA_BASE_DIR)/$(POTLP_BASENAME)/visualize/
	@$(DOCKER_PYTHON) -m modules.potlp.potlp.scripts.simulation_$(PLANNER) \
		$(POTLP_CORE_ARGS) \
	   	--save_dir data/$(POTLP_BASENAME)/visualize/ \
		--network_file /data/$(BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt \
		--seed $(SEED) \
		--map_type $(MAP) \
		--do_plot True \
		--num_robots $(NUM_ROBOTS) \

.PHONY: visualize-office
visualize-office: PLANNER = optimistic
visualize-office: SEED = 2001
visualize-office: NUM_ROBOTS = 2
visualize-office:
	@$(MAKE) visualize BASENAME=$(LSP_OFFICE_BASENAME) \
		PLANNER=$(PLANNER) \
		SEED=$(SEED) \
		NUM_ROBOTS=$(NUM_ROBOTS) \
		XPASSTHROUGH=true \
		MAP=office2

.PHONY: visualize-maze
visualize-maze: PLANNER = optimistic
visualize-maze: SEED = 2001
visualize-maze: NUM_ROBOTS = 2
visualize-maze:
	@$(MAKE) visualize BASENAME=$(LSP_MAZE_BASENAME) \
		PLANNER=$(PLANNER) \
		SEED=$(SEED) \
		NUM_ROBOTS=$(NUM_ROBOTS) \
		XPASSTHROUGH=true \
		MAP=maze
