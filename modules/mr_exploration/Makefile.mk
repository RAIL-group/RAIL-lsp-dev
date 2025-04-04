help::
	@echo ""
	@echo ""
	@echo ""
	@echo ""

MRLSP_BASENAME = mr_exploration
MRLSP_UNITY_BASENAME ?= $(RAIL_SIM_BASENAME)
MRLSP_CORE_ARGS ?= --unity_path /unity/$(MRLSP_UNITY_BASENAME).x86_64 \
		--map_type office2 \
		--base_resolution 0.5 \
		--inflation_radius_m 0.75 \
		--num_primitives 32 \
		--laser_max_range_m 18 \
		--network_file /data/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt \

# Visualize MRLSP and optimistic planner
.PHONY: mrlsp-vis
mrlsp-vis: DOCKER_ARGS ?= -it
mrlsp-vis: xhost-activate arg-check-data
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_BASENAME)/optimistic
	@$(DOCKER_PYTHON) -m mrlsp.scripts.vis_planners \
		$(MRLSP_CORE_ARGS) \
	   	--save_dir data/$(MRLSP_BASENAME)/optimistic \
		--network_file data/training_logs/VisLSPOriented_laptop_trained.pt \
		--num_robots 2 \
		--planner mrlsp \
		--seed 2020

# Run experiments for MRLSP
MRLSP_SEED_START = 2000
MRLSP_NUM_EXPERIMENTS = 200
MRLSP_EXPERIMENT_NAME = mrlsp_eval
define mrlsp_get_seeds
	$(shell seq $(MRLSP_SEED_START) $$(($(MRLSP_SEED_START)+$(MRLSP_NUM_EXPERIMENTS) - 1)))
endef

MRLSP_NUM_ROBOTS = 1 2 3
all-targets-mrlsp-eval = $(foreach num_robots, $(MRLSP_NUM_ROBOTS), \
							$(foreach seed, $(call mrlsp_get_seeds), \
								$(DATA_BASE_DIR)/$(MRLSP_BASENAME)/$(MRLSP_EXPERIMENT_NAME)/mrlsp_eval_$(seed)_r$(num_robots).png))

.PHONY: mrlsp-eval
mrlsp-eval: $(all-targets-mrlsp-eval)
$(all-targets-mrlsp-eval): $(lsp-office-train-file)
$(all-targets-mrlsp-eval): seed = $(shell echo $@ | grep -oE 'mrlsp_eval_[0-9]+' | cut -d'_' -f3)
$(all-targets-mrlsp-eval): num_robots = $(shell echo $@ | grep -oE 'r[0-9]+' | grep -oE '[0-9]+')
$(all-targets-mrlsp-eval):
	@echo "Evaluating: $(num_robots) robots, seed: $(seed)"
	$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_BASENAME)/$(MRLSP_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m mrlsp.scripts.mrlsp_eval \
		$(MRLSP_CORE_ARGS) \
		--save_dir data/$(MRLSP_BASENAME)/$(MRLSP_EXPERIMENT_NAME) \
		--num_robots $(num_robots) \
		--logfile_dir data/$(MRLSP_BASENAME)/log \
		--seed $(seed)

MRLSP_NUM_ROBOTS_RESULTS = $(MRLSP_NUM_ROBOTS)
all-targets-mrlsp-results = $(foreach num_robots, $(MRLSP_NUM_ROBOTS_RESULTS), \
								$(DATA_BASE_DIR)/$(MRLSP_BASENAME)/$(MRLSP_EXPERIMENT_NAME)/r_${num_robots}_scatter.png)
.PHONY: mrlsp-results
mrlsp-results: $(all-targets-mrlsp-results)
$(all-targets-mrlsp-results): num_robots = $(shell echo $@ | grep -oE '[0-9]+')
$(all-targets-mrlsp-results):
	@echo "Results for MRLSP: Robots $(num_robots)"
	@$(DOCKER_PYTHON) -m mrlsp.scripts.mrlsp_results \
		--save_dir data/$(MRLSP_BASENAME)/$(MRLSP_EXPERIMENT_NAME) \
		--num_robots $(num_robots)



# Exploration

# Visualize MRLSP and optimistic planner
.PHONY: mrlsp-vis-exploration
mrlsp-vis-exploration: DOCKER_ARGS ?= -it
mrlsp-vis-exploration: arg-check-data
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_BASENAME)/optimistic
	@$(DOCKER_PYTHON) -m mr_exploration.scripts.mr_exploration_visualize \
		--save_dir data/$(MRLSP_BASENAME)/optimistic \
		--num_robots 2

# Put args here