
SCTP_BASENAME = sctp
SCTP_SEED_START = 3000
SCTP_NUM_EXPERIMENTS = 100
SCTP_NUM_DRONES = 1
SCTP_NUM_GROUNDS = 1
SCTP_NUM_VERTICES = 14
SCTP_NUM_ISLANDs = 5
SCTP_EXPERIMENT_NAME = Oct29_rg${SCTP_NUM_VERTICES}v_jsctp
define sctp_get_seeds
	$(shell seq $(SCTP_SEED_START) $$(($(SCTP_SEED_START)+$(SCTP_NUM_EXPERIMENTS) - 1)))
endef

SCTP_PLANNERS = jsctpig
all-targets-sctp-eval = $(foreach planner, $(SCTP_PLANNERS), \
							$(foreach seed, $(call sctp_get_seeds), \
								$(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/sctp_eval_planner_$(planner)_seed_$(seed).png))

.PHONY: sctpdec-eval-random-graph
sctpdec-eval-random-graph: $(all-targets-sctp-eval)
$(all-targets-sctp-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-sctp-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)
$(all-targets-sctp-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_dec_eval_random_graph \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones 0 \
		--num_grounds $(SCTP_NUM_GROUNDS) \
		--planner $(planner) \
		--seed $(seed) \
		--num_iterations 750 \
		--C 300 \
		--sampling_maps 80 \
		--n_vertex $(SCTP_NUM_VERTICES) \
		--max_depth 20 \

.PHONY: jsctp-eval-random-graph
jsctp-eval-random-graph: $(all-targets-sctp-eval)
$(all-targets-sctp-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-sctp-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)
$(all-targets-sctp-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m sctp.scripts.jsctp_eval_random_graph \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--num_ugvs $(SCTP_NUM_GROUNDS) \
		--planner $(planner) \
		--seed $(seed) \
		--num_iterations 6000 \
		--C 200 \
		--sampling_maps 80 \
		--n_vertex $(SCTP_NUM_VERTICES) \
		--max_depth 35 \

# .PHONY: sctp-eval-bridge-graph
# sctp-eval-bridge-graph: $(all-targets-sctp-eval)
# $(all-targets-sctp-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
# $(all-targets-sctp-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)
# $(all-targets-sctp-eval):
# 	@echo "Evaluating: planner: $(planner), seed: $(seed)"
# 	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
# 	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_eval_bridge_graph \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--planner $(planner) \
# 		--seed $(seed) \
# 		--num_iterations 3000 \
# 		--sampling_maps 80 \
# 		--C 200 \
# 		--max_depth 25 \

# .PHONY: sctp-eval-island-graph
# sctp-eval-island-graph: $(all-targets-sctp-eval)
# $(all-targets-sctp-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
# $(all-targets-sctp-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
# $(all-targets-sctp-eval):
# 	@echo "Evaluating: planner: $(planner), seed: $(seed)"
# 	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
# 	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_eval_island_graph \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--planner $(planner) \
# 		--seed $(seed) \
# 		--num_iterations 1000 \
# 		--C 200 \
# 		--n_vertex $(SCTP_NUM_VERTICES) \
# 		--max_depth 25 \




# .PHONY: sctp-a-random-graph
# sctp-a-random-graph:
# 	@echo "Evaluating: planner execution"
# 	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
# 	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_eval_random_graph \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--planner "base" \
# 		--seed 2035 \
# 		--num_iterations 1000 \
# 		--C 30 \
# 		--v_num $(SCTP_NUM_VERTICES) \
# 		--resolution 0.05 \

.PHONY: sctp-planner-test
sctp-planner-test:
	@echo "Evaluating: planner: $(planner), seed: 3000"
	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_planner \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--num_ugvs $(SCTP_NUM_GROUNDS) \
		--seed 3000 \
		--num_iterations 1000 \
		--C 200.0 \

# .PHONY: drones_planner-test
# drones_planner-test:
# 	@echo "Evaluating: planner: drones_planner, seed: $(seed)"
# 	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_planner \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--num_iterations 1000 \
# 		--C 10 \
# 		--resolution 0.05 \


.PHONY: sctp-execution-test
sctp-execution-test: DOCKER_ARGS ?= -it
sctp-execution-test:
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m modules.tests.test_sctpdec_plan_exe\
		--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--num_iterations 2000 \
		--C 30 \
		--resolution 0.05 \

.PHONY: sctp-results
sctp-results:
	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_results \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES)

# .PHONY: mr-task-vis-net-predictions
# mr-task-vis-net-predictions: DOCKER_ARGS ?= -it
# mr-task-vis-net-predictions:
# 	@rm -f $(DATA_BASE_DIR)/$(SCTP_BASENAME)/raihan_nn/network_output.txt
# 	@touch $(DATA_BASE_DIR)/$(SCTP_BASENAME)/raihan_nn/network_output.txt
# 	@$(DOCKER_PYTHON) -m mr_task.scripts.vis_net_predictions \
# 	 	--save_dir data/$(SCTP_BASENAME)/raihan_nn \
# 		--network_file data/$(SCTP_BASENAME)/raihan_nn/fcnn.pt \
# 		--seed 2020 \
# 		--resolution 0.05
