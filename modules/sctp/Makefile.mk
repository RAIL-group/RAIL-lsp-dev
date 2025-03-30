
SCTP_BASENAME = sctp
SCTP_SEED_START = 2000
SCTP_NUM_EXPERIMENTS = 50
SCTP_NUM_DRONES = 1
SCTP_EXPERIMENT_NAME = dbg_Mar28_randgraph
define sctp_get_seeds
	$(shell seq $(SCTP_SEED_START) $$(($(SCTP_SEED_START)+$(SCTP_NUM_EXPERIMENTS) - 1)))
endef

SCTP_PLANNERS = base sctp
all-targets-sctp-eval = $(foreach planner, $(SCTP_PLANNERS), \
							$(foreach seed, $(call sctp_get_seeds), \
								$(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/sctp_eval_planner_$(planner)_seed_$(seed).png))

.PHONY: sctp-eval-random-graph
sctp-eval-random-graph: $(all-targets-sctp-eval)
$(all-targets-sctp-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-sctp-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
$(all-targets-sctp-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_eval_random_graph \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--planner $(planner) \
		--seed $(seed) \
		--num_iterations 5000 \
		--C 30 \
		--resolution 0.05 \

.PHONY: sctp-planner-test
sctp-planner-test:
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_planner \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--num_iterations 1000 \
		--C 10 \
		--resolution 0.05 \

.PHONY: sctp-planning-loop-test
sctp-planning-loop-test: DOCKER_ARGS ?= -it
sctp-planning-loop-test:
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_planning_loop\
		--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
		--num_drones $(SCTP_NUM_DRONES) \
		--num_iterations 2000 \
		--C 30 \
		--resolution 0.05 \

.PHONY: sctp-results
sctp-results: sctp-eval-random-graph
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
