MRTASK_BASENAME = mr_task
MRTASK_SEED_START = 2000
MRTASK_NUM_EXPERIMENTS = 200
MRTASK_NUM_ROBOTS = 2
MRTASK_EXPERIMENT_NAME = experiment_$(MRTASK_NUM_ROBOTS)_paper_object_search
define mrlsp_get_seeds
	$(shell seq $(MRTASK_SEED_START) $$(($(MRTASK_SEED_START)+$(MRTASK_NUM_EXPERIMENTS) - 1)))
endef

MRTASK_PLANNERS = optimistic learned
all-targets-mrtask-eval = $(foreach planner, $(MRTASK_PLANNERS), \
							$(foreach seed, $(call mrlsp_get_seeds), \
								$(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)/mtask_eval_planner_$(planner)_seed_$(seed).png))
# .PHONY: mr-task-eval-toy
# mr-task-eval-toy: $(all-targets-mrtask-eval)
# $(all-targets-mrtask-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
# $(all-targets-mrtask-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
# $(all-targets-mrtask-eval):
# 	@echo "Evaluating: planner: $(planner), seed: $(seed)"
# 	@mkdir -p $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)
# 	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_toy_env \
# 	 	--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
# 		--seed $(seed) \
# 		--num_robots 2 \
# 		--planner $(planner) \
# 		--num_iterations 500000 \
# 		--C 10

.PHONY: mr-task-eval-procthor
mr-task-eval-procthor: $(all-targets-mrtask-eval)
$(all-targets-mrtask-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_procthor \
	 	--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
		--num_robots $(MRTASK_NUM_ROBOTS) \
		--planner $(planner) \
		--seed $(seed) \
		--num_iterations 100000 \
		--C 10 \
		--resolution 0.05 \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt

.PHONY: mr-task-results
mr-task-results: mr-task-eval-procthor
mr-task-results:
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_results \
	 	--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
		--num_robots $(MRTASK_NUM_ROBOTS)

.PHONY: mr-task-vis-net-predictions
mr-task-vis-net-predictions: DOCKER_ARGS ?= -it
mr-task-vis-net-predictions:
	@rm -f $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/raihan_nn/network_output.txt
	@touch $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/raihan_nn/network_output.txt
	@$(DOCKER_PYTHON) -m mr_task.scripts.vis_net_predictions \
	 	--save_dir data/$(MRTASK_BASENAME)/raihan_nn \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt \
		--seed 2020 \
		--resolution 0.05

.PHONY: mr-task-vis-planner
mr-task-vis-planner: DOCKER_ARGS ?= -it
mr-task-vis-planner:
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_procthor_video \
	 	--save_dir data/$(MRTASK_BASENAME)/ \
		--num_robots 2 \
		--planner learned \
		--seed 2161 \
		--num_iterations 100000 \
		--C 10 \
		--resolution 0.05 \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt
