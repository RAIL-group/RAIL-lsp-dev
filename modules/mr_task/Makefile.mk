MRTASK_BASENAME = mr_task
MRTASK_SEED_START = 4000
MRTASK_NUM_EXPERIMENTS = 200
MRTASK_EXPERIMENT_NAME = dbg_random
define mrlsp_get_seeds
	$(shell seq $(MRTASK_SEED_START) $$(($(MRTASK_SEED_START)+$(MRTASK_NUM_EXPERIMENTS) - 1)))
endef

MRTASK_NUM_ROBOTS = 1 2 3
MRTASK_PLANNERS = optimistic learned learnedgreedy
all-targets-mrtask-eval = $(foreach planner, $(MRTASK_PLANNERS), \
	   						$(foreach num_robots, $(MRTASK_NUM_ROBOTS), \
								$(foreach seed, $(call mrlsp_get_seeds), \
									$(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)/mtask_eval_planner_$(planner)_n_$(num_robots)_seed_$(seed).png)))
.PHONY: mr-task-procthor-eval
mr-task-procthor-eval: $(all-targets-mrtask-eval)
$(all-targets-mrtask-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval): num_robots = $(shell echo $@ | grep -oE '_n_[0-9]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed), num_robots $(num_robots)"
	@mkdir -p $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_procthor \
		--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
		--num_robots $(num_robots) \
		--planner $(planner) \
		--seed $(seed) \
		--num_iterations 100000 \
		--C 10 \
		--resolution 0.05 \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt

.PHONY: mr-task-procthor-results
mr-task-procthor-results: mr-task-procthor-eval
mr-task-procthor-results:
	@for num_robots in $(MRTASK_NUM_ROBOTS); do \
		echo "Results for $$num_robots robots"; \
		$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_results \
			--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
			--num_robots $$num_robots \
			--resolution 0.05; \
	done

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

.PHONY: mr-task-generate-video
mr-task-generate-video: DOCKER_ARGS ?= -it
mr-task-generate-video:
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_procthor_for_video \
	 	--save_dir data/$(MRTASK_BASENAME)/ \
		--num_robots 3 \
		--planner learned \
		--seed 2012 \
		--num_iterations 50000 \
		--C 10 \
		--resolution 0.05 \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt

.PHONY: mr-task-vis-planner
mr-task-vis-planner: DOCKER_ARGS ?= -it
mr-task-vis-planner:
	@mkdir -p $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/generate_plots
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_procthor_for_plots \
	 	--save_dir data/$(MRTASK_BASENAME)/generate_plots \
		--num_robots 2 \
		--planner learnedgreedy \
		--seed 2076 \
		--num_iterations 100000 \
		--C 10 \
		--resolution 0.05 \
		--network_file data/$(MRTASK_BASENAME)/raihan_nn/fcnn.pt

# for overview figure: 1154, 1238
# 2045
