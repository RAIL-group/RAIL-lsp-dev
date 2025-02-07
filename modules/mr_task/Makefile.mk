MRTASK_BASENAME = mr_task
MRTASK_SEED_START = 2000
MRTASK_NUM_EXPERIMENTS = 200
MRTASK_EXPERIMENT_NAME = feb7_known
define mrlsp_get_seeds
	$(shell seq $(MRTASK_SEED_START) $$(($(MRTASK_SEED_START)+$(MRTASK_NUM_EXPERIMENTS) - 1)))
endef

MRTASK_PLANNERS = learned optimistic
all-targets-mrtask-eval = $(foreach planner, $(MRTASK_PLANNERS), \
							$(foreach seed, $(call mrlsp_get_seeds), \
								$(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)/mtask_eval_planner_$(planner)_seed_$(seed)_.png))
.PHONY: mr-task-eval
mr-task-eval: $(all-targets-mrtask-eval)
$(all-targets-mrtask-eval): seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval): planner = $(shell echo $@ | grep -oE '_planner_[a-z]+' | cut -d'_' -f3)
$(all-targets-mrtask-eval):
	@echo "Evaluating: planner: $(planner), seed: $(seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_eval_toy_env \
	 	--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
		--seed $(seed) \
		--num_robots 2 \
		--planner $(planner) \
		--num_iterations 100000 \
		--C 100

.PHONY: mr-task-results
mr-task-results: mr-task-eval
mr-task-results:
	@$(DOCKER_PYTHON) -m mr_task.scripts.mr_task_results \
	 	--save_dir data/$(MRTASK_BASENAME)/$(MRTASK_EXPERIMENT_NAME) \
		--num_robots 2
