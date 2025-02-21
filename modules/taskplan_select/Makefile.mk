TASKPLAN_SELECT_BASENAME ?= taskplan_select
TASKPLAN_SELECT_EXPERIMENT_NAME = procthor
TASKPLAN_SELECT_CORE_ARGS ?= --resolution 0.05


env_image_seeds = $(shell for seed in $$(seq 1000 1100); \
						do echo "$(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/procthor_images/env_image_$${seed}.png"; done)

.PHONY: procthor-env-image
procthor-env-image: $(env_image_seeds)
$(env_image_seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(env_image_seeds):
	@echo $@
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/procthor_images
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.procthor_env_image \
		$(CORE_ARGS) \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/procthor_images \
	 	--current_seed $(seed) \
		--orthographic


.PHONY: object-search-demo
object-search-demo: seed = 13
object-search-demo:
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/object_search/$(TASKPLAN_SELECT_EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.object_search \
		$(CORE_ARGS) \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/object_search/$(TASKPLAN_SELECT_EXPERIMENT_NAME) \
	 	--current_seed $(seed) \
	 	--logfile_name combined_logfile.txt \
		--network_file /data/$(TASKPLAN_SELECT_BASENAME)/logs/$(TASKPLAN_SELECT_EXPERIMENT_NAME)/gnn.pt


# Policy Selection
TASKPLAN_SELECT_NUM_SEEDS_DEPLOY ?= 150
TASKPLAN_SELECT_POLICIES ?= prompttrivial lspgptpromptone lspgptprompttwo lspgptpromptthree lspgeminipromptone lspgeminiprompttwo lspgeminipromptthree fullgptpromptone fullgeminipromptone
TASKPLAN_SELECT_ENVS ?= apartment
TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR ?= policy_selection/replay_costs_full_llm3

# Note: *_start seed variable names are used in get_seed function
apartment_start_seed ?= 1000

define taskplan_select_get_seeds
	$(eval start := $(1)_start_seed)
	$(shell seq $(value $(start)) $$(($(value $(start))+$(2)-1)))
endef

taskplan-select-offline-replay-seeds = $(foreach env,$(TASKPLAN_SELECT_ENVS), \
											$(foreach policy,$(TASKPLAN_SELECT_POLICIES), \
												$(foreach seed,$(call taskplan_select_get_seeds, $(env), $(TASKPLAN_SELECT_NUM_SEEDS_DEPLOY)), \
													$(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)/target_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

.PHONY: object-search-offline-replay-demo
object-search-offline-replay-demo: seed ?= 1021
object-search-offline-replay-demo: policy ?= prompttrivial
object-search-offline-replay-demo: env = apartment
object-search-offline-replay-demo: TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR = object_search/offline_replay_demo
object-search-offline-replay-demo:
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.offline_replay_costs \
		$(TASKPLAN_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--chosen_planner $(policy) \
		--env $(env) \
		--do_not_replay

.PHONY: taskplan-select-offline-replay-costs
taskplan-select-offline-replay-costs: $(taskplan-select-offline-replay-seeds)
$(taskplan-select-offline-replay-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(taskplan-select-offline-replay-seeds): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(taskplan-select-offline-replay-seeds): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(taskplan-select-offline-replay-seeds):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.offline_replay_costs \
		$(TASKPLAN_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--chosen_planner $(policy) \
		--env $(env) \
		> $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)/stdout_$(policy)_$(env)_$(seed).txt

taskplan-select-policy-selection-results: taskplan-select-offline-replay-costs
taskplan-select-policy-selection-results: DOCKER_ARGS ?= -it
taskplan-select-policy-selection-results: xhost-activate
taskplan-select-policy-selection-results:
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)/results
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.prompt_selection_results \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--start_seeds $(apartment_start_seed) \
		--num_seeds $(TASKPLAN_SELECT_NUM_SEEDS_DEPLOY) \
		> $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(TASKPLAN_SELECT_REPLAY_COSTS_SAVE_DIR)/results/results.txt


single_trial_seeds = 1057 1059 1060 1061 1065 1066 1067 1079 1080 1093
single_trial_policies = prompttrivial lspgptpromptone fullgptpromptone
single_trial_envs = apartment
single_trial_save_dir = single_trial
single_trials_targets = $(foreach env,$(single_trial_envs), \
							$(foreach seed,$(single_trial_seeds), \
								$(foreach policy,$(single_trial_policies), \
									$(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(single_trial_save_dir)/img_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

.PHONY: taskplan-select-single-trial
taskplan-select-single-trial: $(single_trials_targets)
$(single_trials_targets): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(single_trials_targets): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(single_trials_targets): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(single_trials_targets):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(TASKPLAN_SELECT_BASENAME)/$(single_trial_save_dir)
	@$(DOCKER_PYTHON) -m taskplan_select.scripts.offline_replay_costs \
		$(TASKPLAN_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(TASKPLAN_SELECT_BASENAME)/$(single_trial_save_dir) \
		--chosen_planner $(policy) \
		--env $(env) \
		--do_not_replay
