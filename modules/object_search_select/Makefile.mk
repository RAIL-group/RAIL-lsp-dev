OBJECT_SEARCH_SELECT_BASENAME ?= object_search_select
OBJECT_SEARCH_SELECT_EXPERIMENT_NAME = procthor
OBJECT_SEARCH_SELECT_CORE_ARGS ?= --resolution 0.05


OBJECT_SEARCH_SELECT_NUM_SEEDS_DEPLOY ?= 150
OBJECT_SEARCH_SELECT_POLICIES ?= optimistic lspgptprompta lspgptpromptb lspgptpromptminimal lspgeminiprompta lspgeminipromptb lspgeminipromptminimal fullgptpromptdirect fullgeminipromptdirect
OBJECT_SEARCH_SELECT_ENVS ?= apartment
OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR ?= policy_selection/replay_costs_iclr

# Note: *_start seed variable names are used in get_seed function
apartment_start_seed ?= 1000

define object_search_select_get_seeds
	$(eval start := $(1)_start_seed)
	$(shell seq $(value $(start)) $$(($(value $(start))+$(2)-1)))
endef

object-search-select-offline-replay-seeds = $(foreach env,$(OBJECT_SEARCH_SELECT_ENVS), \
											$(foreach policy,$(OBJECT_SEARCH_SELECT_POLICIES), \
												$(foreach seed,$(call object_search_select_get_seeds, $(env), $(OBJECT_SEARCH_SELECT_NUM_SEEDS_DEPLOY)), \
													$(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/target_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

.PHONY: object-search-offline-replay-demo
object-search-offline-replay-demo: DOCKER_ARGS ?= -it
object-search-offline-replay-demo: seed ?= 1003
object-search-offline-replay-demo: policy ?= lspgptprompta
object-search-offline-replay-demo: env = apartment
object-search-offline-replay-demo: OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR = object_search/offline_replay_demo
object-search-offline-replay-demo:
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)
	@$(DOCKER_PYTHON) -m object_search_select.scripts.offline_replay_costs \
		$(OBJECT_SEARCH_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--chosen_planner $(policy) \
		--env $(env) \

# 		--do_not_replay
.PHONY: object-search-select-offline-replay-costs
object-search-select-offline-replay-costs: $(object-search-select-offline-replay-seeds)
$(object-search-select-offline-replay-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object-search-select-offline-replay-seeds): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(object-search-select-offline-replay-seeds): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(object-search-select-offline-replay-seeds):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)
	@$(DOCKER_PYTHON) -m object_search_select.scripts.offline_replay_costs \
		$(OBJECT_SEARCH_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--chosen_planner $(policy) \
		--env $(env) \
		> $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/stdout_$(policy)_$(env)_$(seed).txt

# object-search-select-policy-selection-results: object-search-select-offline-replay-costs
object-search-select-policy-selection-results: DOCKER_ARGS ?= -it
object-search-select-policy-selection-results: xhost-activate
object-search-select-policy-selection-results: OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR = prompt_selection_real_robot/iclr_sep19
object-search-select-policy-selection-results:
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/results
	@$(DOCKER_PYTHON) -m object_search_select.scripts.prompt_selection_results_real_robot \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR) \
		> $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/results/results.txt


object_search_select_single_trial_seeds = 1033 1097 1103 1121 1137 1147
object_search_select_single_trial_policies = optimistic lspgptprompta fullgptpromptdirect
object_search_select_single_trial_envs = apartment
object_search_select_single_trial_save_dir = single_trial
object_search_select_single_trials_targets = $(foreach env,$(object_search_select_single_trial_envs), \
												$(foreach seed,$(object_search_select_single_trial_seeds), \
													$(foreach policy,$(object_search_select_single_trial_policies), \
														$(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(object_search_select_single_trial_save_dir)/$(seed)/img_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

.PHONY: object-search-select-single-trial
object-search-select-single-trial: $(object_search_select_single_trials_targets)
$(object_search_select_single_trials_targets): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object_search_select_single_trials_targets): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(object_search_select_single_trials_targets): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(object_search_select_single_trials_targets):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(object_search_select_single_trial_save_dir)/$(seed)
	@$(DOCKER_PYTHON) -m object_search_select.scripts.offline_replay_costs \
		$(OBJECT_SEARCH_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(object_search_select_single_trial_save_dir)/$(seed) \
		--chosen_planner $(policy) \
		--env $(env) \
		--do_not_replay

env_image_seeds = $(shell for seed in $$(seq 1000 1100); \
						do echo "$(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/procthor_images/env_image_$${seed}.png"; done)

.PHONY: procthor-env-image
procthor-env-image: $(env_image_seeds)
$(env_image_seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(env_image_seeds):
	@echo $@
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/procthor_images
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m object_search_select.scripts.procthor_env_image \
		$(CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/procthor_images \
	 	--current_seed $(seed)
