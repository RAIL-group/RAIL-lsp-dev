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
object-search-offline-replay-demo: seed ?= 1021
object-search-offline-replay-demo: policy ?= lspgeminiprompta
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
		--do_not_replay

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

object-search-select-policy-selection-results: object-search-select-offline-replay-costs
object-search-select-policy-selection-results: DOCKER_ARGS ?= -it
object-search-select-policy-selection-results: xhost-activate
object-search-select-policy-selection-results:
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/results
	@$(DOCKER_PYTHON) -m object_search_select.scripts.prompt_selection_results \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR) \
		--start_seeds $(apartment_start_seed) \
		--num_seeds $(OBJECT_SEARCH_SELECT_NUM_SEEDS_DEPLOY) \
		> $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OBJECT_SEARCH_SELECT_REPLAY_COSTS_SAVE_DIR)/results/results.txt

OSS_LLM_RESULTS_DIR_NAME ?= gptoss_results
# OSS_LLM_POLICIES ?= lspgptossprompta lspgptosspromptb lspgptosspromptminimal fullgptosspromptdirect
OSS_LLM_POLICIES ?= lspgptossprompta lspgptosspromptb lspgptosspromptminimal fullgptosspromptdirect
object-search-select-oss-llm-eval-seeds = $(foreach env,$(OBJECT_SEARCH_SELECT_ENVS), \
											$(foreach policy,$(OSS_LLM_POLICIES), \
												$(foreach seed,$(call object_search_select_get_seeds, $(env), $(OBJECT_SEARCH_SELECT_NUM_SEEDS_DEPLOY)), \
													$(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME)/target_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

object-search-select-oss-llm-eval: $(object-search-select-oss-llm-eval-seeds)
$(object-search-select-oss-llm-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object-search-select-oss-llm-eval-seeds): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(object-search-select-oss-llm-eval-seeds): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(object-search-select-oss-llm-eval-seeds): DOCKER_ARGS ?= --network="host"
$(object-search-select-oss-llm-eval-seeds):
	$(call xhost_activate)
	@echo "Evaluating OSS LLM Policy [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME)
	@$(DOCKER_PYTHON) -m object_search_select.scripts.oss_llm_eval \
		$(OBJECT_SEARCH_SELECT_CORE_ARGS) \
	 	--current_seed $(seed) \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME) \
		--chosen_planner $(policy) \
		--env $(env) \
		> $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME)/stdout_$(policy)_$(env)_$(seed).txt

object-search-select-oss-llm-results: DOCKER_ARGS ?= -it
object-search-select-oss-llm-results: xhost-activate
object-search-select-oss-llm-results:
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME)/results
	@$(DOCKER_PYTHON) -m object_search_select.scripts.oss_llm_results \
		--save_dir /data/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME) \
		--start_seeds $(apartment_start_seed) \
		--num_seeds $(OBJECT_SEARCH_SELECT_NUM_SEEDS_DEPLOY) \
		> $(DATA_BASE_DIR)/$(OBJECT_SEARCH_SELECT_BASENAME)/$(OSS_LLM_RESULTS_DIR_NAME)/results/results.txt
