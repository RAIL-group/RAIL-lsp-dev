
help::
	@echo "Multi Robot Policy Selection Experiments:"

MRLSP_SELECT_ENVIRONMENT_NAME ?= mazeA
MRLSP_SELECT_BASENAME = mrlsp_select
MRLSP_SELECT_UNITY_BASENAME ?= $(RAIL_SIM_BASENAME)
MRLSP_SELECT_MAP_TYPE ?= maze
MRLSP_SELECT_CORE_ARGS ?= --unity_path /unity/$(MRLSP_SELECT_UNITY_BASENAME).x86_64 \
		--map_type $(MRLSP_SELECT_MAP_TYPE) \
		--inflation_radius_m 0.75


MRLSP_SELECT_BASENAME ?= mrlsp_select
MRLSP_SELECT_NUM_SEEDS_DEPLOY ?= 150

MAZE_POLICIES ?= "nonlearned lspA lspB lspC"
MAZE_ENVS ?= "envA envB envC"

OFFICE_POLICIES ?= "nonlearned lspmaze lspoffice lspofficewallswap"
OFFICE_ENVS ?= "mazeA office officewall"

# Note: *_start seed variable names are used in get_seed function
envA_start_seed_mrselect ?= 2000
envB_start_seed_mrselect ?= 3000
envC_start_seed_mrselect ?= 4000
MRLSP_SELECT_maze_costs_save_dir ?= mrlsp_policy_selection/maze_costs

mazeA_start_seed_mrselect ?= 2000
office_start_seed_mrselect ?= 3000
officewall_start_seed_mrselect ?= 4000
MRLSP_SELECT_office_costs_save_dir ?= mrlsp_policy_selection/office_costs

define get_seeds_mrlsp_select
	$(eval start := $(1)_start_seed_mrselect)
	$(shell seq $(value $(start)) $$(($(value $(start))+$(2)-1)))
endef

mr-offline-replay-seeds = $(foreach env,$(ENVS_TO_DEPLOY), \
								$(foreach policy,$(POLICIES_TO_RUN), \
									$(foreach seed,$(call get_seeds_mrlsp_select, $(env), $(MRLSP_SELECT_NUM_SEEDS_DEPLOY)), \
										$(DATA_BASE_DIR)/$(MRLSP_SELECT_BASENAME)/$(replay_costs_save_dir)/target_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

mrlsp-select-offline-replay-costs: $(mr-offline-replay-seeds)
$(mr-offline-replay-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(mr-offline-replay-seeds): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(mr-offline-replay-seeds): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(mr-offline-replay-seeds): sim_name = $(if $(filter $(env),envB),_$(GREENFLOOR_SIM_ID),$(if $(filter $(env),officewall),_$(WALLSWAP_SIM_ID)))
$(mr-offline-replay-seeds): MRLSP_SELECT_MAP_TYPE = $(if $(or $(filter $(env),office),$(filter $(env),officewall)),office2,maze)
$(mr-offline-replay-seeds): MRLSP_SELECT_UNITY_BASENAME = $(RAIL_SIM_BASENAME)$(sim_name)
$(mr-offline-replay-seeds):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_SELECT_BASENAME)/$(replay_costs_save_dir)
	@$(DOCKER_PYTHON) -m mrlsp_select.scripts.mrlsp_offline_replay_costs \
		$(MRLSP_SELECT_CORE_ARGS) \
		--experiment_type $(EXPERIMENT_TYPE) \
		--seed $(seed) \
		--save_dir /data/$(MRLSP_SELECT_BASENAME)/$(replay_costs_save_dir) \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner $(policy) \
		--env $(env) \
		--num_robots 2


.PHONY: mrlsp-select-offline-replay-costs-maze
# mrlsp-select-offline-replay-costs-maze: mrlsp-select-train-mazeA mrlsp-select-train-mazeB mrlsp-select-train-mazeC
mrlsp-select-offline-replay-costs-maze:
	@$(MAKE) mrlsp-select-offline-replay-costs \
		EXPERIMENT_TYPE=maze \
		POLICIES_TO_RUN=$(MAZE_POLICIES) \
		ENVS_TO_DEPLOY=$(MAZE_ENVS) \
		replay_costs_save_dir=$(MRLSP_SELECT_maze_costs_save_dir)

.PHONY: mrlsp-select-offline-replay-costs-office
# mrlsp-select-offline-replay-costs-office: mrlsp-select-train-mazeA mrlsp-select-train-officebase mrlsp-select-train-officewall
mrlsp-select-offline-replay-costs-office:
	@$(MAKE) mrlsp-select-offline-replay-costs \
		EXPERIMENT_TYPE=office \
		POLICIES_TO_RUN=$(OFFICE_POLICIES) \
		ENVS_TO_DEPLOY=$(OFFICE_ENVS) \
		replay_costs_save_dir=$(MRLSP_SELECT_office_costs_save_dir)

.PHONY: mrlsp-policy-selection-maze mrlsp-policy-selection-office
mrlsp-policy-selection-maze: mrlsp-select-offline-replay-costs-maze
mrlsp-policy-selection-maze: DOCKER_ARGS ?= -it
mrlsp-policy-selection-maze: xhost-activate
mrlsp-policy-selection-maze:
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_policy_selection_results \
		--save_dir /data/$(MRLSP_SELECT_BASENAME)/$(MRLSP_SELECT_maze_costs_save_dir) \
		--experiment_type maze \
		--start_seeds $(envA_start_seed) $(envB_start_seed) $(envC_start_seed) \
		--num_seeds $(MRLSP_SELECT_NUM_SEEDS_DEPLOY)

.PHONY: mr-offline-replay-demo
mr-offline-replay-demo: DOCKER_ARGS ?= -it
mr-offline-replay-demo: xhost-activate
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_SELECT_BASENAME)/offline-replay-demo
	@$(DOCKER_PYTHON) -m mrlsp_select.scripts.mrlsp_offline_replay_demo \
		$(MRLSP_SELECT_CORE_ARGS) \
		--seed 82 \
		--save_dir /data/$(MRLSP_SELECT_BASENAME)/offline-replay-demo \
		--network_path /data/lsp_select/training_logs \
		--chosen_planner lspA \
		--num_robots 2 \
		--env envA \
		--do_plot
