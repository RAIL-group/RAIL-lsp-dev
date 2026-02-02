SCTP_BASENAME = sctp
SCTP_SEED_START = 3000
SCTP_NUM_EXPERIMENTS = 100
SCTP_NUM_DRONES = 1
SCTP_NUM_GROUNDS = 1
SCTP_NUM_VERTICES = 14
SCTP_NUM_ISLANDs = 5
# SCTP_EXPERIMENT_NAME = Oct29_rg${SCTP_NUM_VERTICES}v_jsctp
SCTP_EXPERIMENT_NAME = Jan30
define sctp_get_seeds
	$(shell seq $(SCTP_SEED_START) $$(($(SCTP_SEED_START)+$(SCTP_NUM_EXPERIMENTS) - 1)))
endef

GRAPHS = dense

JSAP_PLANNERS = jsapiap jsapdap

all-targets-jsap-eval = $(foreach planner, $(JSAP_PLANNERS), \
					$(foreach seed, $(call sctp_get_seeds), \
					$(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS)/sctp_eval_planner_$(planner)_seed_$(seed)_$(SCTP_NUM_DRONES)UAVs.png))
$(all-targets-jsap-eval): jsap_seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-jsap-eval): jsap_planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)


# DSAP_PLANNERS = dsap #dsapavp 
# all-targets-dsap-bridges-eval = $(foreach planner, $(DSAP_PLANNERS), \
# 									$(foreach seed, $(call sctp_get_seeds), \
# 										$(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS)/sctp_eval_planner_$(planner)_seed_$(seed).png))
# $(all-targets-dsap-bridges-eval): dsap_seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
# $(all-targets-dsap-bridges-eval): dsap_planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)


# .PHONY: jsap-eval-bridges-graphs
# jsap-eval-bridges-graphs: $(all-targets-jsap-eval)
# $(all-targets-jsap-eval):
# 	@echo "Evaluating: planner: $(jsap_planner), seed: $(jsap_seed)"
# 	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS)
# 	@$(DOCKER_PYTHON) -m sctp.scripts.jsap_eval_bridges_graph \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--planner $(jsap_planner) \
# 		--seed $(jsap_seed) \
# 		--num_iterations 1000 \
# 		--sampling_maps 200 \
# 		--C 200 \
# 		--max_depth 20 \
# 		--num_ugvs $(SCTP_NUM_GROUNDS)

.PHONY: jsap-eval-dense-graphs
jsap-eval-dense-graphs: $(all-targets-jsap-eval)
$(all-targets-jsap-eval):
	@echo "Evaluating: planner: $(jsap_planner), seed: $(jsap_seed)"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS)
	@$(DOCKER_PYTHON) -m sctp.scripts.jsap_eval_dense_graph \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS) \
		--num_drones $(SCTP_NUM_DRONES) \
		--planner $(jsap_planner) \
		--seed $(jsap_seed) \
		--num_iterations 1000 \
		--sampling_maps 200 \
		--C 200 \
		--max_depth 20 \
		--num_ugvs $(SCTP_NUM_GROUNDS) \

# .PHONY: jsap-eval-island-bridges-graphs
# jsap-eval-island-bridges-graphs: $(all-targets-jsap-eval)
# $(all-targets-jsap-eval):
# 	@echo "Evaluating: planner: $(jsap_planner), seed: $(jsap_seed)"
# 	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS)
# 	@$(DOCKER_PYTHON) -m sctp.scripts.jsap_eval_island_bridges_graph \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/$(GRAPHS)/$(SCTP_NUM_GROUNDS) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--planner $(jsap_planner) \
# 		--seed $(jsap_seed) \
# 		--num_iterations 1000 \
# 		--sampling_maps 200 \
# 		--C 200 \
# 		--max_depth 15 \
# 		--num_ugvs $(SCTP_NUM_GROUNDS) \



.PHONY: sap-generate-data
sap-generate-data:
	@echo "Generating training data"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/graph_data
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/graph_data/pickles
	@$(DOCKER_PYTHON) -m sctp.scripts.data_gen \
	 	--save_dir data/$(SCTP_BASENAME)/graph_data \
		--num_maps 500 \
		--num_graphs 1 \
		--graph_type "island" \
		
# .PHONY: sctp-planner-test
# sctp-planner-test:
# 	@echo "Evaluating: planner: $(planner), seed: 3000"
# 	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_planner \
# 	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME) \
# 		--num_drones $(SCTP_NUM_DRONES) \
# 		--num_ugvs $(SCTP_NUM_GROUNDS) \
# 		--seed 3000 \
# 		--num_iterations 1000 \
# 		--C 200.0 \


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
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/plot_data \
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
