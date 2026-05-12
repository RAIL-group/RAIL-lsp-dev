SCTP_BASENAME = sctp
SCTP_SEED_START = 3075
SCTP_NUM_EXPERIMENTS =1
SCTP_DATA_SEED = 1243
SCTP_DATA_NUM = 157
SCTP_NUM_DRONES = 1
SCTP_NUM_GROUNDS = 1
SCTP_NUM_VERTICES = 16
SCTP_NUM_ISLANDs = 5
SCTP_NUM_PRUNE = 1
SCTP_NUM_SAMPLE = 1000
SCTP_EXPERIMENT_NAME = Apr30
define sctp_get_seeds
	$(shell seq $(SCTP_SEED_START) $$(($(SCTP_SEED_START)+$(SCTP_NUM_EXPERIMENTS) - 1)))
endef

GRAPHS = islands #random#bridges   
JSAP_PLANNERS = jsapdap jsapiap#jsapliap# jsapiap2 #jsap jsap2 jsapdap2# ctp#
EXP_NAME = plot_all#action_candidates#statistics#prune_num_action# 

all-targets-jsap-eval = $(foreach planner, $(JSAP_PLANNERS), \
					$(foreach seed, $(call sctp_get_seeds), \
					$(foreach graph, $(GRAPHS),\
					$(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/_graph_$(graph)_/$(SCTP_NUM_GROUNDS)/sctp_eval_planner_$(planner)_seed_$(seed)_$(SCTP_NUM_DRONES)UAVs.png)))
$(all-targets-jsap-eval): jsap_seed = $(shell echo $@ | grep -oE '_seed_[0-9]+' | cut -d'_' -f3)
$(all-targets-jsap-eval): jsap_planner = $(shell echo $@ | grep -oE '_planner_[a-z0-9]+' | cut -d'_' -f3)
$(all-targets-jsap-eval): jsap_graph = $(shell echo $@ | grep -oE '_graph_[a-z]+' | cut -d'_' -f3)


.PHONY: jsap-eval-all-graphs
jsap-eval-all-graphs: $(all-targets-jsap-eval)
$(all-targets-jsap-eval):
	@echo "Evaluating: planner: $(jsap_planner), seed: $(jsap_seed), graph: $(jsap_graph), action candidates: $(SCTP_NUM_PRUNE), sample num: $(SCTP_NUM_SAMPLE)"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/_graph_$(jsap_graph)_/$(SCTP_NUM_GROUNDS)
	@$(DOCKER_PYTHON) -m sctp.scripts.jsap_eval_all_graphs \
	 	--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/_graph_$(jsap_graph)_/$(SCTP_NUM_GROUNDS) \
		--num_drones $(SCTP_NUM_DRONES) \
		--planner $(jsap_planner) \
		--seed $(jsap_seed) \
		--num_iterations $(SCTP_NUM_SAMPLE) \
		--sampling_maps 200 \
		--C 200 \
		--max_depth 30 \
		--num_ugvs $(SCTP_NUM_GROUNDS) \
		--max_uanum $(SCTP_NUM_PRUNE) \
		--env_type $(jsap_graph) \
		--n_vertex $(SCTP_NUM_VERTICES)


DATA_SEEDS := $(shell seq $(SCTP_DATA_SEED) $$(($(SCTP_DATA_SEED) + $(SCTP_DATA_NUM) - 1)))
.PHONY: sap-generate-data
sap-generate-data: $(addprefix seed-,$(DATA_SEEDS))
seed-%:
	@echo "Generating training data for IAP-GNN with seed: $*"
	@mkdir -p $(DATA_BASE_DIR)/$(SCTP_BASENAME)/graph_data/rollouts
	@$(DOCKER_PYTHON) -m sctp.scripts.data_gen \
	 	--save_dir data/$(SCTP_BASENAME)/graph_data/rollouts \
		--num_maps 1000 \
		--seed $* \
		--num_steps 17 \
		--graph_type $(GRAPHS) \
		--n_vertex $(SCTP_NUM_VERTICES) \
	


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
sctp-results: DOCKER_ARGS ?= -it
sctp-results:
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m sctp.scripts.sctp_results \
	 	--num_ugvs $(SCTP_NUM_GROUNDS) \
		--save_dir data/$(SCTP_BASENAME)/$(SCTP_EXPERIMENT_NAME)/ \
		--num_drones $(SCTP_NUM_DRONES) \
		--exp_name $(EXP_NAME) \


.PHONY: iapgnn-train
iapgnn-train: DOCKER_ARGS ?= -it
iapgnn-train:
	@echo "Training IAP-GNN"
	@$(DOCKER_PYTHON) -m sctp.scripts.iapgnn_training \
		--graph_type $(GRAPHS) \