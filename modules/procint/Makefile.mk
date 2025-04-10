## == -- proc-INT: Training by explainbale intervention -- == ##
PROC_INT_BASENAME = procint/$(EXPERIMENT_NAME)
PROC_INT_NUM_TRAINING_SEEDS = 20

PROC_INT_CORE_ARGS ?= --save_dir /data/$(PROC_INT_BASENAME)/ \
 		--network_file /data/$(PROC_INT_BASENAME)/logs/fcnn.pt \
		--learning_rate 1e-3 \
		--resolution 0.05 \
		--cache_path /data/.cache

proc-int-network-file = $(DATA_BASE_DIR)/$(PROC_INT_BASENAME)/logs/fcnn.pt
$(proc-int-network-file):
	@mkdir -p $(DATA_BASE_DIR)/$(PROC_INT_BASENAME)/logs
	@$(DOCKER_PYTHON) -m procint.scripts.model_init \
		--save_dir /data/$(PROC_INT_BASENAME)/logs/

.PHONY: proc-int-model-init
proc-int-model-init: $(proc-int-network-file)

proc-int-data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(PROC_INT_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(PROC_INT_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done)

$(proc-int-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(proc-int-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(proc-int-data-gen-seeds): #$(proc-int-network-file)
	@echo "Generating Data [$(PROC_INT_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/$(PROC_INT_BASENAME)/images
	@mkdir -p $(DATA_BASE_DIR)/$(PROC_INT_BASENAME)/data_completion_logs
	@$(DOCKER_PYTHON) -m procint.scripts.intervene \
		$(PROC_INT_CORE_ARGS) \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest)

.PHONY: proc-int-train
proc-int-train: $(proc-int-data-gen-seeds)
