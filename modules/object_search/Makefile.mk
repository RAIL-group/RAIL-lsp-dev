OBJECT_SEARCH_BASENAME ?= object_search
OBJECT_SEARCH_EXPERIMENT_NAME ?= dbg
OBJECT_SEARCH_CORE_ARGS ?= --resolution 0.05
OBJECT_SEARCH_NUM_TRAINING_SEEDS ?= 500
OBJECT_SEARCH_NUM_TESTING_SEEDS ?= 200
OBJECT_SEARCH_NUM_EVAL_SEEDS ?= 200


object-search-data-gen-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(OBJECT_SEARCH_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_data/data_collect_plots/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 700 $$((700 + $(OBJECT_SEARCH_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_data/data_collect_plots/data_testing_$${ii}.png"; done)

$(object-search-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(object-search-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object-search-data-gen-seeds):
	@echo "Generating Data [$(OBJECT_SEARCH_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_data/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_data/data
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_data/data_collect_plots
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_generate_data \
		$(OBJECT_SEARCH_CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/training_data/ \
	 	--current_seed $(seed) \
		--data_file_base_name data_$(traintest)

.SECONDARY: $(object-search-data-gen-seeds)

object-search-train-file = $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_logs/$(OBJECT_SEARCH_EXPERIMENT_NAME)/FCNNObjectSearch.pt
$(object-search-train-file): $(object-search-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/training_logs/$(OBJECT_SEARCH_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_train_net \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/training_logs/$(OBJECT_SEARCH_EXPERIMENT_NAME) \
		--data_csv_dir /data/$(OBJECT_SEARCH_BASENAME)/training_data/

object-search-eval-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(OBJECT_SEARCH_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/results/$(OBJECT_SEARCH_EXPERIMENT_NAME)/object_search_$${ii}.png"; done)
$(object-search-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object-search-eval-seeds): $(object-search-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/results/$(OBJECT_SEARCH_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_eval \
		$(OBJECT_SEARCH_CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/results/$(OBJECT_SEARCH_EXPERIMENT_NAME) \
		--network_file /data/$(OBJECT_SEARCH_BASENAME)/training_logs/$(OBJECT_SEARCH_EXPERIMENT_NAME)/FCNNObjectSearch.pt \
	 	--current_seed $(seed) \
		--image_filename object_search_$(seed).png

.PHONY: object-search-results
object-search-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		--data_file /data/$(OBJECT_SEARCH_BASENAME)/results/$(OBJECT_SEARCH_EXPERIMENT_NAME)/costs_log.txt \
		--output_image_file /data/$(OBJECT_SEARCH_BASENAME)/results/results_$(OBJECT_SEARCH_EXPERIMENT_NAME).png

.PHONY: object-search-data-gen object-search-train object-search-eval object-search-results
object-search-data-gen: $(object-search-data-gen-seeds)
object-search-train: $(object-search-train-file)
object-search-eval: $(object-search-eval-seeds)

# Object Search with Frontier
# ===========================

.PHONY: object-search-frontier
object-search-frontier: $(object-search-eval-seeds)
$(object-search-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(object-search-eval-seeds):
	@$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_frontier \
		$(OBJECT_SEARCH_CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME) \
	 	--current_seed $(seed) \
		--network_file /data/$(OBJECT_SEARCH_BASENAME)/logs/FCNNforObjectSearch.pt \
		--image_filename object_search_frontier_$(seed).png


.PHONY: object-search-frontier-demo
object-search-frontier-demo: DOCKER_ARGS ?= -it
object-search-frontier-demo: seed = 1021
object-search-frontier-demo:
	@$(call xhost_activate)
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/demo
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_frontier \
		$(OBJECT_SEARCH_CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/demo \
	 	--current_seed $(seed) \
		--network_file /data/$(OBJECT_SEARCH_BASENAME)/logs/FCNNforObjectSearch.pt \
		--image_filename object_search_frontier_$(seed).png

object-search-frontier-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		--data_file /data/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME)/costs_log.txt \
		--output_image_file /data/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME)/results.png
