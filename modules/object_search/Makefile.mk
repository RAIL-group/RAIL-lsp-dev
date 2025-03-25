OBJECT_SEARCH_BASENAME ?= object_search
OBJECT_SEARCH_EXPERIMENT_NAME ?= frontier_results
OBJECT_SEARCH_CORE_ARGS ?= --resolution 0.05
OBJECT_SEARCH_NUM_EVAL_SEEDS ?= 200


.PHONY: object-search-demo
object-search-demo: DOCKER_ARGS ?= -it
object-search-demo: seed = 1021
object-search-demo:
	@mkdir -p $(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME)
	@$(call xhost_activate)
	@echo "Running object search | Seed: $(seed)"
	@$(DOCKER_PYTHON) -m object_search.scripts.object_search_demo \
		$(OBJECT_SEARCH_CORE_ARGS) \
		--save_dir /data/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME) \
	 	--current_seed $(seed) \
		--network_file /data/$(OBJECT_SEARCH_BASENAME)/logs/FCNNforObjectSearch.pt


object-search-eval-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(OBJECT_SEARCH_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(OBJECT_SEARCH_BASENAME)/$(OBJECT_SEARCH_EXPERIMENT_NAME)/object_search_frontier_$${ii}.png"; done)

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
