blender-basename = blender-4.2.4-linux-x64
BLENDER_DIR ?= $(shell pwd)/resources/blender/
blender-full-name = $(BLENDER_DIR)/$(blender-basename)/blender

.PHONY: blender
blender: $(blender-full-name)
$(blender-full-name):
	@echo "Downloading Blender"
	@mkdir -p $(BLENDER_DIR)
	@cd $(BLENDER_DIR) \
		&& wget -c https://mirror.clarkson.edu/blender/release/Blender4.2/$(blender-basename).tar.xz \
		&& tar -xf $(blender-basename).tar.xz \
		&& rm $(blender-basename).tar.xz
	@echo "Blender downloaded and unpacked."

blender-build: $(blender-full-name)
	@$(DOCKER_BASE) /blender/blender --python /modules/install.py

blender-dbg: $(blender-full-name)
	@$(DOCKER_BASE) /blender/blender --background --python /modules/blendervsim/blenderscripts/check_blender_working.py

blender-dbg-render:
	@$(DOCKER_PYTHON) -m blendervsim.scripts.dbg

dbg-blender: blender
	@$(DOCKER_BASE) /blender/blender --background --python /modules/simulator/tests/test_blender_sim_core.py --log-level 0
