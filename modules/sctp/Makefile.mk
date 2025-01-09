help::
	@echo ""
	@echo ""
	@echo ""
	@echo ""

SCTP_BASENAME = SCTP
SCTP_UNITY_BASENAME ?= $(RAIL_SIM_BASENAME)
SCTP_CORE_ARGS ?= --unity_path /unity/$(SCTP_UNITY_BASENAME).x86_64 \
		--map_type office2 \
		--base_resolution 0.5 \
		--inflation_radius_m 0.75 \
		--laser_max_range_m 18 \
		--iterations 40000 \
		--ucb_c 500

# For debugging purposes
.PHONY: test-sctpbase create_graph

create_graph: DOCKER_ARGS ?= -it
create_graph:
	$(call xhost_activate)
	@$(DOCKER_PYTHON) -m modules.pouct_planner.sctp.graphs

test-sctpbase: DOCKER_ARGS ?= -it
test-sctpbase:
	$(call xhost_activate)
	@$(DOCKER_PYTHON) -m modules.tests.test_sctp_ground
