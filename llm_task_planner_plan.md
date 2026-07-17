# LLM-Based Task Planner for `taskplan` — Design & Implementation Plan

**Repo:** `RAIL-group/RAIL-lsp-dev`
**Branch:** `raihan/update-sbert`
**Module:** `modules/taskplan`

This document is a handoff spec for an agentic coding tool (e.g. Antigravity) to implement.
It captures the decisions already made and the concrete file-level changes required.

---

## 0. Context the agent needs

`taskplan` plans household pick/place/prepare tasks (pick, place, move, find, boil, peel,
toast, pour-water, pour-coffee, make-coffee) over a PDDL domain/problem built per-episode
from a scene graph (`taskplan/pddl/domain.py`, `taskplan/pddl/problem.py`,
`taskplan/pddl/helper.py`, `taskplan/pddl/task.py`). The domain is solved with
`pddlstream.algorithms.search.solve_from_pddl` (FF-astar), and re-solved every time a `find`
action reveals new information, inside the receding-horizon loop in
`taskplan/planners/task_loop.py` and `taskplan/scripts/eval_replan.py`.

A sibling module, `object_search`, already has a working (cloud-API) LLM planner precedent —
`object_search/planners/llm_planner.py` + `object_search/learning/models/llm.py` — that we
are structurally reusing, but pointed at a local model instead of GPT-4/Gemini.

## 1. Decisions made (do not re-litigate these)

1. **Full replacement**, not a hybrid. The LLM planner replaces `solve_from_pddl` entirely
   for the LLM-baseline condition. It is not just swapping in for the GNN/FCNN subgoal
   scorer inside the existing LSP cost framework.
2. **New, simplified `find` operator for the LLM-facing domain only.** The existing domain's
   `find` action bundles movement + search + pickup into one action, parameterized by
   `(?obj ?from ?to)`, with cost given by a precomputed fluent `(find-cost ?obj ?from ?to)`.
   That fluent is populated in `taskplan/pddl/problem.py` (`init_fluents[('find-cost', ...)]`,
   ~lines 138/154) via `taskplan/pddl/helper.py::update_find_costs`, which calls the
   GNN/FCNN-driven expected-cost machinery in `taskplan/core.py`. **This is exactly the
   information the LLM baseline must not have access to** — it's the output of the
   architecture being compared against. The LLM domain therefore gets a new operator:
   `(find ?obj ?loc)` — robot must already be `(rob-at ?loc)` (via the existing, unchanged
   `move` action), searches at that single location, and on success is holding the object.
   Cost is the existing flat constant `costs['find'] = 20` (see
   `taskplan/utilities/utils.py::get_action_costs`), not a lookup table.
3. **Model: Gemma 4, local, via Ollama.** Target hardware: RTX 4060 laptop, 8GB VRAM, 32GB
   system RAM.
   - **Primary: Gemma 4 E4B**, QAT build (~5GB VRAM at 4-bit). This is the safe daily
     driver for 8GB cards and leaves headroom for KV cache + whatever else is using the GPU
     (e.g. AI2-THOR rendering during data generation).
   - **Stretch option: Gemma 4 12B**, QAT build (~7GB VRAM). Only usable when nothing else
     is holding VRAM — check with `nvidia-smi` before relying on it, it's tight on an 8GB card.
   - Do **not** target 26B-A4B or 31B on this laptop (need ~15GB+ VRAM even quantized).
   - Gemma 4 supports a native thinking mode (`<|think|>` in the system prompt) — treat this
     as a config flag to A/B test (better plan quality vs. latency), not a fixed choice.
   - No cloud calls, no API key. Local endpoint only.

## 2. Domain / problem-generation changes

Files: `taskplan/pddl/domain.py`, `taskplan/pddl/problem.py`, `taskplan/pddl/helper.py`

- Add `taskplan/pddl/domain.py::get_llm_domain(whole_graph)` — a copy of `get_domain` with
  the `find` action replaced:
  ```
  (:action find
      :parameters (?obj - item ?loc - location)
      :precondition (and
          (not (ban-find))
          (rob-at ?loc)
          (not (is-located ?obj))
          (is-pickable ?obj)
          (hand-is-free)
      )
      :effect (and
          (is-located ?obj)
          (not (hand-is-free))
          (is-holding ?obj)
          (not (ban-move))
          (increase (total-cost) {costs['find']})
      )
  )
  ```
  Keep `(:functions (known-cost ?start ?end) (total-cost))` — drop `(find-cost ?obj ?loc)`
  entirely, it's no longer needed.
- Do **not** modify `get_domain` — it must stay untouched for the PDDL/learned/naive
  baselines to remain runnable and comparable.
- In `taskplan/pddl/problem.py::get_problem`, add a `planner_mode` (or reuse `args.cost_type`
  the way the rest of the module does) branch that skips all `find-cost` fluent generation
  when building the LLM-mode problem struct. `missing_objects` bookkeeping stays as-is (still
  needed to know what hasn't been found yet).
- In `taskplan/pddl/helper.py`, the LLM path should never call `update_find_costs` (that
  function *is* the expected-cost architecture — it must not run for this baseline). Add an
  equivalent `update_problem_find_llm` alongside the existing `update_problem_find` if the
  predicate bookkeeping differs for the 2-arg action (check `is-at` handling since the LLM
  `find` doesn't move the robot — the object's location as discovered doesn't require a
  `rob-at` update anymore, but this needs verifying against test cases once implemented).
- Add tests: `tests/test_llm_domain.py` — confirm FF-astar can still solve the simplified
  domain on a couple of the existing test fixtures (sanity-check the domain grammar itself,
  independent of any LLM work, before touching prompting).

## 3. New module layout

```
modules/taskplan/taskplan/
  pddl/
    domain.py            # + get_llm_domain()
    problem.py           # + planner_mode branch, no find-cost fluents
    helper.py            # + update_problem_find_llm, no update_find_costs call
  llm/                                  <-- new package
    __init__.py
    client.py             # thin wrapper over local OpenAI-compatible endpoint (Ollama)
    prompts.py            # state serialization + operator description + few-shot example
    plan_parser.py        # structured JSON -> Action objects
    validator.py           # forward-simulator: checks preconditions/applies effects per action
  planners/
    llm_planner.py         # LLMSequentialPlanner — the solve_from_pddl replacement
tests/
  test_llm_domain.py
  test_llm_prompts.py
  test_llm_validator.py
  test_llm_planner.py
```

## 4. Planner architecture

`LLMSequentialPlanner` must be a drop-in for the call site in `task_loop.py` /
`eval_replan.py`:

```python
plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'],
                              planner=pddl['planner'], max_planner_time=240)
```

becomes (behind a backend flag, e.g. `args.planner_backend in {'pddl', 'llm'}`):

```python
plan, cost = taskplan.llm.planner.solve_with_llm(
    pddl['domain'], pddl['problem_struct'], partial_map, args)
```

Requirements on the return value: `plan` must be a list of objects exposing `.name` and
`.args` (the same interface `pddlstream` actions expose), since `task_loop.py`'s dispatch
loop (`action.name == 'move'`, `action.args[0]`, etc., lines ~14–201) must work completely
unmodified. A `namedtuple('Action', ['name', 'args'])` is sufficient.

Flow inside `solve_with_llm`:
1. Serialize current state (graph, robot pose, held object, discovered contents, remaining
   goal predicates) into a prompt — **explicitly excluding** any expected-cost / probability
   signal (see prompting principles below).
2. Call the local Gemma 4 endpoint, requesting a structured JSON action list matching the
   LLM-facing operator set (`move`, `find`, `pick`, `place`, `pour-water`, `pour-coffee`,
   `make-coffee`, `boil`, `peel`, `toast`).
3. Parse the response (`plan_parser.py`) into `Action` objects.
4. Validate the sequence against a forward state-simulator (`validator.py`) that mirrors the
   LLM domain's preconditions/effects — reject or truncate at the first invalid or
   hallucinated (unknown object/location) action.
5. Return the validated prefix and its summed cost. Since `task_loop.py` already replans
   after every `find`, it's reasonable (and lower-risk) to only trust the plan up through the
   next `find` action, then let the existing replanning loop re-invoke the LLM — don't rely
   on one LLM call producing a full valid plan to the goal.

## 5. Prompting principles

- **Must not expose:** `find-cost` values, any expected-cost/frontier computation, GNN/FCNN
  probability outputs. This is the entire point of the baseline — it must reason about which
  container to search using only its own judgment from the natural-language scene
  description, not the architecture's cost model.
- **Should expose:** room/container graph structure and names, current robot location, held
  object, contents discovered so far, remaining goal predicates, and the operator set with
  preconditions described in plain language — follow the style already used in
  `object_search/learning/models/llm.py::generate_prompt_llm_as_planner`.
- **Output format:** request strict structured JSON (`{"action": ..., "args": [...]}` list)
  via Gemma 4's native structured-output/function-calling support rather than free-text
  parsing — this cuts a large class of parsing failures out entirely.

## 6. Validation / forward simulator

Non-optional. Removing FF-astar means losing its soundness guarantee — nothing else in the
pipeline checks that an LLM-proposed action sequence is actually legal. `validator.py` should
maintain a fact-set mirroring the LLM domain's predicates, and for each proposed action:
check preconditions hold, reject if not (including references to objects/locations that
don't exist in the current graph — a distinct failure mode from precondition violation), else
apply effects and continue.

## 7. Local serving setup

- Install Ollama, pull a Gemma 4 E4B QAT build, confirm it serves an OpenAI-compatible
  endpoint at `http://localhost:11434/api/chat`.
- `taskplan/llm/client.py` wraps this with the existing `openai` Python package already in
  `modules/requirements.txt` — just point `base_url` at the local endpoint, no real API key
  needed.
- Smoke-test with a hand-written prompt before wiring in `prompts.py`/`plan_parser.py`.

## 8. Evaluation plan

- Reuse the harness in `taskplan/scripts/eval_replan.py` and the existing seed sets; add an
  `eval-llm` Makefile target alongside `eval-known`, `eval-learned`, `eval-naive`.
- Track per seed: task success rate, total action cost, plan validity rate (fraction of LLM
  calls producing a directly valid sequence vs. requiring truncation/fallback), and
  wall-clock time per replanning call (local inference latency matters for a receding-horizon
  loop that calls the planner repeatedly).
- Compare Gemma 4 E4B vs. 12B (if VRAM allows) and thinking-mode on/off as ablations.

## 9. Phased implementation order

1. Domain/problem changes (Section 2) + `test_llm_domain.py` — verify FF-astar itself can
   still solve the simplified domain. Do this before writing any LLM code.
2. Ollama + Gemma 4 E4B running locally; a standalone smoke-test script confirming structured
   JSON output on a hand-written prompt.
3. `prompts.py` state serializer, unit-tested against a couple of synthetic seeds — confirm
   no expected-cost information leaks into the prompt.
4. `plan_parser.py` + `validator.py`, unit-tested against known-good and deliberately invalid
   plans.
5. Wire `LLMSequentialPlanner`/`solve_with_llm` into `task_loop.py` and `eval_replan.py`
   behind the backend flag; run a handful of seeds end-to-end.
6. Full eval sweep vs. PDDL/learned/naive baselines; tune model size / thinking-mode / prompt
   based on results.

## 10. Open risks to track

- **Context length**: large households (many rooms/containers/objects) may still get long
  even at Gemma 4's 128K context for E4B — decide on a truncation/summarization strategy
  before it bites you on the largest scenes.
- **Hallucinated references**: the validator must reject actions naming objects/locations
  not present in the current graph, not just precondition violations.
- **Determinism**: use low temperature and a fixed seed for reproducible eval comparisons.
- **VRAM contention**: if AI2-THOR rendering and the LLM server run concurrently on the same
  8GB card, they'll compete — may need to serialize (LLM eval after data generation
  finishes) rather than run both live.

## 11. Non-goals

- Do not modify the existing PDDL/learned/naive planners or `get_domain` — they must remain
  runnable, unmodified, as the comparison baselines.
- No fine-tuning in this phase — pure prompting/in-context first. Fine-tuning Gemma 4 on
  successful plans is a plausible future phase, not part of this one.


## 12. Existing repo related instruction

- Any changes to a file needs to be built before that change can be reflected when running
  a script or testing. The command is `make build`.
- Addition to building when any change is made currectly the taskplan repo has a `Makefile`
  that has all the targets to run different script and new targets should be added here.
- To run tests the command is `make test PYTEST_FILTER=TEST_NAME`