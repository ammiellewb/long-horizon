# `world_model.py` — executable world model + backtest-against-history

Port of the core technique from **Schema: Discovering Unknown Environments via Agentic Program
Induction** (Zeng et al., 2026; arXiv:2609.39140), from
`Schema/digbench/schema/agent/world_model/{world.py,tools.py}`.

The mechanism: the agent writes its understanding of the environment as an **executable Python
program** (`step(obs, action)` / `predict(state, obs, action)`, plus an optional
`is_win_condition`), and the harness then **backtests that program against the whole recorded
trajectory** to find where it mispredicts. The program is a falsifiable artifact — replay can
prove the model's claim about the environment wrong, which is what neither M3 (`state_ext.py`,
ground-truth probe of the *present* state) nor M6 (`planning.py`, unverified single-shot replan)
can do.

Status: **self-contained and tested, not wired in.** Nothing in this module imports or is imported
by `runner.py`. Integration was deliberately deferred until the `runner.py` refactor lands.

---

## 1. What the module gives you

| Symbol | Purpose |
| --- | --- |
| `compile_world_model(code, call_timeout_s=2.0, load_timeout_s=5.0)` | Load agent-authored source under the sandbox → `WorldModel`. Raises `SandboxError` / `SandboxTimeout`. |
| `WorldModel` | `.step(obs, action)`, `.predict(state, obs, action)`, `.init_state(entry_obs)`, `.is_win_condition(obs, state=None)`, `.is_bfs_goal(...)`, `.set_entry(entry)`, `.has_win_condition`, `.stateful` |
| `Transition` | One recorded `(obs, action) -> next_obs` with `flags`, `segment`, `meta` |
| `trajectory_from_history(history)` | Adapter for `runner._run_loop`'s in-memory `history` list |
| `trajectory_from_log(log)` | Adapter for a saved `logs/<run_id>.json` transcript |
| `backtest(model, trajectory, max_details=5)` | The falsification step. Returns per-step predicted-vs-actual, a mismatch list, an accuracy score and a rendered `report` |
| `WorldModelMechanism` | `.induce(...)` (one authoring round) and `.refine(...)` (induce → backtest → feed the earliest mismatch back → re-author) |
| `llm_author(llm_client)` | Adapts any object with `generate()` / `parse_completion()` into the injected `author_fn`. **The module never imports `llm.py`** — the client is passed in |
| `FLAG_NAMES` | `('progress_increased', 'score_increased', 'game_over', 'win')` |

`backtest()` returns:

```python
{
  'total': int, 'skipped': int, 'exact': int,
  'accuracy': float,          # exact / total, 0.0 when total == 0
  'green': bool,              # accuracy == 1.0 and total > 0
  'win_verified': int, 'warnings': int,
  'steps': [{'index', 'segment', 'action', 'predicted', 'actual', 'match',
             'predicted_flags', 'actual_flags', 'errors', 'kinds', 'warnings', 'ok'}],
  'mismatches': [...],        # subset of 'steps' with errors, in trajectory order
  'first_mismatch': dict|None,
  'report': str,              # the text fed back to the authoring model
}
```

---

## 2. Sandbox: what is allowed and what is blocked

`_ALLOWED_IMPORTS` and `_SAFE_BUILTINS` are **Schema's sets, copied unwidened**
(`world.py` lines 12–45).

* **Imports allowed (14):** `numpy, math, collections, itertools, functools, heapq, re, string,
  copy, dataclasses, typing, enum, operator, bisect`. Everything else (`os`, `sys`, `subprocess`,
  `socket`, `shutil`, `pickle`, `importlib`, …) is rejected — including relative imports and
  imports hidden inside a function body.
* **Builtins:** only the 59-name allowlist. Absent, therefore unavailable: `eval`, `exec`,
  `compile`, `open`, `input`, `__import__` (replaced by the allowlisting `_safe_import` hook),
  `globals`, `locals`, `vars`, `dir`, `breakpoint`, `help`, `exit`, `quit`, `memoryview`, `print`.
* **Two local additions** (hardening, not widening):
  1. A static AST pre-check runs *before any code executes*: it rejects non-allowlisted imports,
     references to blocked builtin names, and dunder-traversal escape vectors
     (`__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__mro__`, `__code__`,
     `__dict__`, `__reduce__`, `f_globals`, …) whether written as attribute access or as a string
     literal handed to `getattr`. Schema's allowlist leaves `getattr`/`type`/`object`/`super`
     available, so `().__class__.__bases__[0].__subclasses__()` would otherwise walk back to the
     real builtins. Known false positives, accepted on purpose: legitimate uses of
     `self.__dict__` / `obj.__module__` are rejected, and so is a program whose *source* contains
     one of those names as a plain string literal. Defining `__init__` / `__post_init__` methods
     is fine (a `def` is not an attribute access).
  2. A **wall-clock timeout** on every entry into model code (module-level execution,
     `init_state`, `step`/`predict`, the predicates). Implemented as a traced worker thread: a
     per-line trace hook raises a `BaseException` subclass once the budget is spent (so
     `while True: try: ... except Exception: pass` cannot swallow its own timeout), and if the
     thread is still alive after the budget plus a 1 s grace period — possible when blocked inside
     an untraceable C call — it is abandoned as a daemon and `SandboxTimeout` is raised anyway, so
     the caller never hangs.

**Threat model — read this before enabling it on a shared machine.** This is an in-process
allowlist sandbox, the same class of protection Schema's `_exec_code` provides, and in-process
Python sandboxes are not a hard security boundary. Schema backs its own sandbox with OS-level
isolation that is **not ported here**: `landlock_exec.py` (Linux Landlock filesystem restriction),
bubblewrap for MazeBench's `run_python`, and `--jail` running Claude Code and the agent's code in
podman/enroot/docker containers. Those are Linux-only and would need a container story this repo
does not have. Treat `world_model.py` as "defence in depth against a careless model", not
"containment of a hostile one". A memory/recursion cap is also not implemented — a model that
allocates a huge list will be stopped by the timeout, but only after it has allocated.

---

## 3. Integration hook the runner needs

### 3.1 Trajectory format to pass in

`backtest()` consumes a `list[Transition]`. Two things matter about the runner's existing data:

* `runner._run_loop` appends `{'obs', 'action', 'reasoning', 'progress'}` where **`obs` is the
  observation the agent acted ON** and `progress` is the value **after** the action. The resulting
  observation of step *i* is therefore the `obs` of step *i+1*. `trajectory_from_history()` does
  that pairing and drops the final step (no recorded successor).
* The same pairing applies to `logs/<run_id>.json` → `steps[]`; use `trajectory_from_log()`.

Verified against a real transcript (`experiments/logs/01f5ddbb…json`, 100 steps → 99 transitions,
backtested in 0.01 s).

> Checked against `runner.py` **while it was being refactored in parallel** (the `history.append`
> call still has the shape above, now around line 778). `config.yaml` and `logger.py` are also
> being edited concurrently — the concurrent work is about model identity, stop reasons and
> context truncation, so none of the keys suggested below collide, but re-confirm the line numbers
> before wiring.

### 3.2 Call sites in `runner.py`

```python
# imports (top of runner.py, alongside the other mechanism imports)
from mechanisms.world_model import (
    WorldModelMechanism, llm_author, trajectory_from_history,
)

# _build_mechanisms(), next to mech['state_ext'] / mech['planning']
mech['world_model'] = WorldModelMechanism(
    game=self.game_name,
    max_rounds=self.m7_max_rounds,
    max_history_steps=self.m7_history_steps,
) if self.m7_enabled else None

# __init__, next to self.m6_enabled
wm_cfg = config.get('world_model', {}) or {}
self.m7_enabled      = mech_cfg.get('m7_world_model', False)
self.m7_max_rounds   = wm_cfg.get('max_rounds', 3)
self.m7_history_steps = wm_cfg.get('max_history_steps', 40)
self.m7_every        = wm_cfg.get('induce_every', 25)
self.m7_min_history  = wm_cfg.get('min_history', 10)
```

Inside `_run_loop`, in the **dispatch** section, *before* the M6 branch and gated on the same
`bundle` that M4 produces (so the expensive authoring round only happens when a channel says the
agent is in trouble):

```python
world_model = mech.get('world_model')

# --- Dispatch: M7 (world-model induction / re-induction) ---
if (world_model is not None
        and len(history) >= self.m7_min_history
        and (step % self.m7_every == 0 or bundle['trigger_replan_from_drift'] or is_ale)):
    trajectory = trajectory_from_history(history)
    model, bt, wtoks = world_model.refine(llm_author(self.llm), trajectory)
    judge_tokens += wtoks                      # same token accounting as M6
    if bt is not None:
        logger.add_step(...)                   # see 3.3 for the fields
        mechanisms_used.append('m7_world_model')
        if not bt['green']:
            # the earliest falsification is the most useful thing to show the agent
            wm_note = bt['report']             # feed into build_step_messages / memory
```

What it needs passed in, precisely:

| Needs | Source in `_run_loop` today |
| --- | --- |
| `history` (before-obs/action pairs) | the `history` list already threaded through the loop |
| an LLM client | `self.llm` — wrapped by `llm_author(self.llm)`; nothing else in the module can reach the network |
| step index, for the cadence gate | the `for step in range(...)` variable |
| the risk `bundle` | `mech['adaptive'].decide(risk_vector)` output |
| *(optional)* a segment id | `0` today; bump it on resume/fork so state re-inits per segment |

`refine()` returns `(model, result, tokens)` and never raises — a program that will not load, a
timeout, or an `author_fn` that throws all come back as a recorded error, which is exactly the
feedback the next round consumes.

### 3.3 Where it slots among the existing switches

* **Relative to M3 (`state_ext`)**: complementary, not competing. M3 measures the present
  (location/inventory via a read-only probe). M7 models the *dynamics*. The natural upgrade once
  both are on is to let M3's probe supply the ground-truth flags that `backtest()` currently has
  to infer from progress deltas (see §4, flag coverage).
* **Relative to M6 (`planning`)**: same trigger surface, strictly stronger artifact. M6's
  `replan()` produces an unverified sentence; M7 produces a program that gets replayed against
  history. The cheapest real experiment is a three-arm comparison on the same seed —
  `m6_planning` alone / `m7_world_model` alone / both — driven by
  `run_mechanism_comparison.py`. If both run in the same step, run M7 first and pass
  `bt['report']` (or `bt['first_mismatch']`) into `PlanningMechanism.replan()` as the
  `suggested_combination` slot, so the replan is grounded in a concrete falsification.
* **Dispatch priority**: M1 (lead) > M7 (model repair) > M6 (replan) > normal generation, because a
  falsified world model is hard evidence while the integration/drift signals are heuristics.
* **Config** (`config.yaml`, under `experiments:` — *not edited by this change*):

```yaml
  mechanisms:
    m7_world_model: false      # candidate: executable world model + backtest

  world_model:
    max_rounds: 3              # induce -> backtest -> re-author rounds per trigger
    max_history_steps: 40      # transitions rendered into the induction prompt
    induce_every: 25           # re-induce at least this often
    min_history: 10            # don't bother before this many recorded steps
    call_timeout_s: 2.0        # per-call wall-clock budget for model code
    load_timeout_s: 5.0        # module-level execution budget
```

* **Logging** (`logger.py`, *not edited*): add a `world_model_events` list, mirroring
  `add_drift_event` / `add_integration_event`, with
  `{'step', 'rounds', 'accuracy', 'total', 'exact', 'mismatches', 'win_verified', 'green',
  'first_mismatch_index', 'code'}`. Store the program text once per change, not once per step —
  it is the single most valuable research artifact this mechanism produces (it is what lets you
  ask *what did the agent actually believe about Zork, and where was it wrong*). Until that lands,
  `WorldModelMechanism.rounds` holds the per-round record in memory.
* **Authoring prompt max_tokens**: `llm_author()` defaults to `max_tokens=2048`. The runner's
  normal generation uses `192` and `planning.replan()` uses `96`; a world-model program does not
  fit in either. Budget for it — this is the most expensive mechanism in the harness.

---

## 4. What did **not** port, and why

### 4.1 Blocked by the harness difference (chat completions vs. coding CLI)

Schema's agent does not talk to a chat-completions endpoint at all. Its drivers
(`claude_driver.py`, `codex_cli_driver.py` / `codex_driver.py`) drive **Claude Code and the Codex
CLI as agentic processes** with a filesystem workdir, a tool loop, and multi-hour sessions.
This repo's `llm.py` is a single-shot OpenRouter `chat/completions` POST with no tools, no
workdir and no session. Consequences:

* **The tool loop is gone.** Schema's agent calls `write_file` to save `world_model.py`, then
  `run_backtest`, then `read_history`, then `run_bfs`, iterating *by itself* over however many
  turns it needs, with `events.jsonl` and `--resume` for continuity. Here the loop had to be
  inverted: `WorldModelMechanism.refine()` runs a **fixed, harness-driven** number of
  induce→backtest→re-author rounds and does the feeding-back itself. The agent does not choose
  when to re-examine its model; the runner does. This is the single biggest fidelity loss, and it
  is a direct consequence of having no tool-calling loop.
* **No workdir artifact.** Schema keeps the program on disk (`world_model.py`, agent notes,
  `events.jsonl`) so a run is resumable and the program evolves across turns. Here the program
  lives in memory on `WorldModelMechanism`; persisting it needs the `logger.py` hook in §3.3.
* **`run_backtest`'s selection filters** (`indices`, `start`/`end`, `level`, `mode`, `segment`,
  `max_details`) exist because the *agent* chooses what slice of history to interrogate. With no
  tool loop there is nobody to choose, so `backtest()` always scores the whole trajectory it is
  given. Slicing is trivially done by the caller (`traj[a:b]`).
* **Effort/reasoning controls** (`--effort max`, `--reasoning high`, `--service-tier`) are
  CLI-driver concepts with no OpenRouter equivalent beyond `reasoning_enabled`.

### 4.2 Deliberately out of scope for this module

* **The planner.** `world.py`'s `bfs()` (and MazeBench's A*) searches *inside* the induced model
  to produce a plan, and Schema then executes that plan under step-by-step verification. That is
  the other half of the paper and a much larger piece of work; porting it needs an action
  enumeration for a text adventure (Schema's bench hands the agent the legal action list every
  state, Zork does not) and a plan-execution path in `runner.py`, which is off-limits right now.
  `is_bfs_goal` is kept on `WorldModel` for contract parity and is explicitly **never scored** —
  it is a planner hook with no planner behind it yet.
* **OS-level jail**: `landlock_exec.py`, bubblewrap, `--jail` containers. Linux-only; see the
  threat model in §2.
* **`run_python` / `run_shell`.** Schema also lets the agent run arbitrary scratch code against
  its model. Not ported: it widens the attack surface a long way for no benefit until there is a
  planner.

### 4.3 Semantics that do not exist in TextQuests

* **Levels, lives, step budgets, creative mode.** Schema's flags are
  `('level_up', 'life_lost', 'dead', 'win')` and its leniency rules are built on DiG-bench events
  (`budget_exhausted`, `challenge_outcome`, creative toggles, the bench's "final board" reporting
  on boundary steps). Zork has none of that. `FLAG_NAMES` became
  `('progress_increased', 'score_increased', 'game_over', 'win')`, matching what `env.step`/the
  logger actually record, and the boundary-step/final-board scoring logic is dropped.
* **Flag coverage is thin, and the backtest is lenient about it.** Schema's bench asserts every
  flag on every step, so a wrong flag is always a mismatch. This repo's recording only reliably
  gives progress and score deltas, so `_flag_mismatches()` checks **only flags the trajectory
  actually asserts** — an absent flag is *unknown*, not False, and predicting it is never
  penalised. That means `game_over` / `win` are effectively unscored today, and
  `is_win_condition` can only be confirmed if a transition carries `flags={'win': True}`. To close
  this, `runner.py` would need to record `game_over` (it already has it from `env.step`) per step.
  **Read accuracy numbers with that in mind: a green backtest today means "reproduced the
  observation text", not "reproduced the full outcome".**
* **Embedded-history prefix leniency.** Schema accepts a prediction that matches the `(current)`
  frame but differs in the `(previous)` prefix, because some DiG-bench games embed history in the
  observation. Text adventures do not. The analogous leniency implemented here is a
  whitespace/case-insensitive match, scored as a warning (`match == 'normalized'`,
  `kinds == ['obs_soft']`) rather than a mismatch. This is an **adaptation, not a port** — judge it
  on its own merits.
* **Exact-string prediction is much harder here than in DiG-bench.** Schema's observations are
  rendered grids; Zork's are prose with randomised flavour text, so an exact-match accuracy score
  is a harsh and noisy metric. A naive "nothing ever changes" model scores 0.13 on a real 99-step
  Zork log, which is a useful floor but shows how much of the signal is formatting. If the metric
  turns out to be too brittle to drive the loop, the honest fix is a semantic/structural match
  predicate (replacing `match_level()`), not loosening the sandbox or the flags.
* **Segments.** Schema cuts the run at every level clear / lost life / mode switch and re-inits
  the model's state at each boundary. The `segment` field on `Transition` preserves that
  behaviour, but every adapter currently emits `segment=0`, since a TextQuests episode is one
  continuous run. The runner should bump it on resume/fork.
* **`aligned_diff`** (Schema's caret-annotated grid diff) is grid-specific; replaced by
  `diff_summary()` (ported) plus raw `repr`s of predicted/actual in the report.

---

## 5. Tests

`experiments/tests/test_world_model.py` — 37 tests, fully offline (the authoring LLM is a stub
callable; no network, no API key, no paid call). Style matches `tests/test_harness.py`.

`pytest` is not installed in any interpreter on this machine, so the file carries a minimal
self-runner:

```
"/Users/ammiellewambobecker/github/BASE Project/.venv-inspect/bin/python" \
    experiments/tests/test_world_model.py
```

Last run: **37 passed, 0 failed**. Coverage: disallowed imports (`os`, `sys`, `subprocess`,
`socket`, `shutil`, `pickle`, `importlib`, `from os import …`, imports hidden in a function body);
dangerous builtins (`eval`, `exec`, `open`, `compile`, `globals`, `vars`, `__import__("os")`);
dunder-traversal escapes; the `_safe_import` hook; allowlisted imports still working; timeout on an
infinite loop in `step`, on a module-level infinite loop, and on a loop that swallows
`Exception`; a timeout recorded as a mismatch rather than a crash; both trajectory adapters;
mismatch identification on a hand-written toy trajectory (exact index, predicted vs actual, kind);
a perfect model scoring 1.0/green; the normalized-match warning; flag mismatches and flag
leniency; `is_win_condition` verification and falsification; model exceptions; the stateful
`predict` contract with per-segment state re-init; `ENTRY_OBS`; and the full
induce→falsify→re-author loop with a stub author (including early stop on green, feeding the
earliest mismatch back, keeping the most accurate model, unloadable code, and an author that
raises).
