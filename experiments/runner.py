"""Experiment runner.

A Runner manages one game + one LLM client + one harness configuration.
Call run_once() to execute a single episode and get back a run_id that can
be used to look up the transcript written to out_dir.

Run directly to execute one episode with the settings in config.yaml:

    cd experiments
    python runner.py

Or specify a game name as an argument:

    python runner.py wishbringer

Three properties this module is responsible for, each of which was previously
left implicit and silently wrong:

* **Stop reason.** Every run records exactly one ``stop_reason`` from
  ``logger.STOP_REASONS``. RunLogger.save() refuses to write a transcript
  without one, so a run that died on an API failure can never again be
  aggregated as if it had honestly exhausted its step budget.
* **Context truncation.** History is kept inside a token budget by dropping the
  oldest whole observation/action pairs, and a provider context-overflow error
  is recovered from by evicting pairs sized to the reported deficit. This is
  *truncation, not summarization* — see ``truncate_history_if_needed``.
* **Model identity.** ``resolve_model()`` resolves the model id from one
  documented precedence order and refuses to run when the environment and
  config.yaml disagree.
"""

import os
import sys
import uuid
from pathlib import Path

import yaml

# Ensure the experiments/ directory is in sys.path so sub-packages resolve
# whether the script is run as 'python runner.py' from inside experiments/
# or as 'python experiments/runner.py' from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent import build_step_messages, format_history
from channels import (
    AcceptedLocalErrorDetector,
    BudgetLivenessDetector,
    DriftDetector,
    IntegrationDetector,
    LLMCoherenceJudge,
    LLMIntegrationJudge,
    make_env_checker,
)
from env import find_game_path, load_env_class, resolve_root
from llm import (
    OpenRouterClient,
    extract_step_fields,
    is_context_overflow,
    parse_overflow_tokens,
)
from logger import RunLogger
from mechanisms import (
    AdaptiveComputeController,
    InfoSeekingMechanism,
    MemoryMechanism,
    PlanningMechanism,
    StateExternalizer,
    normalize_action,
)

# ---------------------------------------------------------------------------
# Model identity: one resolution order, no silent fallbacks
# ---------------------------------------------------------------------------
# Environment variables that are allowed to name a model. OPENROUTER_MODEL is a
# source (it is this runner's own override); INSPECT_EVAL_MODEL is NOT — it
# configures inspect-ai — but it is still cross-checked, because a stale value
# there is exactly the kind of thing that silently diverges from config.yaml.
MODEL_ENV_VARS = ('OPENROUTER_MODEL', 'INSPECT_EVAL_MODEL')
_MODEL_SOURCE_ENV_VARS = ('OPENROUTER_MODEL',)
_OPENROUTER_ROUTE_PREFIX = 'openrouter/'


class ModelConfigError(RuntimeError):
    """The configured model id is missing, or the sources disagree about it."""


def _canonical_model(model_id):
    """Strip the provider route so ids spelled by different tools compare equal.

    inspect-ai spells the same model ``openrouter/meta-llama/llama-3.2-3b-instruct:free``
    where config.yaml spells it ``meta-llama/llama-3.2-3b-instruct:free``. Only
    the ``openrouter/`` route is stripped — this runner talks to exactly one
    provider, so nothing else is a route — and only when a ``vendor/model`` pair
    remains underneath, so a model genuinely published as ``openrouter/<name>``
    is left alone.
    """
    model_id = (model_id or '').strip()
    if model_id.lower().startswith(_OPENROUTER_ROUTE_PREFIX):
        remainder = model_id[len(_OPENROUTER_ROUTE_PREFIX):]
        if '/' in remainder:
            return remainder
    return model_id


def resolve_model(cfg, env=None):
    """Resolve the one model id this run will use. Returns ``(model_id, source)``.

    PRECEDENCE, highest first:

      1. ``OPENROUTER_MODEL``             — the operator's override for this process
      2. ``openrouter.model`` in config.yaml — the checked-in default

    Every variable in :data:`MODEL_ENV_VARS` is additionally CROSS-CHECKED
    against config.yaml regardless of precedence: if any of them names a
    different model, this raises :class:`ModelConfigError` instead of picking
    one. Precedence only ever resolves *which spelling to record* among sources
    that already agree — a disagreement is always a hard stop, because the
    failure mode being fixed here is a run that reported one model in its config
    and sent its prompts to another.

    Raises:
        ModelConfigError: if the sources disagree, or if none names a model.
    """
    env = os.environ if env is None else env

    cfg_model = ((cfg or {}).get('openrouter') or {}).get('model')
    config_entry = ('config.yaml:openrouter.model', cfg_model) if (cfg_model or '').strip() else None
    env_entries = [
        (f'env:{name}', env.get(name))
        for name in MODEL_ENV_VARS
        if (env.get(name) or '').strip()
    ]

    present = ([config_entry] if config_entry else []) + env_entries
    if len({_canonical_model(raw) for _, raw in present}) > 1:
        detail = '\n'.join(f'    {label} = {raw!r}' for label, raw in present)
        raise ModelConfigError(
            'Model id mismatch between configuration sources — refusing to run.\n'
            + detail + '\n'
            'All of these must name the same model (the leading "openrouter/" route\n'
            'is ignored when comparing). Reconcile them and re-run. This runner will\n'
            'not choose one for you: which model the experiment is on is a research\n'
            'decision, and a silently-resolved mismatch is how run logs come to\n'
            'disagree with the configuration they were supposedly produced under.'
        )
    if not present:
        raise ModelConfigError(
            'No model id configured. Set openrouter.model in experiments/config.yaml, '
            'or export OPENROUTER_MODEL. There is deliberately no default.'
        )

    # The id returned is the canonical one — the route prefix is not part of the
    # model id the provider accepts, so it must not reach the request or the log.
    for label, raw in env_entries:
        if label in (f'env:{name}' for name in _MODEL_SOURCE_ENV_VARS):
            return _canonical_model(raw), label
    if config_entry:
        return _canonical_model(config_entry[1]), config_entry[0]
    # Only a cross-checked-but-not-a-source variable is set (today:
    # INSPECT_EVAL_MODEL). Its value agrees with nothing else because nothing
    # else is set, and its "openrouter/..." spelling is an inspect-ai route, not
    # an OpenRouter model id — so make the researcher say which they meant.
    names = ', '.join(label.split(':', 1)[1] for label, _ in env_entries)
    raise ModelConfigError(
        f'{names} is set but is not a model source for this runner (it configures '
        'inspect-ai). Set openrouter.model in experiments/config.yaml, or export '
        'OPENROUTER_MODEL, to the model this experiment should run on.'
    )


# ---------------------------------------------------------------------------
# Rolling-window context truncation
# ---------------------------------------------------------------------------
# Ported from dig-bench baseline-harness core/history.py. The evictable unit
# here is one entry of the runner's ``history`` list — a whole
# observation/action pair, exactly as agent.format_history() renders it.
#
# THIS IS TRUNCATION, NOT SUMMARIZATION, AND THAT IS A METHODOLOGICAL
# REQUIREMENT, NOT A SIMPLIFICATION. This repo ablates a separate M2 memory
# mechanism. If the truncation path condensed, summarized, or wrote evicted
# pairs back into the M2 scratchpad, the baseline harness would gain memory help
# through the back door and the M2 ablation would be confounded. Evicted pairs
# are dropped and nothing is derived from them.
#
# Invariants (same as dig-bench):
#   - evict only WHOLE, oldest, completed pairs;
#   - the first/seed pair is PINNED and never evicted;
#   - the newest pair is never evicted (the in-flight turn is not in `history`
#     at all — the live observation is passed alongside it — so the newest
#     completed pair is the closest analogue of dig-bench's active unit);
#   - keep at least MIN_KEEP_PAIRS pairs;
#   - evict the FEWEST pairs needed to get under budget.
PINNED_HEAD_PAIRS = 1   # history[0] is the seed turn: pinned, never evicted
MIN_KEEP_PAIRS = 2      # the pinned head + the newest pair, always retained

MAX_OVERFLOW_ROUNDS = 8  # evict-and-retry rounds after a provider overflow error
_OVERFLOW_MARGIN = 1.25  # est_pair_tokens is a chars//4 proxy — overshoot the deficit by 25%
_OVERFLOW_SLACK = 1000   # absolute headroom on top, for whatever the retry itself appends

# Reserve for one turn's output, in tokens: ~1.5x the 192-token per-step cap.
# Output counts toward the hard context window, and truncation lags one turn
# (its trigger, the server-reported prompt size, is only known after a call).
DEFAULT_OUTPUT_RESERVE = 288


def est_pair_tokens(pair):
    """Cheap size proxy (chars // 4) for one history pair, for MINIMAL eviction.

    Covers the fields agent.format_history() actually renders into the prompt.
    This is a size estimate for deciding how much to drop, not accounting — the
    next call's real server-reported prompt size corrects any drift.
    """
    if not isinstance(pair, dict):
        return 0
    return sum(len(str(pair.get(key, '') or '')) for key in ('obs', 'action', 'reasoning')) // 4


def resolve_context_budget(window_tokens, output_reserve=DEFAULT_OUTPUT_RESERVE, budget_tokens=None):
    """The prompt-token budget for proactive truncation, or None when disabled.

    ``budget_tokens`` pins an absolute value (dig-bench's ``--context-budget``).
    Otherwise the budget is the model's own context window minus a fixed reserve
    for one turn's output, so a model keeps as much history as its window allows
    while every model reserves the same absolute output room.

    With neither configured, proactive truncation is OFF and only the reactive
    overflow-recovery path runs. That is the safe default for a repo whose
    config does not pin a model: we cannot know a window we were not told.
    """
    if budget_tokens:
        return int(budget_tokens)
    if window_tokens:
        return max(1, int(window_tokens) - int(output_reserve or 0))
    return None


def truncate_history_if_needed(history, last_prompt_tokens, budget, log=None):
    """Shrink ``history`` in place if the LAST call's prompt exceeded ``budget``.

    Returns the number of pairs evicted. The trigger is necessarily reactive —
    the prompt size is only known after a call — so we evict before the next
    one. Eviction is minimal: the projected prompt size is reduced by each
    evicted pair's ``est_pair_tokens`` and we stop as soon as the projection is
    under budget, never reflexively down to the floor.
    """
    if not budget or budget <= 0:          # truncation disabled / no resolvable window
        return 0
    if last_prompt_tokens <= 0:            # first-step bootstrap: no server count yet
        return 0
    if last_prompt_tokens <= budget:
        return 0

    projected = last_prompt_tokens
    evicted = 0
    while projected > budget and len(history) > MIN_KEEP_PAIRS:
        oldest = history.pop(PINNED_HEAD_PAIRS)
        projected -= max(0, est_pair_tokens(oldest))
        evicted += 1

    if evicted and log:
        over = '' if projected <= budget else (
            f' > budget {budget} (floor MIN_KEEP_PAIRS={MIN_KEEP_PAIRS} reached, still over)'
        )
        log(f'context truncated: evicted {evicted} oldest pair(s), projected prompt '
            f'~{projected}{over or f" <= budget {budget}"}, kept {len(history)} pair(s)')
    return evicted


def evict_for_overflow(history, err, round_idx=0):
    """Evict enough oldest pairs to cover the deficit a provider overflow reported.

    Returns the number of pairs evicted; 0 means eviction cannot shrink the
    prompt any further (only the pinned head and the newest pair remain) and the
    caller must give up with ``context_exhausted``.

    Overflow wording usually states the exact numbers, so we evict oldest pairs
    until their summed ``est_pair_tokens`` covers the reported deficit plus a
    margin. Sizing to the deficit matters: the oldest pairs are the smallest, so
    a fixed per-retry count starves against a multi-thousand-token overshoot.
    When the wording carries no counts, fall back to a doubling batch
    (``2 ** round_idx``) so repeated rounds still converge.
    """
    parsed = parse_overflow_tokens(err)
    target = None
    if parsed:
        total, limit = parsed
        target = int((total - limit) * _OVERFLOW_MARGIN) + _OVERFLOW_SLACK

    evicted = 0
    removed_est = 0
    while (removed_est < target) if target is not None else (evicted < 2 ** round_idx):
        if len(history) <= MIN_KEEP_PAIRS:
            break
        removed_est += max(0, est_pair_tokens(history.pop(PINNED_HEAD_PAIRS)))
        evicted += 1
    return evicted


class Runner:
    """Run one game episode under a given harness configuration.

    Parameters
    ----------
    game_name : str
        Name of the TextQuests game (e.g. 'zork1', 'wishbringer').
    out_dir : Path-like
        Directory where per-run JSONL logs are written.
    llm_client : OpenRouterClient
        Pre-configured LLM client.
    config : dict
        Experiment config dict (from the 'experiments' key in config.yaml).
    textquests_root : Path-like, optional
        Override for the textquests checkout path. Defaults to the submodule.
    model_source : str, optional
        Where the model id came from, as returned by resolve_model(). Recorded
        in the run log alongside the id itself.
    """

    def __init__(self, game_name, out_dir, llm_client, config, textquests_root=None,
                 model_source=None):
        self.game_name = game_name
        self.out_dir = Path(out_dir)
        self.max_steps = config.get('max_steps', 500)
        self.ale_k = config.get('accepted_local_error_k', 3)
        self.ale_window = config.get('info_seek_window', 10)
        self.textquests_root = (
            Path(textquests_root).expanduser().resolve() if textquests_root else None
        )

        # Cost cap. This repo meters tokens, not USD, so max_total_tokens is the
        # seam dig-bench spells --max-cost-usd; either way the stop is recorded
        # as 'cost_cap'. None = no cap (the previous behaviour).
        self.max_total_tokens = config.get('max_total_tokens')
        # Consecutive steps whose model response yields no usable action before
        # the run is declared wedged. None/0 = no cap.
        self.max_invalid_actions = config.get('max_invalid_actions', 10)

        ctx_cfg = config.get('context', {}) or {}
        self.context_window_tokens = ctx_cfg.get('window_tokens')
        self.context_output_reserve = ctx_cfg.get('output_reserve', DEFAULT_OUTPUT_RESERVE)
        self.context_max_overflow_rounds = ctx_cfg.get('max_overflow_rounds', MAX_OVERFLOW_ROUNDS)
        self.context_budget = resolve_context_budget(
            self.context_window_tokens,
            self.context_output_reserve,
            ctx_cfg.get('budget_tokens'),
        )

        drift_cfg = config.get('drift', {}) or {}
        self.drift_enabled      = drift_cfg.get('enabled', True)
        self.drift_window       = drift_cfg.get('window', 20)
        self.drift_threshold    = drift_cfg.get('threshold', 0.65)
        self.drift_w_stagnation = drift_cfg.get('w_stagnation', 0.5)
        self.drift_w_loop       = drift_cfg.get('w_loop', 0.3)
        self.drift_w_coherence  = drift_cfg.get('w_coherence', 0.2)

        mem_cfg = config.get('memory', {}) or {}
        self.memory_max_visited     = mem_cfg.get('max_visited', 25)
        self.memory_max_discoveries = mem_cfg.get('max_discoveries', 10)

        budget_cfg = config.get('budget', {}) or {}
        self.budget_enabled        = budget_cfg.get('enabled', True)
        self.budget_token_budget   = budget_cfg.get('token_budget')
        self.budget_stall_tokens   = budget_cfg.get('stall_token_threshold', 4000)
        self.budget_w_step         = budget_cfg.get('w_step', 0.3)
        self.budget_w_token        = budget_cfg.get('w_token', 0.3)
        self.budget_w_stall        = budget_cfg.get('w_stall', 0.4)
        self.budget_threshold      = budget_cfg.get('threshold', 0.7)

        integration_cfg = config.get('integration', {}) or {}
        self.integration_enabled       = integration_cfg.get('enabled', True)
        self.integration_min_unused    = integration_cfg.get('min_unused', 2)
        self.integration_recent_window = integration_cfg.get('recent_window', 10)

        mech_cfg = config.get('mechanisms', {}) or {}
        self.m1_enabled = mech_cfg.get('m1_info_seeking', True)
        self.m2_enabled = mech_cfg.get('m2_memory', True)
        self.m3_enabled = mech_cfg.get('m3_state_ext', True)
        self.m4_enabled = mech_cfg.get('m4_adaptive', True)
        self.m5_enabled = mech_cfg.get('m5_action_templating', True)
        self.m6_enabled = mech_cfg.get('m6_planning', True)

        self.logger = RunLogger(self.out_dir)
        self.llm = llm_client
        # Recorded in every transcript: the id actually sent to the provider,
        # and which configuration source it was resolved from.
        self.model = getattr(llm_client, 'model', None)
        self.model_source = model_source
        self._env_class = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_env_class(self):
        if self._env_class is None:
            self._env_class = load_env_class(self.textquests_root)
        return self._env_class

    def _new_env(self, seed):
        game_path = find_game_path(self.game_name, self.textquests_root)
        return self._get_env_class()(str(game_path), seed=seed)

    def _build_mechanisms(self):
        """Instantiate the per-episode detector/mechanism set from config flags.

        Returns a dict of optional components; any entry is None when its
        governing mechanism/channel is disabled, so _run_loop only has to
        check truthiness rather than re-reading config.
        """
        mech = {}

        mech['drift_detector'] = DriftDetector(
            window=self.drift_window,
            threshold=self.drift_threshold,
            w_stagnation=self.drift_w_stagnation,
            w_loop=self.drift_w_loop,
            w_coherence=self.drift_w_coherence,
        ) if self.drift_enabled else None
        mech['coherence_judge'] = LLMCoherenceJudge() if self.drift_enabled else None

        mech['memory'] = MemoryMechanism(
            max_visited=self.memory_max_visited,
            max_discoveries=self.memory_max_discoveries,
        ) if self.m2_enabled else None

        mech['state_ext'] = StateExternalizer() if self.m3_enabled else None

        mech['budget_detector'] = BudgetLivenessDetector(
            max_steps=self.max_steps,
            token_budget=self.budget_token_budget,
            stall_token_threshold=self.budget_stall_tokens,
            w_step=self.budget_w_step,
            w_token=self.budget_w_token,
            w_stall=self.budget_w_stall,
            threshold=self.budget_threshold,
        ) if self.budget_enabled else None

        mech['integration_detector'] = IntegrationDetector(
            min_unused=self.integration_min_unused,
            recent_window=self.integration_recent_window,
        ) if self.integration_enabled else None
        mech['integration_judge'] = LLMIntegrationJudge() if self.integration_enabled else None

        mech['adaptive'] = AdaptiveComputeController() if self.m4_enabled else None
        mech['planning'] = PlanningMechanism(window=self.ale_window) if self.m6_enabled else None

        return mech

    def _generate_with_overflow_recovery(self, history, observation, scratchpad, logger, step):
        """llm.generate() for one step, recovering from a context-overflow error.

        On an overflow the oldest whole observation/action pairs are evicted —
        sized to the deficit the provider itself reported — the prompt is rebuilt
        from the shortened history, and the call is retried, bounded by
        self.context_max_overflow_rounds. Ported from dig-bench
        core/harness._generate_recovering_overflow().

        Evicted pairs are DROPPED, never summarized and never written into the
        M2 scratchpad: see the truncation notes at module level.

        Returns ``(response, info, error)``. ``info`` carries ``evicted``,
        ``rounds`` and, when the call ultimately failed, the ``stop_reason`` to
        record — 'context_exhausted' if eviction ran out of room, 'api_failure'
        for anything else.
        """
        info = {'evicted': 0, 'rounds': 0, 'stop_reason': None}
        for round_idx in range(self.context_max_overflow_rounds + 1):
            messages = build_step_messages(self.game_name, history, observation,
                                           scratchpad=scratchpad)
            try:
                resp = self.llm.generate(messages, max_tokens=192, reasoning_enabled=False)
            except KeyboardInterrupt:
                raise          # an interrupt is its own stop reason, never an api_failure
            except Exception as exc:
                if not is_context_overflow(exc):
                    info['stop_reason'] = 'api_failure'
                    return None, info, exc
                if round_idx >= self.context_max_overflow_rounds:
                    info['stop_reason'] = 'context_exhausted'
                    return None, info, exc
                evicted = evict_for_overflow(history, exc, round_idx=round_idx)
                if not evicted:
                    # Only the pinned seed pair and the newest pair remain:
                    # eviction cannot shrink this prompt any further.
                    info['stop_reason'] = 'context_exhausted'
                    return None, info, exc
                info['evicted'] += evicted
                info['rounds'] += 1
                message = (f'context overflow — evicted {evicted} oldest pair(s), '
                           f'retrying ({info["rounds"]})')
                logger.add_truncation_event({'step': step, 'event': message,
                                             'kept_pairs': len(history)})
                print(f'[overflow] step {step:>4}  {message}')
            else:
                return resp, info, None

    def _run_loop(self, env, logger, run_id, history, observation, info, progress, score,
                  start_step, detector, env_checker, mech, info_seeking=None, capture_fork=False):
        """Run steps [start_step, self.max_steps) on a live env.

        Per-step pipeline:
          1. Check for a confirmed Accepted-Local-Error (channel, always
             evaluated) and the cheap integration/budget signals (channels).
          2. Ask mech['adaptive'] (M4) to turn those signals into a mechanism
             bundle for this step (many-to-many: e.g. sustained drift alone
             can trigger a replan with no ALE and no integration gap).
          3. Dispatch: M1 intervention (lead, takes priority) > M6 replan
             (integration-gap primary, high-drift secondary) > normal
             generation. M5 normalizes whatever action results.
          4. Update M2/M3/drift/budget bookkeeping and log the step.

        Before each generation the history is rolled to the context budget, and
        a provider context-overflow is recovered from by evicting oldest pairs
        (see _generate_with_overflow_recovery).

        EXACTLY ONE stop_reason is set on every exit from the loop:
          solved / game_over  — env reported game_over (>=100% progress or not)
          step_cap            — the loop ran to self.max_steps
          cost_cap            — self.max_total_tokens reached
          api_failure         — the provider call failed, not recoverably
          context_exhausted   — overflow that eviction could no longer shrink
          invalid_action_cap  — too many consecutive unusable model responses
          env_failure         — env.step() raised
          interrupted         — Ctrl-C
        RunLogger.save() rejects a transcript whose stop_reason is missing or
        unrecognised, so a new exit path cannot be added without one.

        Returns a dict: observation, progress, score, game_over, last_step,
        history, fork (None unless capture_fork captured one), stop_reason,
        context_evictions, overflow_recoveries, total_tokens.
        """
        memory              = mech.get('memory')
        state_ext           = mech.get('state_ext')
        drift_detector       = mech.get('drift_detector')
        coherence_judge      = mech.get('coherence_judge')
        budget_detector      = mech.get('budget_detector')
        integration_detector = mech.get('integration_detector')
        integration_judge    = mech.get('integration_judge')
        adaptive             = mech.get('adaptive')
        planning             = mech.get('planning')

        game_over = False
        last_checkpoint_id = None
        fork = None
        step = start_step - 1

        stop_reason = None
        context_evictions = 0
        overflow_recoveries = 0
        run_tokens = 0
        consecutive_invalid = 0
        last_prompt_tokens = 0   # server-reported prompt size; the truncation trigger

        current_goal = memory.goal if memory else None
        current_subgoal = memory.subgoal if memory else None
        last_confidence = None
        state = state_ext.probe(env) if state_ext else None

        try:
            for step in range(start_step, self.max_steps):
                is_ale, ale_details = detector.check_ale(env_checker)
                used_intervention = False
                mechanisms_used = []
                judge_tokens = 0
                action = ''
                reasoning = ''
                tokens = 0
                belief = None
                confidence = last_confidence

                if is_ale:
                    logger.add_ale({'step': step, **ale_details})
                    if capture_fork and fork is None:
                        fork = {
                            'step': step,
                            'checkpoint_id': last_checkpoint_id,
                            'progress': progress,
                            'score': score,
                            'steps_snapshot': list(logger._steps),
                            'milestones_snapshot': list(logger._milestones),
                        }

                # --- Cheap channel signals used by M4's dispatch (§integration/budget) ---
                integration_tripped, unused_items = (False, [])
                if integration_detector is not None:
                    integration_tripped, unused_items = integration_detector.precheck(
                        [],
                        memory.inventory_items if memory else [],
                        memory.recent_actions if memory else [],
                    )

                budget_risk = 0.0
                budget_preview = None
                if budget_detector is not None:
                    _, budget_preview = budget_detector.check_budget()
                    budget_risk = budget_preview.get('budget_risk', 0.0)

                drift_score = drift_detector.drift_score() if drift_detector else 0.0

                risk_vector = {
                    'drift_score': drift_score,
                    'ale_confirmed': is_ale,
                    'budget_risk': budget_risk,
                    'integration_precheck': integration_tripped,
                    'confidence': confidence,
                }

                if adaptive is not None:
                    bundle = adaptive.decide(risk_vector)
                else:
                    bundle = {
                        'trigger_info_seek': is_ale,
                        'run_integration_judge': False,
                        'trigger_replan_from_drift': False,
                        'budget_alert': False,
                    }

                if bundle['budget_alert'] and budget_preview is not None:
                    logger.add_budget_event({'step': step, **budget_preview})

                # --- Rolling window: roll before ANY dispatch path ---
                # Shrink history if the LAST call's server-reported prompt size
                # exceeded the budget. Done here rather than inside the normal
                # generation branch because M1's intervention prompt is built
                # from the same history and would otherwise grow unbounded.
                # The M2 scratchpad is untouched: only `history` pairs are
                # dropped, and they are dropped, never summarized.
                context_evictions += truncate_history_if_needed(
                    history, last_prompt_tokens, self.context_budget,
                    log=lambda msg: logger.add_truncation_event({'step': step, 'event': msg}),
                )

                # --- Dispatch: M1 (lead) ---
                if bundle['trigger_info_seek'] and info_seeking is not None:
                    try:
                        proposed_action, parsed = info_seeking.intervene(
                            format_history(history), ale_details['belief'], ale_details['evidence'],
                        )
                    except Exception as exc:
                        logger.add_error({'step': step, 'error': f'info_seeking.intervene failed: {exc}'})
                        proposed_action = ''
                    if proposed_action:
                        action = proposed_action
                        reasoning = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                        tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                        used_intervention = True
                        mechanisms_used.append('m1_info_seeking')

                # --- Dispatch: M6 (integration-gap primary, drift secondary) ---
                replanned = False
                if not used_intervention and bundle['run_integration_judge'] and integration_judge is not None:
                    result, itoks = integration_judge.check(
                        current_subgoal, unused_items,
                        memory.recent_actions if memory else [], self.llm,
                    )
                    judge_tokens += itoks
                    if result['gap']:
                        logger.add_integration_event({'step': step, 'unused': unused_items, **result})
                        mechanisms_used.append('m6_integration_gap')
                        if planning is not None:
                            new_subgoal, rtoks = planning.replan(
                                current_goal, memory, state, self.llm,
                                suggested_combination=result['suggested_combination'],
                            )
                            judge_tokens += rtoks
                            if new_subgoal:
                                current_subgoal = new_subgoal
                                replanned = True
                                mechanisms_used.append('m6_planning')

                if not used_intervention and not replanned and bundle['trigger_replan_from_drift'] and planning is not None:
                    new_subgoal, rtoks = planning.replan(current_goal, memory, state, self.llm)
                    judge_tokens += rtoks
                    if new_subgoal:
                        current_subgoal = new_subgoal
                        replanned = True
                        mechanisms_used.append('m6_planning_drift')

                # --- Dispatch: normal generation ---
                if not used_intervention:
                    scratchpad = memory.render() if memory else None
                    resp, overflow, exc = self._generate_with_overflow_recovery(
                        history, observation, scratchpad, logger, step,
                    )
                    context_evictions += overflow['evicted']
                    overflow_recoveries += overflow['rounds']
                    if exc is not None:
                        logger.add_error({'step': step, 'error': f'llm.generate failed: {exc}'})
                        stop_reason = overflow['stop_reason']
                        break
                    parsed = self.llm.parse_completion(resp)
                    last_prompt_tokens = (
                        parsed.get('prompt_tokens', 0) if isinstance(parsed, dict) else 0
                    )
                    content = parsed.get('content', '') if isinstance(parsed, dict) else str(parsed)
                    tokens = parsed.get('tokens', 0) if isinstance(parsed, dict) else 0
                    fields = extract_step_fields(content)
                    action = fields['action']
                    reasoning = fields['reasoning']
                    if fields['goal']:
                        current_goal = fields['goal']
                    if fields['subgoal']:
                        current_subgoal = fields['subgoal']
                    belief = fields['belief']
                    confidence = fields['confidence']
                    if memory:
                        memory.set_goal(current_goal or memory.goal)

                # --- M5: action templating guardrail ---
                if self.m5_enabled:
                    action = normalize_action(action)

                if not (action or '').strip():
                    # The model produced nothing usable this step. Charge it
                    # against the step budget (as dig-bench charges a nudge turn)
                    # rather than sending an empty command to the parser, and
                    # stop if the model is wedged. Without this, a wedged run was
                    # indistinguishable from one that played out its budget.
                    consecutive_invalid += 1
                    logger.add_error({'step': step,
                                      'error': 'no usable action in model response',
                                      'consecutive_invalid': consecutive_invalid})
                    if self.max_invalid_actions and consecutive_invalid >= self.max_invalid_actions:
                        stop_reason = 'invalid_action_cap'
                        break
                    continue
                consecutive_invalid = 0

                try:
                    new_observation, reward, game_over, info = env.step(action)
                except Exception as exc:
                    logger.add_error({'step': step, 'error': str(exc)})
                    stop_reason = 'env_failure'
                    break

                new_progress = info.get('game_progress', progress) if isinstance(info, dict) else progress
                score = info.get('score', 0) if isinstance(info, dict) else 0
                progress_increased = new_progress > progress

                # --- M3: ground-truth post-step probe ---
                new_state = state_ext.probe(env) if state_ext else None
                action_failed = bool(
                    state_ext and state and new_state
                    and state.get('location') == new_state.get('location')
                    and state.get('inventory_items') == new_state.get('inventory_items')
                )

                # --- Drift score (channel; incoherence scored by LLMCoherenceJudge) ---
                # Computed before logging so the step's 'tokens' total honestly
                # includes this judge call's cost, not just next step's.
                incoherence_score = 0.0
                if drift_detector:
                    recent_actions = [h['action'] for h in history[-5:]]
                    if coherence_judge is not None:
                        incoherence_score, ctoks = coherence_judge.score(
                            current_goal, current_subgoal, recent_actions, reasoning, self.llm,
                        )
                        judge_tokens += ctoks
                    else:
                        incoherence_score = 0.0

                total_tokens = tokens + judge_tokens
                run_tokens += total_tokens

                logger.add_step({
                    'step': step,
                    'obs': observation,
                    'action': action,
                    'reasoning': reasoning,
                    'progress': new_progress,
                    'score': score,
                    'reward': reward,
                    'tokens': total_tokens,
                    'judge_tokens': judge_tokens,
                    'goal': current_goal,
                    'subgoal': current_subgoal,
                    'belief': belief,
                    'confidence': confidence,
                    'mechanisms_used': mechanisms_used,
                })
                history.append({'obs': observation, 'action': action, 'reasoning': reasoning, 'progress': new_progress})

                if new_progress > progress:
                    logger.add_milestone({
                        'step': step,
                        'progress_before': progress,
                        'progress_after': new_progress,
                        'score': score,
                        'action': action,
                    })
                    print(f'[{run_id[:8]}] step {step:>4}  {progress:.1f}% → {new_progress:.1f}%  score {score}  action: {action}')

                # --- M2: update memory scratchpad from M3 ground truth ---
                if memory:
                    memory.update(
                        new_state.get('location') if new_state else None,
                        new_state.get('inventory_items') if new_state else None,
                        action, action_failed,
                        goal=current_goal, subgoal=current_subgoal,
                    )

                # --- Drift detection (channel) ---
                if drift_detector:
                    drift_detector.add(action, new_progress, incoherence_score)
                    is_drifting, drift_details = drift_detector.check_drift()
                    if is_drifting:
                        logger.add_drift_event({'step': step, **drift_details})
                        print(f'[{run_id[:8]}] step {step:>4}  DRIFT  score={drift_details["drift_score"]:.2f}'
                              f'  (stagnation={drift_details["stagnation"]:.2f}'
                              f'  loop={drift_details["loop_rate"]:.2f}'
                              f'  incoherence={drift_details["incoherence"]:.2f})')

                # --- Budget/liveness bookkeeping (channel) ---
                if budget_detector:
                    budget_detector.add(step, total_tokens, progress_increased)

                try:
                    last_checkpoint_id = env.save_checkpoint(new_observation, info)
                except Exception:
                    pass

                detector.add(action, new_progress, belief)

                if new_state is not None:
                    state = new_state
                last_confidence = confidence

                if game_over:
                    # The env reports only that the episode ended, not whether it
                    # was won — death and quit end it too — so 'solved' is claimed
                    # only at full progress and anything else is recorded honestly
                    # as a terminal 'game_over'.
                    stop_reason = 'solved' if new_progress >= 100 else 'game_over'
                    break

                if self.max_total_tokens and run_tokens >= self.max_total_tokens:
                    stop_reason = 'cost_cap'
                    break

                observation = new_observation
                progress = new_progress
        except KeyboardInterrupt:
            # Ctrl-C is a stop reason, not a crash: record it and still write the
            # transcript for the steps that did complete, so an interrupted run is
            # never mistaken for one that exhausted its step budget.
            stop_reason = 'interrupted'
            print(f'[{run_id[:8]}] interrupted by user (Ctrl-C) at step {step}')

        if stop_reason is None:
            # The only way out of the loop that sets nothing: it ran to
            # self.max_steps. Everything else breaks with a reason.
            stop_reason = 'step_cap'

        return {
            'observation': observation,
            'progress': progress,
            'score': score,
            'game_over': game_over,
            'last_step': step,
            'history': history,
            'fork': fork,
            'stop_reason': stop_reason,
            'context_evictions': context_evictions,
            'overflow_recoveries': overflow_recoveries,
            'total_tokens': run_tokens,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self, seed=0, label='baseline'):
        """Run one episode with the harness configured in config.yaml.

        Args:
            label: recorded in the transcript summary and used by plot.py to
                distinguish lines when comparing multiple runs.

        Returns
        -------
        str
            The run_id (hex string). Transcript written to out_dir/<run_id>.json.
        """
        run_id = uuid.uuid4().hex
        env = self._new_env(seed)
        observation, info = env.reset()
        detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        env_checker = make_env_checker(env)

        mech = self._build_mechanisms()
        if mech['memory']:
            mech['memory'].update('', [], '', False)

        info_seeking = InfoSeekingMechanism(self.llm, window=self.ale_window) if self.m1_enabled else None

        result = self._run_loop(
            env, self.logger, run_id, [], observation, info, 0, 0, 0,
            detector, env_checker, mech, info_seeking=info_seeking, capture_fork=False,
        )

        path = self.logger.save(run_id, self.game_name, {
            'total_steps': result['last_step'] + 1,
            'final_progress': result['progress'],
            'final_score': result['score'],
            'game_over': result['game_over'],
            'label': label,
            'fork_step': None,
            **self._run_summary(result),
        })
        print(f'[{run_id[:8]}] done — {result["last_step"] + 1} steps, progress '
              f'{result["progress"]:.1f}%, stop_reason {result["stop_reason"]}  →  {path}')

        return run_id

    def _run_summary(self, result):
        """The provenance/stop fields every transcript carries, from a _run_loop result."""
        return {
            'stop_reason': result['stop_reason'],
            'model': self.model,
            'model_source': self.model_source,
            'context_evictions': result['context_evictions'],
            'truncated': result['context_evictions'] > 0,
            'overflow_recoveries': result['overflow_recoveries'],
            'context_budget': self.context_budget,
            'total_tokens': result['total_tokens'],
        }

    def run_comparison(self, seed=0):
        """Run a baseline episode, and if an ALE is detected, fork a harness episode.

        The baseline (control) plays the whole episode with M1 (info-seeking)
        disabled. If an Accepted-Local-Error is confirmed at some step, the env
        is rewound (via restore_checkpoint) to the decision point right before
        that step, and a second episode (harness/treatment) is played from
        there with InfoSeekingMechanism intervening whenever an ALE is
        confirmed, instead of letting the model act on the flagged belief
        unchecked. All other configured mechanisms (M2–M6) run identically on
        both branches, so the only variable being compared is M1 itself.

        Returns
        -------
        (baseline_run_id, harness_run_id_or_None, fork_step_or_None)
        """
        baseline_id = uuid.uuid4().hex
        env = self._new_env(seed)
        observation, info = env.reset()
        detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        env_checker = make_env_checker(env)

        baseline_mech = self._build_mechanisms()
        if baseline_mech['memory']:
            baseline_mech['memory'].update('', [], '', False)

        result = self._run_loop(
            env, self.logger, baseline_id, [], observation, info, 0, 0, 0,
            detector, env_checker, baseline_mech, info_seeking=None, capture_fork=True,
        )

        baseline_path = self.logger.save(baseline_id, self.game_name, {
            'total_steps': result['last_step'] + 1,
            'final_progress': result['progress'],
            'final_score': result['score'],
            'game_over': result['game_over'],
            'label': 'baseline',
            'fork_step': None,
            **self._run_summary(result),
        })
        print(f'[{baseline_id[:8]}] baseline done — {result["last_step"] + 1} steps, '
              f'progress {result["progress"]:.1f}%, stop_reason {result["stop_reason"]}'
              f'  →  {baseline_path}')

        fork = result['fork']
        if fork is None:
            print(f'No ALE detected during baseline run of {self.game_name}; nothing to fork.')
            return baseline_id, None, None

        obs, ckpt_info, ok = env.restore_checkpoint(fork['checkpoint_id'])
        if not ok:
            print('Checkpoint restore failed; cannot run harness branch.')
            return baseline_id, None, fork['step']

        harness_detector = AcceptedLocalErrorDetector(k=self.ale_k, window=self.ale_window)
        for a, p, b in zip(detector.actions, detector.progress, detector.beliefs):
            harness_detector.add(a, p, b)

        info_seeking = InfoSeekingMechanism(self.llm, window=self.ale_window)
        harness_mech = self._build_mechanisms()

        for step_record in fork['steps_snapshot']:
            self.logger.add_step(step_record)
        for milestone_record in fork['milestones_snapshot']:
            self.logger.add_milestone(milestone_record)

        harness_id = uuid.uuid4().hex
        harness_history = result['history'][:fork['step']]

        harness_result = self._run_loop(
            env, self.logger, harness_id, harness_history, obs, ckpt_info,
            fork['progress'], fork['score'], fork['step'],
            harness_detector, env_checker, harness_mech, info_seeking=info_seeking, capture_fork=False,
        )

        harness_path = self.logger.save(harness_id, self.game_name, {
            'total_steps': harness_result['last_step'] + 1,
            'final_progress': harness_result['progress'],
            'final_score': harness_result['score'],
            'game_over': harness_result['game_over'],
            'label': 'harness',
            'fork_step': fork['step'],
            **self._run_summary(harness_result),
        })
        print(f'[{harness_id[:8]}] harness done — {harness_result["last_step"] + 1} steps, '
              f'progress {harness_result["progress"]:.1f}%, '
              f'stop_reason {harness_result["stop_reason"]}  →  {harness_path}')

        return baseline_id, harness_id, fork['step']


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _load_config():
    cfg_path = _HERE / 'config.yaml'
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _build_runner(cfg, game_name=None, mechanism_overrides=None):
    """Construct a Runner from a config dict.

    Args:
        mechanism_overrides: optional dict merged into cfg['experiments']['mechanisms'],
            letting callers (e.g. run_mechanism_comparison.py) toggle individual
            mechanisms without editing config.yaml.
    """
    textquests_root = resolve_root(cfg)
    or_cfg = cfg.get('openrouter', {}) or {}
    api_key = (
        or_cfg.get('api_key')
        or or_cfg.get('OPENROUTER_API_KEY')
        or os.getenv('OPENROUTER_API_KEY')
    )
    # Resolved once, here, at startup — before an env is opened or a token is
    # spent — and raises ModelConfigError if the sources disagree.
    model, model_source = resolve_model(cfg)
    print(f'[config] model {model}  (from {model_source})')
    llm = OpenRouterClient(
        api_key=api_key,
        model=model,
        base_url=or_cfg.get('base_url'),
        request_timeout=or_cfg.get('request_timeout', 15),
    )
    exp_cfg = dict(cfg.get('experiments', {}) or {})
    if mechanism_overrides:
        exp_cfg['mechanisms'] = {**(exp_cfg.get('mechanisms', {}) or {}), **mechanism_overrides}
    out_dir = _HERE / 'logs'
    chosen_game = game_name or (exp_cfg.get('games') or ['zork1'])[0]
    return Runner(
        game_name=chosen_game,
        out_dir=out_dir,
        llm_client=llm,
        config=exp_cfg,
        textquests_root=textquests_root,
        model_source=model_source,
    )


if __name__ == '__main__':
    game_arg = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = _load_config()
    runner = _build_runner(cfg, game_name=game_arg)
    runner.run_once()
