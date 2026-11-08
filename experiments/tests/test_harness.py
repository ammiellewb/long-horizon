import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channels.ale import AcceptedLocalErrorDetector
from channels.budget import BudgetLivenessDetector
from channels.drift import DriftDetector, LLMCoherenceJudge
from channels.integration import IntegrationDetector, LLMIntegrationJudge
from llm import OpenRouterClient, is_context_overflow, parse_overflow_tokens
from logger import STOP_REASONS, RunLogger
from mechanisms.action_templating import normalize_action
from mechanisms.adaptive import AdaptiveComputeController
from mechanisms.planning import PlanningMechanism
from runner import (
    MIN_KEEP_PAIRS,
    ModelConfigError,
    Runner,
    est_pair_tokens,
    evict_for_overflow,
    resolve_context_budget,
    resolve_model,
    truncate_history_if_needed,
)


class FakeLLMClient:
    """Minimal test double for OpenRouterClient: generate() records the prompt
    and returns a canned response; parse_completion() unwraps it exactly like
    the real client's {content, tokens} shape."""

    def __init__(self, content, tokens=5):
        self.content = content
        self.tokens = tokens
        self.calls = []

    def generate(self, prompt, max_tokens=None, reasoning_enabled=False, **kwargs):
        self.calls.append(prompt)
        return {'content': self.content, 'tokens': self.tokens}

    def parse_completion(self, resp):
        return resp


class RaisingLLMClient:
    def generate(self, *args, **kwargs):
        raise RuntimeError('boom')

    def parse_completion(self, resp):
        return resp


# ---------------------------------------------------------------------------
# ale.py — repetition detector (unchanged pure logic)
# ---------------------------------------------------------------------------

def test_detector_triggers():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 0)
    assert d.check_repetition()


def test_detector_no_trigger_on_progress():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('open door', 0)
    d.add('open door', 0)
    d.add('open door', 1)
    assert not d.check_repetition()


# ---------------------------------------------------------------------------
# ale.py — check_ale() with a structured belief + mock env_checker
# ---------------------------------------------------------------------------

def test_check_ale_confirms_contradiction_from_structured_belief():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    belief = {'type': 'have_item', 'item': 'sword', 'location': None}
    d.add('attack troll with sword', 0, belief)
    d.add('attack troll with sword', 0, belief)
    d.add('attack troll with sword', 0, belief)

    def env_checker(belief_tuple):
        assert belief_tuple == ('have_item', 'sword')
        return True, {'reason': 'inventory_probe', 'probe_text': 'You are carrying nothing.'}

    is_ale, details = d.check_ale(env_checker)
    assert is_ale
    assert details['belief'] == ('have_item', 'sword')
    assert details['contradiction'] is True


def test_check_ale_no_repetition_short_circuits():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('go north', 0, {'type': 'have_item', 'item': 'sword'})

    def env_checker(belief_tuple):
        raise AssertionError('env_checker should not be called without repetition')

    is_ale, details = d.check_ale(env_checker)
    assert not is_ale
    assert details['reason'] == 'no_repetition'


def test_check_ale_no_belief_detected():
    d = AcceptedLocalErrorDetector(k=3, window=5)
    d.add('wait', 0, None)
    d.add('wait', 0, None)
    d.add('wait', 0, None)

    def env_checker(belief_tuple):
        raise AssertionError('env_checker should not be called with no belief')

    is_ale, details = d.check_ale(env_checker)
    assert not is_ale
    assert details['reason'] == 'no_belief_detected'


# ---------------------------------------------------------------------------
# mechanisms/action_templating.py
# ---------------------------------------------------------------------------

def test_normalize_action_strips_label_prefixes():
    assert normalize_action('ACT: SEARCH ROAD') == 'SEARCH ROAD'
    assert normalize_action('ACTION: go north') == 'go north'
    assert normalize_action('COMMAND: open door') == 'open door'
    assert normalize_action('> go north') == 'go north'


def test_normalize_action_passthrough_when_no_prefix():
    assert normalize_action('open door') == 'open door'


def test_normalize_action_handles_empty():
    assert normalize_action('') == ''
    assert normalize_action(None) is None


# ---------------------------------------------------------------------------
# channels/budget.py — pure arithmetic, no LLM
# ---------------------------------------------------------------------------

def test_budget_liveness_below_threshold():
    b = BudgetLivenessDetector(max_steps=100, token_budget=None)
    b.add(0, 100, progress_increased=True)
    is_at_risk, details = b.check_budget()
    assert not is_at_risk
    assert details['reason'] == 'below_threshold'


def test_budget_liveness_flags_on_stall():
    b = BudgetLivenessDetector(max_steps=1000, token_budget=None,
                                stall_token_threshold=1000, threshold=0.3, w_stall=1.0,
                                w_step=0.0, w_token=0.0)
    for step in range(5):
        b.add(step, 300, progress_increased=False)
    is_at_risk, details = b.check_budget()
    assert is_at_risk
    assert details['tokens_since_progress'] == 1500
    assert details['stall_ratio'] == 1.0


def test_budget_liveness_resets_stall_on_progress():
    b = BudgetLivenessDetector(max_steps=1000, stall_token_threshold=1000)
    b.add(0, 900, progress_increased=False)
    b.add(1, 100, progress_increased=True)
    _, details = b.check_budget()
    assert details['tokens_since_progress'] == 0
    assert details['cumulative_tokens'] == 1000


# ---------------------------------------------------------------------------
# channels/integration.py — precheck (no LLM) + judge (fake client)
# ---------------------------------------------------------------------------

def test_integration_precheck_trips_on_unused_items():
    d = IntegrationDetector(min_unused=2, recent_window=10)
    tripped, unused = d.precheck(
        discoveries=['lamp', 'key'],
        inventory_items=['spell'],
        recent_actions=['go north', 'take lamp'],
    )
    assert tripped
    assert set(unused) == {'key', 'spell'}


def test_integration_precheck_does_not_trip_when_referenced():
    d = IntegrationDetector(min_unused=2, recent_window=10)
    tripped, unused = d.precheck(
        discoveries=['lamp', 'key'],
        inventory_items=[],
        recent_actions=['use lamp on key', 'go north'],
    )
    assert not tripped
    assert unused == []


def test_integration_judge_parses_gap_response():
    client = FakeLLMClient('{"gap": true, "suggested_combination": "unlock grate with key"}')
    judge = LLMIntegrationJudge()
    result, tokens = judge.check('escape the cellar', ['key', 'lamp'], ['go north'], client)
    assert result == {'gap': True, 'suggested_combination': 'unlock grate with key'}
    assert tokens == 5


def test_integration_judge_falls_back_on_llm_error():
    judge = LLMIntegrationJudge()
    result, tokens = judge.check('goal', ['key'], [], RaisingLLMClient())
    assert result == {'gap': False, 'suggested_combination': None}
    assert tokens == 0


# ---------------------------------------------------------------------------
# channels/drift.py — LLMCoherenceJudge + pure DriftDetector math
# ---------------------------------------------------------------------------

def test_coherence_judge_parses_score():
    client = FakeLLMClient('{"incoherence": 0.8}')
    judge = LLMCoherenceJudge()
    score, tokens = judge.score('escape', 'find key', ['go north'], 'I am looking around', client)
    assert score == 0.8
    assert tokens == 5


def test_coherence_judge_empty_reasoning_short_circuits():
    client = FakeLLMClient('{"incoherence": 0.8}')
    judge = LLMCoherenceJudge()
    score, tokens = judge.score('escape', 'find key', [], '', client)
    assert score == 0.5
    assert tokens == 0
    assert client.calls == []


def test_drift_detector_flags_high_composite_score():
    d = DriftDetector(window=3, threshold=0.5, w_stagnation=0.5, w_loop=0.5, w_coherence=0.0)
    d.add('look', 0, 0.0)
    d.add('look', 0, 0.0)
    d.add('look', 0, 0.0)
    is_drifting, details = d.check_drift()
    assert is_drifting
    # 3 identical actions -> 1 unique / 3 total -> loop_rate = 1 - 1/3
    assert details['loop_rate'] == round(1 - 1 / 3, 3)


def test_drift_detector_insufficient_history():
    d = DriftDetector(window=5)
    d.add('look', 0, 0.0)
    is_drifting, details = d.check_drift()
    assert not is_drifting
    assert details['reason'] == 'insufficient_history'


# ---------------------------------------------------------------------------
# mechanisms/adaptive.py — many-to-many risk-vector dispatch
# ---------------------------------------------------------------------------

def test_adaptive_ale_takes_priority():
    c = AdaptiveComputeController()
    bundle = c.decide({'drift_score': 0.9, 'ale_confirmed': True,
                        'budget_risk': 0.95, 'integration_precheck': True, 'confidence': 0.5})
    assert bundle['trigger_info_seek']
    assert not bundle['run_integration_judge']  # ALE dispatch takes priority this step


def test_adaptive_drift_alone_triggers_replan():
    c = AdaptiveComputeController(drift_replan_threshold=0.65)
    bundle = c.decide({'drift_score': 0.7, 'ale_confirmed': False,
                        'budget_risk': 0.0, 'integration_precheck': False, 'confidence': None})
    assert not bundle['trigger_info_seek']
    assert bundle['trigger_replan_from_drift']


def test_adaptive_skips_integration_judge_under_critical_budget():
    c = AdaptiveComputeController(skip_integration_judge_above_budget_risk=0.9)
    bundle = c.decide({'drift_score': 0.0, 'ale_confirmed': False,
                        'budget_risk': 0.95, 'integration_precheck': True, 'confidence': None})
    assert not bundle['run_integration_judge']
    assert bundle['budget_alert']


def test_adaptive_runs_integration_judge_when_precheck_trips_and_budget_ok():
    c = AdaptiveComputeController()
    bundle = c.decide({'drift_score': 0.0, 'ale_confirmed': False,
                        'budget_risk': 0.1, 'integration_precheck': True, 'confidence': None})
    assert bundle['run_integration_judge']
    assert not bundle['budget_alert']


# ---------------------------------------------------------------------------
# mechanisms/planning.py — replan (fake client)
# ---------------------------------------------------------------------------

class FakeMemory:
    def __init__(self):
        self.goal = 'escape the cellar'
        self.subgoal = 'find a light source'
        self.recent_actions = ['go north', 'take lamp']


def test_planning_replan_updates_memory_subgoal():
    client = FakeLLMClient('{"subgoal": "unlock the grate with the key then go down"}')
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(
        memory.goal, memory, {'location': 'cellar', 'inventory_items': ['key']}, client,
        suggested_combination='use key on grate',
    )
    assert new_subgoal == 'unlock the grate with the key then go down'
    assert memory.subgoal == new_subgoal
    assert tokens == 5


def test_planning_replan_returns_none_on_unusable_response():
    client = FakeLLMClient('not json at all')
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(memory.goal, memory, None, client)
    assert new_subgoal is None
    assert memory.subgoal == 'find a light source'  # left unchanged


def test_planning_replan_handles_llm_failure():
    memory = FakeMemory()
    planner = PlanningMechanism()
    new_subgoal, tokens = planner.replan(memory.goal, memory, None, RaisingLLMClient())
    assert new_subgoal is None
    assert tokens == 0


# ---------------------------------------------------------------------------
# runner.resolve_model — single source of truth for the model id
# ---------------------------------------------------------------------------

CFG = {'openrouter': {'model': 'anthropic/claude-sonnet-5'}}


def test_resolve_model_from_config_when_env_empty():
    assert resolve_model(CFG, env={}) == ('anthropic/claude-sonnet-5',
                                          'config.yaml:openrouter.model')


def test_resolve_model_env_var_wins_when_sources_agree():
    model, source = resolve_model(CFG, env={'OPENROUTER_MODEL': 'anthropic/claude-sonnet-5'})
    assert (model, source) == ('anthropic/claude-sonnet-5', 'env:OPENROUTER_MODEL')


def test_resolve_model_ignores_openrouter_route_prefix_when_comparing():
    # inspect-ai spells the same model with the provider route in front.
    model, source = resolve_model(
        {'openrouter': {'model': 'meta-llama/llama-3.2-3b-instruct:free'}},
        env={'INSPECT_EVAL_MODEL': 'openrouter/meta-llama/llama-3.2-3b-instruct:free'},
    )
    assert model == 'meta-llama/llama-3.2-3b-instruct:free'
    assert source == 'config.yaml:openrouter.model'


def test_resolve_model_strips_route_from_the_recorded_id():
    model, _ = resolve_model({}, env={'OPENROUTER_MODEL': 'openrouter/vendor/model-1'})
    assert model == 'vendor/model-1'  # the route is not part of the provider's id


def test_resolve_model_fails_loudly_when_env_disagrees_with_config():
    with pytest.raises(ModelConfigError) as exc:
        resolve_model(CFG, env={'OPENROUTER_MODEL': 'google/gemma-4-26b-a4b-it-20260403'})
    message = str(exc.value)
    assert 'anthropic/claude-sonnet-5' in message
    assert 'google/gemma-4-26b-a4b-it-20260403' in message


def test_resolve_model_fails_loudly_when_inspect_env_disagrees_with_config():
    # The real three-way mismatch: config.yaml, .env and the run logs all named
    # different models. INSPECT_EVAL_MODEL is not a source, but it must not be
    # allowed to disagree silently either.
    with pytest.raises(ModelConfigError):
        resolve_model(CFG, env={
            'INSPECT_EVAL_MODEL': 'openrouter/meta-llama/llama-3.2-3b-instruct:free',
        })


def test_resolve_model_fails_loudly_when_nothing_is_configured():
    with pytest.raises(ModelConfigError):
        resolve_model({}, env={})


def test_resolve_model_rejects_inspect_env_as_the_only_source():
    with pytest.raises(ModelConfigError) as exc:
        resolve_model({}, env={'INSPECT_EVAL_MODEL': 'openrouter/vendor/model-1'})
    assert 'INSPECT_EVAL_MODEL' in str(exc.value)


# ---------------------------------------------------------------------------
# llm.py — context-overflow classification (ported from dig-bench)
# ---------------------------------------------------------------------------

OVERFLOW_MESSAGE = (
    'HTTP 400: This model\'s maximum context length is 32768 tokens, however you '
    'requested 41000 tokens.'
)


def test_is_context_overflow_matches_provider_wording():
    assert is_context_overflow(RuntimeError(OVERFLOW_MESSAGE))
    assert is_context_overflow({'message': 'context_length_exceeded'})
    assert not is_context_overflow(RuntimeError('HTTP 401: invalid api key'))
    assert not is_context_overflow(None)


def test_parse_overflow_tokens_reads_the_reported_counts():
    assert parse_overflow_tokens(RuntimeError(OVERFLOW_MESSAGE)) == (41000, 32768)
    assert parse_overflow_tokens(RuntimeError('context window too small')) is None
    # incoherent pair (limit >= total) is refused rather than yielding a negative deficit
    assert parse_overflow_tokens(
        RuntimeError('maximum context length of 9000 tokens, requested 100 tokens')
    ) is None


# ---------------------------------------------------------------------------
# runner.py — rolling-window truncation (NOT summarization)
# ---------------------------------------------------------------------------

def _history(n, obs_chars=400):
    return [{'obs': f'{i:03d}' + 'x' * obs_chars, 'action': 'go north',
             'reasoning': 'because', 'progress': 0} for i in range(n)]


def test_est_pair_tokens_is_a_chars_over_four_proxy():
    pair = {'obs': 'a' * 40, 'action': 'b' * 4, 'reasoning': 'c' * 4}
    assert est_pair_tokens(pair) == 12
    assert est_pair_tokens('not a dict') == 0


def test_resolve_context_budget():
    assert resolve_context_budget(None, 288, None) is None      # nothing configured -> off
    assert resolve_context_budget(32768, 288, None) == 32480     # window minus output reserve
    assert resolve_context_budget(32768, 288, 1000) == 1000      # absolute override wins


def test_truncate_is_a_noop_without_a_budget_or_a_server_count():
    history = _history(6)
    assert truncate_history_if_needed(history, 99999, None) == 0
    assert truncate_history_if_needed(history, 0, 100) == 0      # first-step bootstrap
    assert truncate_history_if_needed(history, 50, 100) == 0     # under budget
    assert len(history) == 6


def test_truncate_evicts_the_fewest_oldest_pairs_needed():
    history = _history(6)                      # each pair ~103 est tokens
    evicted = truncate_history_if_needed(history, 400, 250)
    assert evicted == 2                        # 400 - 2*103 = 194 <= 250; a third is not needed
    assert len(history) == 4


def test_truncate_pins_the_seed_pair_and_keeps_the_newest():
    history = _history(4)
    first, newest = history[0], history[-1]
    truncate_history_if_needed(history, 10_000, 1)   # demand far more than can be given
    assert len(history) == MIN_KEEP_PAIRS
    assert history[0] is first                       # seed turn is pinned
    assert history[-1] is newest                     # in-flight-adjacent pair is never evicted


def test_truncate_logs_what_it_dropped():
    history = _history(6)
    messages = []
    truncate_history_if_needed(history, 400, 250, log=messages.append)
    assert len(messages) == 1
    assert 'evicted 2 oldest pair(s)' in messages[0]


def test_truncate_drops_pairs_without_summarizing_them():
    # The M2 ablation depends on this: eviction must remove information, never
    # condense it into something the model still gets to see.
    history = _history(5)
    evicted_pair = history[1]
    truncate_history_if_needed(history, 10_000, 1)
    assert evicted_pair not in history
    assert all(isinstance(pair, dict) and set(pair) == {'obs', 'action', 'reasoning', 'progress'}
               for pair in history)
    assert not any(evicted_pair['obs'] in str(pair) for pair in history)


def test_evict_for_overflow_sizes_eviction_to_the_reported_deficit():
    history = _history(12)                     # ~103 est tokens per pair
    # deficit 41000-32768 = 8232 -> * 1.25 + 1000 = 11290 est tokens wanted, but
    # the MIN_KEEP floor stops it: everything evictable goes.
    evicted = evict_for_overflow(history, RuntimeError(OVERFLOW_MESSAGE))
    assert evicted == 10
    assert len(history) == MIN_KEEP_PAIRS


def test_evict_for_overflow_stops_at_the_deficit():
    history = _history(12, obs_chars=4000)     # ~1003 est tokens per pair
    evicted = evict_for_overflow(history, RuntimeError(
        'maximum context length of 1000 tokens, requested a total of 3000 tokens'))
    assert evicted == 4                        # (2000 * 1.25) + 1000 = 3500 / ~1003
    assert len(history) == 8


def test_evict_for_overflow_falls_back_to_a_doubling_batch():
    # No counts in the wording -> 2**round_idx pairs per round.
    history = _history(12)
    assert evict_for_overflow(history, RuntimeError('context window exceeded'), round_idx=0) == 1
    assert evict_for_overflow(history, RuntimeError('context window exceeded'), round_idx=2) == 4


def test_evict_for_overflow_returns_zero_when_it_cannot_shrink():
    history = _history(MIN_KEEP_PAIRS)
    assert evict_for_overflow(history, RuntimeError(OVERFLOW_MESSAGE)) == 0
    assert len(history) == MIN_KEEP_PAIRS


# ---------------------------------------------------------------------------
# runner.py — stop-reason taxonomy, driven end-to-end through run_once()
# ---------------------------------------------------------------------------

class ScriptedLLMClient:
    """LLM double whose nth generate() either returns canned content or raises.

    errors maps a 1-based call index to the exception to raise on that call.
    """

    model = 'test/model-1'

    def __init__(self, content='{"action": "go north", "reasoning": "r"}',
                 errors=None, tokens=5, prompt_tokens=0):
        self.content = content
        self.errors = dict(errors or {})
        self.tokens = tokens
        self.prompt_tokens = prompt_tokens
        self.calls = []

    def generate(self, prompt, max_tokens=None, reasoning_enabled=False, **kwargs):
        self.calls.append(prompt)
        error = self.errors.get(len(self.calls))
        if error is not None:
            raise error
        return {'content': self.content, 'tokens': self.tokens,
                'prompt_tokens': self.prompt_tokens}

    def parse_completion(self, resp):
        return resp


class FakeEnv:
    """Minimal TextQuestsEnv stand-in: deterministic observations, no game."""

    def __init__(self, game_over_at=None, final_progress=0, step_raises_at=None):
        self.game_over_at = game_over_at
        self.final_progress = final_progress
        self.step_raises_at = step_raises_at
        self.actions = []

    def reset(self):
        return 'you are in a room', {'game_progress': 0, 'score': 0}

    def step(self, action):
        self.actions.append(action)
        n = len(self.actions)
        if self.step_raises_at == n:
            raise RuntimeError('frotz interpreter died')
        done = self.game_over_at == n
        progress = self.final_progress if done else 0
        return f'obs-{n}', 0, done, {'game_progress': progress, 'score': 0}

    def save_checkpoint(self, observation, info):
        return f'ckpt-{len(self.actions)}'


def _run(tmp_path, llm_client, env, **config_overrides):
    """run_once() against a FakeEnv; returns the parsed transcript."""
    config = {
        'max_steps': 3,
        'accepted_local_error_k': 9999,   # keep the ALE channel out of these tests
        'info_seek_window': 10,
        'drift': {'enabled': False},
        'budget': {'enabled': False},
        'integration': {'enabled': False},
        'mechanisms': {name: False for name in (
            'm1_info_seeking', 'm2_memory', 'm3_state_ext',
            'm4_adaptive', 'm5_action_templating', 'm6_planning')},
    }
    mechanisms = {**config['mechanisms'], **config_overrides.pop('mechanisms', {})}
    config.update(config_overrides)
    config['mechanisms'] = mechanisms
    runner = Runner(game_name='zork1', out_dir=tmp_path, llm_client=llm_client,
                    config=config, model_source='test')
    runner._new_env = lambda seed: env
    run_id = runner.run_once()
    return json.loads((tmp_path / f'{run_id}.json').read_text())


def test_every_stop_reason_in_the_taxonomy_is_distinct():
    assert len(set(STOP_REASONS)) == len(STOP_REASONS)


def test_stop_reason_step_cap(tmp_path):
    summary = _run(tmp_path, ScriptedLLMClient(), FakeEnv())['summary']
    assert summary['stop_reason'] == 'step_cap'
    assert summary['total_steps'] == 3
    assert summary['model'] == 'test/model-1'
    assert summary['model_source'] == 'test'
    assert summary['truncated'] is False
    assert summary['context_evictions'] == 0


def test_stop_reason_solved_only_at_full_progress(tmp_path):
    summary = _run(tmp_path, ScriptedLLMClient(),
                   FakeEnv(game_over_at=2, final_progress=100))['summary']
    assert summary['stop_reason'] == 'solved'
    assert summary['total_steps'] == 2


def test_stop_reason_game_over_when_the_episode_ends_unsolved(tmp_path):
    # Death/quit also ends the episode; it must not be recorded as 'solved'.
    summary = _run(tmp_path, ScriptedLLMClient(),
                   FakeEnv(game_over_at=2, final_progress=40))['summary']
    assert summary['stop_reason'] == 'game_over'


def test_stop_reason_api_failure(tmp_path):
    client = ScriptedLLMClient(errors={2: RuntimeError('HTTP 500: upstream exploded')})
    transcript = _run(tmp_path, client, FakeEnv())
    assert transcript['summary']['stop_reason'] == 'api_failure'
    assert 'llm.generate failed' in transcript['errors'][0]['error']


def test_stop_reason_api_failure_is_not_recorded_as_a_full_run(tmp_path):
    # The bug this taxonomy exists for: a run that died at step 1 used to be
    # indistinguishable from one that played out its whole budget.
    client = ScriptedLLMClient(errors={2: RuntimeError('HTTP 500: upstream exploded')})
    summary = _run(tmp_path, client, FakeEnv(), max_steps=100)['summary']
    assert summary['stop_reason'] == 'api_failure'
    assert summary['total_steps'] < 100


def test_stop_reason_context_exhausted(tmp_path):
    # Overflow on the very first call: history is empty, so eviction cannot help.
    client = ScriptedLLMClient(errors={1: RuntimeError(OVERFLOW_MESSAGE)})
    summary = _run(tmp_path, client, FakeEnv())['summary']
    assert summary['stop_reason'] == 'context_exhausted'


def test_stop_reason_env_failure(tmp_path):
    summary = _run(tmp_path, ScriptedLLMClient(), FakeEnv(step_raises_at=2))['summary']
    assert summary['stop_reason'] == 'env_failure'


def test_stop_reason_cost_cap(tmp_path):
    summary = _run(tmp_path, ScriptedLLMClient(tokens=5), FakeEnv(),
                   max_steps=100, max_total_tokens=8)['summary']
    assert summary['stop_reason'] == 'cost_cap'
    assert summary['total_steps'] == 2          # 5 tokens, then 10 >= 8
    assert summary['total_tokens'] == 10


def test_stop_reason_invalid_action_cap(tmp_path):
    # 'ACT:' normalizes to the empty string: no usable action to send.
    client = ScriptedLLMClient(content='{"action": "ACT:", "reasoning": "r"}')
    transcript = _run(tmp_path, client, FakeEnv(), max_steps=100,
                      max_invalid_actions=2,
                      mechanisms={'m5_action_templating': True})
    assert transcript['summary']['stop_reason'] == 'invalid_action_cap'
    assert transcript['errors'][-1]['consecutive_invalid'] == 2
    assert transcript['steps'] == []            # no empty command ever reached the env


def test_stop_reason_interrupted(tmp_path):
    client = ScriptedLLMClient(errors={2: KeyboardInterrupt()})
    summary = _run(tmp_path, client, FakeEnv(), max_steps=100)['summary']
    assert summary['stop_reason'] == 'interrupted'


# ---------------------------------------------------------------------------
# runner.py — truncation and overflow recovery wired into a run
# ---------------------------------------------------------------------------

def test_run_records_proactive_truncation(tmp_path):
    # Every call reports a 1000-token prompt against a 10-token budget, so the
    # window rolls as soon as there are evictable pairs.
    client = ScriptedLLMClient(prompt_tokens=1000)
    transcript = _run(tmp_path, client, FakeEnv(), max_steps=6,
                      context={'budget_tokens': 10})
    summary = transcript['summary']
    assert summary['stop_reason'] == 'step_cap'
    assert summary['context_evictions'] > 0
    assert summary['truncated'] is True
    assert summary['context_budget'] == 10
    assert any('context truncated' in e['event'] for e in transcript['truncation_events'])

    # The pinned seed turn survives in the final prompt; an evicted middle pair
    # is gone from it entirely — dropped, not summarized.
    last_prompt = '\n'.join(m['content'] for m in client.calls[-1])
    assert 'you are in a room' in last_prompt
    assert 'obs-1' not in last_prompt


def test_run_recovers_from_a_context_overflow_and_records_it(tmp_path):
    # Overflow on the 4th call, by which point there are evictable pairs.
    client = ScriptedLLMClient(errors={4: RuntimeError(OVERFLOW_MESSAGE)})
    transcript = _run(tmp_path, client, FakeEnv(), max_steps=5)
    summary = transcript['summary']
    assert summary['stop_reason'] == 'step_cap'      # recovered, run continued
    assert summary['overflow_recoveries'] == 1
    assert summary['context_evictions'] > 0
    assert summary['total_steps'] == 5
    assert any('context overflow' in e['event'] for e in transcript['truncation_events'])


def test_overflow_recovery_is_bounded(tmp_path):
    # Every call overflows: recovery evicts until it cannot, then gives up.
    client = ScriptedLLMClient(errors={n: RuntimeError(OVERFLOW_MESSAGE) for n in range(1, 50)})
    summary = _run(tmp_path, client, FakeEnv(), max_steps=20,
                   context={'max_overflow_rounds': 3})['summary']
    assert summary['stop_reason'] == 'context_exhausted'
    assert len(client.calls) <= 4                    # 1 attempt + at most 3 retries


# ---------------------------------------------------------------------------
# logger.py — the taxonomy is enforced at the write boundary
# ---------------------------------------------------------------------------

def test_logger_refuses_a_transcript_without_a_stop_reason(tmp_path):
    logger = RunLogger(tmp_path)
    with pytest.raises(ValueError, match='stop_reason'):
        logger.save('abc', 'zork1', {'total_steps': 1, 'model': 'test/model-1'})


def test_logger_refuses_an_unrecognised_stop_reason(tmp_path):
    logger = RunLogger(tmp_path)
    with pytest.raises(ValueError, match='stop_reason'):
        logger.save('abc', 'zork1', {'stop_reason': 'ran_out_of_patience',
                                      'model': 'test/model-1'})


def test_logger_refuses_a_transcript_without_a_model_id(tmp_path):
    logger = RunLogger(tmp_path)
    with pytest.raises(ValueError, match='model'):
        logger.save('abc', 'zork1', {'stop_reason': 'step_cap'})


# ---------------------------------------------------------------------------
# llm.py — fairness: no sampling parameters, and no silent model default
# ---------------------------------------------------------------------------

class _FakeResponse:
    status_code = 200

    def json(self):
        return {'choices': [{'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}}


def _client_with_captured_payload():
    client = OpenRouterClient(api_key='k', model='vendor/model-1')
    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(json)
        return _FakeResponse()

    client.session.post = post
    return client, sent


def test_no_sampling_parameters_are_sent():
    # FAIRNESS (dig-bench README): every model runs its own sampling defaults.
    # Forcing one temperature across providers is a different intervention per
    # model, so neither temperature nor top_p may appear in the payload.
    client, sent = _client_with_captured_payload()
    client.generate('hello', max_tokens=16)
    assert 'temperature' not in sent
    assert 'top_p' not in sent
    assert sent['model'] == 'vendor/model-1'


def test_temperature_is_only_sent_when_a_caller_explicitly_asks():
    client, sent = _client_with_captured_payload()
    client.generate('hello', temperature=0.2, max_tokens=16)
    assert sent['temperature'] == 0.2  # opt-in only; no caller in this repo does


def test_parse_completion_exposes_the_server_reported_prompt_size():
    client, _ = _client_with_captured_payload()
    parsed = client.parse_completion(_FakeResponse().json())
    assert parsed['prompt_tokens'] == 7   # the truncation trigger
    assert parsed['tokens'] == 10


def test_client_refuses_to_invent_a_model_id():
    # The silent 'openrouter/free' default used to be a fourth source of truth.
    with pytest.raises(RuntimeError, match='explicit model id'):
        OpenRouterClient(api_key='k', model=None)


def test_logger_writes_the_stop_reason_and_model(tmp_path):
    logger = RunLogger(tmp_path)
    path = logger.save('abc', 'zork1', {'stop_reason': 'step_cap', 'model': 'test/model-1'})
    written = json.loads(Path(path).read_text())
    assert written['summary']['stop_reason'] == 'step_cap'
    assert written['summary']['model'] == 'test/model-1'
