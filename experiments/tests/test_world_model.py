"""Tests for mechanisms/world_model.py (M7 candidate: program induction + backtest).

Everything here runs offline: the only LLM dependency is injected as a
callable, so the authoring step is exercised with a stub. No network, no API
key, no paid call.

Style matches tests/test_harness.py (module-level sys.path insert, plain
test_* functions with asserts). pytest is not installed in this project's
interpreter, so a minimal self-runner is provided at the bottom:

    .venv-inspect/bin/python experiments/tests/test_world_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mechanisms.world_model import (
    FLAG_NAMES,
    SandboxError,
    SandboxTimeout,
    Transition,
    WorldModelMechanism,
    _ALLOWED_IMPORTS,
    _SAFE_BUILTINS,
    _safe_import,
    backtest,
    compile_world_model,
    extract_code,
    llm_author,
    trajectory_from_history,
    trajectory_from_log,
)


class StubAuthor:
    """Test double for the injected authoring callable.

    Returns the next canned program on each call and records the prompts, so a
    refinement loop can be driven without any network access.
    """

    def __init__(self, *programs, tokens=7):
        self.programs = list(programs)
        self.tokens = tokens
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        code = self.programs.pop(0) if self.programs else ''
        return f'Here is my model:\n```python\n{code}\n```', self.tokens


class RaisingAuthor:
    def __call__(self, prompt):
        raise RuntimeError('boom')


class FakeLLMClient:
    """Same shape as tests/test_harness.FakeLLMClient (generate/parse_completion)."""

    def __init__(self, content, tokens=5):
        self.content = content
        self.tokens = tokens
        self.calls = []

    def generate(self, prompt, max_tokens=None, reasoning_enabled=False, **kwargs):
        self.calls.append(prompt)
        return {'content': self.content, 'tokens': self.tokens}

    def parse_completion(self, resp):
        return resp


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(f'expected {exc_type.__name__}, got {type(exc).__name__}: {exc}')
    raise AssertionError(f'expected {exc_type.__name__}, nothing raised')


# ---------------------------------------------------------------------------
# Toy world: a two-room corridor. Used by the backtest tests below.
# ---------------------------------------------------------------------------

PERFECT_MODEL = '''
ROOMS = {"West of House": "North of House", "North of House": "West of House"}

def step(obs, action):
    if action == "north" and obs in ROOMS:
        return ROOMS[obs]
    return obs
'''

# Wrong on exactly one rule: it thinks "north" out of North of House is a
# no-op. Everything else reproduces.
BROKEN_MODEL = '''
def step(obs, action):
    if action == "north" and obs == "West of House":
        return "North of House"
    return obs
'''

TOY_HISTORY = [
    {'obs': 'West of House', 'action': 'north', 'reasoning': '', 'progress': 0},
    {'obs': 'North of House', 'action': 'look', 'reasoning': '', 'progress': 0},
    {'obs': 'North of House', 'action': 'north', 'reasoning': '', 'progress': 0},
    {'obs': 'West of House', 'action': 'wait', 'reasoning': '', 'progress': 0},
    {'obs': 'West of House', 'action': 'north', 'reasoning': '', 'progress': 0},
]


def _toy_trajectory():
    return trajectory_from_history(TOY_HISTORY)


# ---------------------------------------------------------------------------
# Sandbox: disallowed imports
# ---------------------------------------------------------------------------

def test_sandbox_rejects_disallowed_imports():
    for module in ('os', 'sys', 'subprocess', 'socket', 'shutil', 'pickle', 'importlib'):
        code = f'import {module}\n\ndef step(obs, action):\n    return obs\n'
        exc = _assert_raises(SandboxError, compile_world_model, code)
        assert module in str(exc), (module, str(exc))


def test_sandbox_rejects_disallowed_from_imports():
    code = 'from os import system\n\ndef step(obs, action):\n    return obs\n'
    exc = _assert_raises(SandboxError, compile_world_model, code)
    assert 'os' in str(exc)


def test_sandbox_rejects_import_hidden_inside_a_function():
    code = (
        'def step(obs, action):\n'
        '    import socket\n'
        '    return obs\n'
    )
    _assert_raises(SandboxError, compile_world_model, code)


def test_sandbox_allows_the_allowlisted_imports():
    code = (
        'import math\n'
        'from collections import deque\n'
        'import re, itertools, functools, heapq, copy, bisect, operator, string\n'
        '\n'
        'def step(obs, action):\n'
        '    return obs + str(int(math.floor(1.5))) + str(len(deque([1, 2])))\n'
    )
    model = compile_world_model(code)
    obs, info = model.step('x', 'look')
    assert obs == 'x12'
    assert info == {}


def test_safe_import_hook_allowlist():
    assert _safe_import('math').__name__ == 'math'
    _assert_raises(ImportError, _safe_import, 'os')
    _assert_raises(ImportError, _safe_import, 'os.path')
    # Allowlist is Schema's, unwidened.
    assert _ALLOWED_IMPORTS == {
        'numpy', 'math', 'collections', 'itertools', 'functools', 'heapq', 're',
        'string', 'copy', 'dataclasses', 'typing', 'enum', 'operator', 'bisect',
    }


# ---------------------------------------------------------------------------
# Sandbox: dangerous builtins
# ---------------------------------------------------------------------------

def test_dangerous_builtins_are_absent_from_the_namespace():
    for name in ('eval', 'exec', 'compile', 'open', 'input', '__import__',
                 'globals', 'locals', 'vars', 'dir', 'breakpoint', 'help',
                 'exit', 'quit', 'memoryview', 'print'):
        assert name not in _SAFE_BUILTINS, name


def test_sandbox_rejects_dangerous_builtins():
    for snippet in (
        'eval("1+1")',
        'exec("x = 1")',
        'open("/etc/passwd").read()',
        '__import__("os").system("echo pwned")',
        'compile("1", "<s>", "eval")',
        'globals()',
        'vars()',
    ):
        code = f'def step(obs, action):\n    return str({snippet})\n'
        _assert_raises(SandboxError, compile_world_model, code)


def test_sandbox_rejects_dunder_traversal_escape():
    for snippet in (
        '().__class__.__bases__[0].__subclasses__()',
        'step.__globals__',
        'getattr(obs, "__class__")',
        'type(obs).__mro__',
    ):
        code = f'def step(obs, action):\n    return str({snippet})\n'
        _assert_raises(SandboxError, compile_world_model, code)


def test_sandbox_rejects_code_with_no_transition_function():
    code = 'def helper(x):\n    return x\n'
    exc = _assert_raises(SandboxError, compile_world_model, code)
    assert 'predict' in str(exc) and 'step' in str(exc)


def test_sandbox_reports_syntax_errors_as_sandbox_error():
    _assert_raises(SandboxError, compile_world_model, 'def step(obs action):\n  return obs\n')


# ---------------------------------------------------------------------------
# Sandbox: wall-clock timeout
# ---------------------------------------------------------------------------

def test_sandbox_enforces_timeout_on_infinite_loop_in_step():
    code = (
        'def step(obs, action):\n'
        '    while True:\n'
        '        pass\n'
    )
    model = compile_world_model(code, call_timeout_s=0.3)
    _assert_raises(SandboxTimeout, model.step, 'obs', 'look')


def test_sandbox_timeout_survives_a_model_that_swallows_exceptions():
    # The timeout signal must not be catchable by `except Exception`.
    code = (
        'def step(obs, action):\n'
        '    while True:\n'
        '        try:\n'
        '            pass\n'
        '        except Exception:\n'
        '            pass\n'
    )
    model = compile_world_model(code, call_timeout_s=0.3)
    _assert_raises(SandboxTimeout, model.step, 'obs', 'look')


def test_sandbox_enforces_timeout_on_module_level_infinite_loop():
    code = (
        'while True:\n'
        '    pass\n'
        '\n'
        'def step(obs, action):\n'
        '    return obs\n'
    )
    _assert_raises(SandboxTimeout, compile_world_model, code, load_timeout_s=0.3)


def test_backtest_records_a_timeout_as_a_mismatch_not_a_crash():
    code = (
        'def step(obs, action):\n'
        '    if action == "look":\n'
        '        while True:\n'
        '            pass\n'
        '    return obs\n'
    )
    model = compile_world_model(code, call_timeout_s=0.3)
    result = backtest(model, _toy_trajectory())
    assert len(result['mismatches']) >= 1
    timeouts = [m for m in result['mismatches'] if 'budget' in ' '.join(m['errors'])]
    assert timeouts, result['mismatches']
    assert timeouts[0]['kinds'] == ['error']


# ---------------------------------------------------------------------------
# Trajectory adapters
# ---------------------------------------------------------------------------

def test_trajectory_from_history_pairs_before_and_after_observations():
    traj = _toy_trajectory()
    # Last recorded step has no successor observation, so it is dropped.
    assert len(traj) == len(TOY_HISTORY) - 1
    assert (traj[0].obs, traj[0].action, traj[0].next_obs) == (
        'West of House', 'north', 'North of House')
    assert (traj[2].obs, traj[2].action, traj[2].next_obs) == (
        'North of House', 'north', 'West of House')


def test_trajectory_from_log_derives_progress_and_score_flags():
    log = {'steps': [
        {'step': 0, 'obs': 'a', 'action': 'x', 'progress': 0, 'score': 0},
        {'step': 1, 'obs': 'b', 'action': 'y', 'progress': 5, 'score': 10},
        {'step': 2, 'obs': 'c', 'action': 'z', 'progress': 5, 'score': 10},
    ]}
    traj = trajectory_from_log(log)
    assert [t.next_obs for t in traj] == ['b', 'c']
    assert traj[1].flags == {'progress_increased': True, 'score_increased': True}
    assert traj[0].flags == {'progress_increased': False, 'score_increased': False}
    assert traj[1].meta['step'] == 1


# ---------------------------------------------------------------------------
# backtest: falsification against recorded history
# ---------------------------------------------------------------------------

def test_backtest_scores_a_perfect_model_as_fully_accurate():
    model = compile_world_model(PERFECT_MODEL)
    result = backtest(model, _toy_trajectory())
    assert result['total'] == 4
    assert result['exact'] == 4
    assert result['accuracy'] == 1.0
    assert result['green'] is True
    assert result['mismatches'] == []
    assert result['first_mismatch'] is None
    assert all(s['ok'] and s['match'] == 'exact' for s in result['steps'])
    assert 'GREEN' in result['report']


def test_backtest_identifies_mismatches_on_a_toy_trajectory():
    model = compile_world_model(BROKEN_MODEL)
    result = backtest(model, _toy_trajectory())
    assert result['total'] == 4
    assert result['exact'] == 3
    assert result['accuracy'] == 0.75
    assert result['green'] is False

    assert len(result['mismatches']) == 1
    bad = result['mismatches'][0]
    # The only falsified transition is 'north' out of North of House (index 2).
    assert bad['index'] == 2
    assert bad['action'] == 'north'
    assert bad['predicted'] == 'North of House'
    assert bad['actual'] == 'West of House'
    assert bad['kinds'] == ['obs']
    assert result['first_mismatch'] is bad
    assert 'RED' in result['report']
    assert "#2 action='north'" in result['report']


def test_backtest_reports_per_step_predicted_versus_actual():
    model = compile_world_model(BROKEN_MODEL)
    result = backtest(model, _toy_trajectory())
    pairs = [(s['index'], s['predicted'], s['actual'], s['ok']) for s in result['steps']]
    assert pairs == [
        (0, 'North of House', 'North of House', True),
        (1, 'North of House', 'North of House', True),
        (2, 'North of House', 'West of House', False),
        (3, 'West of House', 'West of House', True),
    ]


def test_backtest_skips_transitions_with_no_recorded_outcome():
    model = compile_world_model(PERFECT_MODEL)
    traj = [Transition(index=0, obs='West of House', action='north', next_obs=None)]
    result = backtest(model, traj)
    assert result['total'] == 0
    assert result['skipped'] == 1
    assert result['accuracy'] == 0.0
    assert result['green'] is False
    assert 'nothing to check' in result['report']


def test_backtest_accepts_whitespace_only_difference_as_a_warning():
    code = (
        'def step(obs, action):\n'
        '    return "  WEST   of   house  "\n'
    )
    model = compile_world_model(code)
    traj = [Transition(index=0, obs='x', action='look', next_obs='West of House')]
    result = backtest(model, traj)
    assert result['green'] is True
    assert result['steps'][0]['match'] == 'normalized'
    assert result['warnings'] == 1
    assert result['steps'][0]['kinds'] == ['obs_soft']


def test_backtest_flags_a_wrong_flag_prediction():
    code = (
        'def step(obs, action):\n'
        '    return obs, {"progress_increased": True}\n'
    )
    model = compile_world_model(code)
    traj = [Transition(index=0, obs='a', action='look', next_obs='a',
                       flags={'progress_increased': False})]
    result = backtest(model, traj)
    assert result['accuracy'] == 0.0
    assert result['mismatches'][0]['kinds'] == ['progress_increased']
    assert result['mismatches'][0]['predicted_flags'] == {'progress_increased': True}


def test_backtest_ignores_flags_the_recording_does_not_assert():
    code = (
        'def step(obs, action):\n'
        '    return obs, {"win": True}\n'
    )
    model = compile_world_model(code)
    traj = [Transition(index=0, obs='a', action='look', next_obs='a', flags={})]
    assert backtest(model, traj)['green'] is True


def test_backtest_verifies_is_win_condition_on_a_real_win():
    code = (
        'def step(obs, action):\n'
        '    return "You have won!", {"win": True}\n'
        '\n'
        'def is_win_condition(obs):\n'
        '    return "won" in obs\n'
    )
    model = compile_world_model(code)
    assert model.has_win_condition
    traj = [Transition(index=0, obs='a', action='pray', next_obs='You have won!',
                       flags={'win': True})]
    result = backtest(model, traj)
    assert result['green'] is True
    assert result['win_verified'] == 1
    assert 'confirmed on 1 real win' in result['report']


def test_backtest_falsifies_an_over_eager_win_rule():
    code = (
        'def step(obs, action):\n'
        '    return obs\n'
        '\n'
        'def is_win_condition(obs):\n'
        '    return True\n'
    )
    model = compile_world_model(code)
    traj = [Transition(index=0, obs='a', action='look', next_obs='a', flags={'win': False})]
    result = backtest(model, traj)
    assert result['mismatches'][0]['kinds'] == ['win_cond']


def test_backtest_catches_an_exception_raised_by_the_model():
    code = (
        'def step(obs, action):\n'
        '    raise ValueError("I do not know this action")\n'
    )
    model = compile_world_model(code)
    result = backtest(model, _toy_trajectory())
    assert result['accuracy'] == 0.0
    assert all(m['kinds'] == ['error'] for m in result['mismatches'])
    assert 'ValueError' in result['mismatches'][0]['errors'][0]


# ---------------------------------------------------------------------------
# Stateful predict() contract + segments
# ---------------------------------------------------------------------------

def test_stateful_predict_contract_threads_state_through_the_backtest():
    code = (
        'def init_state(entry_obs):\n'
        '    return {"n": 0}\n'
        '\n'
        'def predict(state, obs, action):\n'
        '    n = state["n"] + 1\n'
        '    return "step %d" % n, {}, {"n": n}\n'
    )
    model = compile_world_model(code)
    assert model.stateful
    traj = [
        Transition(index=0, obs='a', action='x', next_obs='step 1'),
        Transition(index=1, obs='step 1', action='x', next_obs='step 2'),
        Transition(index=2, obs='step 2', action='x', next_obs='step 3'),
    ]
    assert backtest(model, traj)['green'] is True


def test_state_is_reinitialised_at_every_segment_boundary():
    code = (
        'def init_state(entry_obs):\n'
        '    return {"n": 0}\n'
        '\n'
        'def predict(state, obs, action):\n'
        '    n = state["n"] + 1\n'
        '    return "step %d" % n, {}, {"n": n}\n'
    )
    model = compile_world_model(code)
    traj = [
        Transition(index=0, obs='a', action='x', next_obs='step 1', segment=0),
        Transition(index=1, obs='step 1', action='x', next_obs='step 2', segment=0),
        Transition(index=2, obs='b', action='x', next_obs='step 1', segment=1),
    ]
    assert backtest(model, traj)['green'] is True


def test_entry_obs_global_is_published_per_segment():
    code = (
        'def step(obs, action):\n'
        '    return ENTRY_OBS\n'
    )
    model = compile_world_model(code)
    traj = [
        Transition(index=0, obs='room A', action='x', next_obs='room A', segment=0),
        Transition(index=1, obs='room B', action='x', next_obs='room B', segment=1),
    ]
    assert backtest(model, traj)['green'] is True


# ---------------------------------------------------------------------------
# Induction loop with an injected stub author (no network)
# ---------------------------------------------------------------------------

def test_extract_code_prefers_the_fenced_block():
    assert extract_code('blah\n```python\ndef step(o, a):\n    return o\n```\ntrailing') == (
        'def step(o, a):\n    return o')
    assert extract_code('def step(o, a):\n    return o') == 'def step(o, a):\n    return o'
    assert extract_code('') == ''


def test_induce_prompt_contains_history_evidence_and_the_import_allowlist():
    wm = WorldModelMechanism(game='zork1')
    prompt = wm.build_prompt(_toy_trajectory())
    assert 'zork1' in prompt
    assert "action='north'" in prompt
    assert 'West of House' in prompt
    assert 'bisect' in prompt and 'numpy' in prompt
    assert 'os' not in prompt.split('You may ONLY import:')[1].split('.')[0]
    for flag in FLAG_NAMES:
        assert flag in prompt


def test_refine_stops_early_on_a_green_backtest():
    author = StubAuthor(PERFECT_MODEL, BROKEN_MODEL)
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(author, _toy_trajectory(), max_rounds=3)
    assert result['green'] is True
    assert tokens == 7
    assert len(author.prompts) == 1          # stopped as soon as it went green
    assert model.has_win_condition is False


def test_refine_feeds_the_mismatch_back_and_accepts_the_corrected_model():
    author = StubAuthor(BROKEN_MODEL, PERFECT_MODEL)
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(author, _toy_trajectory(), max_rounds=3)
    assert len(author.prompts) == 2
    second = author.prompts[1]
    assert 'FALSIFIED' in second
    assert "#2 action='north'" in second     # the earliest falsifying transition
    assert 'def step(obs, action):' in second  # and the program that got it wrong
    assert result['green'] is True
    assert result['accuracy'] == 1.0
    assert tokens == 14
    assert [r['accuracy'] for r in wm.rounds] == [0.75, 1.0]


def test_refine_keeps_the_most_accurate_model_when_nothing_goes_green():
    half_right = (
        'def step(obs, action):\n'
        '    return obs\n'
    )
    author = StubAuthor(BROKEN_MODEL, half_right)
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(author, _toy_trajectory(), max_rounds=2)
    assert result['green'] is False
    assert result['accuracy'] == 0.75        # BROKEN_MODEL beat the no-op model
    assert wm.best is result


def test_refine_survives_code_that_will_not_load():
    author = StubAuthor('import os\n\ndef step(obs, action):\n    return obs\n', PERFECT_MODEL)
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(author, _toy_trajectory(), max_rounds=2)
    assert wm.rounds[0]['error'] is not None
    assert 'could not even be loaded' in author.prompts[1]
    assert result['green'] is True


def test_refine_survives_an_author_that_raises():
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(RaisingAuthor(), _toy_trajectory(), max_rounds=2)
    assert model is None
    assert result is None
    assert tokens == 0
    assert all(r['error'].startswith('author_fn failed') for r in wm.rounds)


def test_llm_author_adapter_uses_generate_and_parse_completion():
    client = FakeLLMClient(f'```python\n{PERFECT_MODEL}\n```', tokens=42)
    author = llm_author(client)
    wm = WorldModelMechanism(game='zork1')
    model, result, tokens = wm.refine(author, _toy_trajectory(), max_rounds=1)
    assert len(client.calls) == 1
    assert tokens == 42
    assert result['green'] is True


# ---------------------------------------------------------------------------
# Minimal self-runner (pytest is not installed in this project's interpreter)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f'FAIL {name}')
        else:
            print(f'ok   {name}')
    print(f'\n{len(tests) - len(failures)} passed, {len(failures)} failed')
    for name, tb in failures:
        print(f'\n--- {name} ---\n{tb}')
    sys.exit(1 if failures else 0)
