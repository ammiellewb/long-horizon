"""M7 (candidate): executable world model + backtest-against-history.

Port of the core technique from *Schema: Discovering Unknown Environments via
Agentic Program Induction* (Zeng et al., 2026; arXiv:2609.39140), specifically
``digbench/schema/agent/world_model/{world.py,tools.py}``.

The technique
-------------
Instead of describing the environment in prose, the agent writes its
understanding of the environment as an **executable Python program** — a world
model defining ``step(obs, action)`` (or the stateful ``predict(state, obs,
action)``) plus an optional ``is_win_condition`` predicate. The harness then
**backtests that program against the full recorded interaction history**: it
replays every transition the agent actually took, compares the program's
predicted next observation against what really happened, and reports the
earliest mispredicted transition. The agent rewrites the program only where
reality falsified it. That falsification loop — not the prose summary, not the
single-step replan — is the mechanism.

Relation to the mechanisms already in this package
--------------------------------------------------
  * ``state_ext.py`` (M3) probes the live env for ground-truth location /
    inventory. That is a *measurement* of the present state.
  * ``planning.py`` (M6) asks the LLM for one refreshed subgoal. That is a
    *single-shot* re-plan with no verification.
  * This module asks the LLM for a *falsifiable artifact* (code) and then
    verifies it against the whole trajectory. The LLM's claim about the
    environment can be proven wrong by replay, which neither M3 nor M6 can do.

Design constraints (deliberate)
-------------------------------
  * **No network, no LLM import.** The call that AUTHORS the world-model code is
    injected as a plain callable (``author_fn``); see ``llm_author()`` for an
    adapter around any object with the repo's ``generate`` /
    ``parse_completion`` duck type. This module never imports ``llm.py``.
  * **Stdlib only** (numpy is used only if already installed, matching Schema,
    which preloads it as ``np``).
  * **Integration is deferred**: nothing here touches ``runner.py``. See
    ``README_world_model.md`` for the exact hook the runner needs.

Security
--------
``compile_world_model(code)`` executes model-generated code. The sandbox is
Schema's allowlist approach (``_ALLOWED_IMPORTS`` / ``_SAFE_BUILTINS`` /
``_safe_import``), unwidened, plus two additions: a static AST pre-check that
rejects dunder-traversal escape vectors and non-allowlisted imports before any
code runs, and a wall-clock timeout on every call into model code. This is
*not* an OS-level jail — see the "Threat model" note in the README.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import re
import sys
import threading
import time
from dataclasses import dataclass, field

try:  # numpy is preloaded as `np` by Schema; optional here.
    import numpy as _np
except Exception:  # pragma: no cover - numpy absent is a supported config
    _np = None


# ---------------------------------------------------------------------------
# Sandbox: allowlists (ported verbatim from Schema world.py lines 12-45)
# ---------------------------------------------------------------------------

# Exactly Schema's set. Do NOT widen: every name here is a module the model
# may import inside its world model. Nothing that touches the filesystem,
# the network, the process, or the import system is present.
_ALLOWED_IMPORTS = {
    'numpy', 'math', 'collections', 'itertools', 'functools', 'heapq', 're', 'string',
    'copy', 'dataclasses', 'typing', 'enum', 'operator', 'bisect',
}

# Exactly Schema's set. Notably ABSENT (and therefore unavailable to model
# code): eval, exec, compile, open, input, __import__ (replaced below by
# _safe_import), globals, locals, vars, dir, breakpoint, help, exit, quit,
# memoryview, print.
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        'range', 'len', 'min', 'max', 'abs', 'enumerate', 'zip', 'list', 'dict',
        'set', 'tuple', 'int', 'float', 'bool', 'str', 'sum', 'sorted', 'any',
        'all', 'map', 'filter', 'reversed', 'round', 'divmod', 'pow', 'isinstance',
        'getattr', 'setattr', 'hasattr', 'Exception', 'ValueError', 'IndexError',
        'KeyError', 'TypeError', 'ZeroDivisionError', 'slice', 'frozenset', 'bytes',
        'chr', 'ord', 'repr', 'hash', 'type', 'iter', 'next', 'callable', 'object',
        'StopIteration', 'AttributeError', 'RuntimeError', 'AssertionError',
        'NotImplementedError', 'property', 'staticmethod', 'classmethod', 'super',
        'id', 'format', 'bytearray',
    )
}

# Local hardening (an ADDITION, not a widening). Schema's allowlist leaves
# `getattr`/`type`/`object`/`super` available, which in principle permits the
# classic ``().__class__.__bases__[0].__subclasses__()`` walk back to the real
# builtins. These attribute names are rejected statically before execution.
_BLOCKED_ATTRS = frozenset({
    '__class__', '__bases__', '__base__', '__mro__', '__subclasses__',
    '__globals__', '__code__', '__builtins__', '__dict__', '__getattribute__',
    '__reduce__', '__reduce_ex__', '__init_subclass__', '__subclasshook__',
    '__closure__', '__func__', '__self__', '__loader__', '__spec__',
    '__module__', '__import__', 'gi_frame', 'f_globals', 'f_builtins', 'f_locals',
})

# Names that must never be referenced by model code. All are already missing
# from _SAFE_BUILTINS (so they would raise NameError at runtime); rejecting
# them statically turns a confusing mid-rollout NameError into a clear
# load-time error the authoring model can act on.
_BLOCKED_NAMES = frozenset({
    'eval', 'exec', 'compile', 'open', 'input', '__import__', 'globals', 'locals',
    'vars', 'dir', 'breakpoint', 'help', 'exit', 'quit', 'memoryview', 'print',
    '__builtins__', '__loader__', '__spec__',
})

# Flag names the model may return in `info`, adapted from Schema's
# ("level_up", "life_lost", "dead", "win") to what a TextQuests episode
# actually records (progress %, score, game_over from env.step).
FLAG_NAMES = ('progress_increased', 'score_increased', 'game_over', 'win')

DEFAULT_LOAD_TIMEOUT_S = 5.0
DEFAULT_CALL_TIMEOUT_S = 2.0


class SandboxError(Exception):
    """Model-generated code was rejected, failed to load, or blew its budget."""


class SandboxTimeout(SandboxError):
    """Model-generated code exceeded its wall-clock budget."""


class _Deadline(BaseException):
    """Internal timeout signal.

    Derives from BaseException, not Exception, so that a world model doing
    ``while True: try: ... except Exception: pass`` cannot swallow its own
    timeout. Converted to SandboxTimeout at the sandbox boundary.
    """


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Schema's import hook: allowlist the root module, then delegate."""
    root = name.split('.')[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import '{name}' is not allowed in world-model code")
    return __import__(name, globals, locals, fromlist, level)


def _static_check(code):
    """Reject obvious escape vectors before executing anything.

    Raises SandboxError on a syntax error, a non-allowlisted import, a
    reference to a blocked builtin name, or an attribute access that could
    walk out of the sandbox via dunder traversal.
    """
    try:
        tree = ast.parse(code, filename='<world_model>')
    except SyntaxError as exc:
        raise SandboxError(f'world-model code does not parse: {exc}') from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    raise SandboxError(
                        f"import '{alias.name}' is not allowed in world-model code"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if node.level or root not in _ALLOWED_IMPORTS:
                raise SandboxError(
                    f"import from '{node.module or '.'}' is not allowed in world-model code"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                raise SandboxError(
                    f"attribute '{node.attr}' is not allowed in world-model code "
                    '(sandbox-escape vector)'
                )
        elif isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES:
                raise SandboxError(
                    f"name '{node.id}' is not available in world-model code"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Blocks getattr(x, "__class__") style indirection.
            if node.value in _BLOCKED_ATTRS:
                raise SandboxError(
                    f"string literal {node.value!r} is not allowed in world-model code "
                    '(sandbox-escape vector)'
                )
    return tree


def _call_guarded(fn, args=(), kwargs=None, timeout_s=DEFAULT_CALL_TIMEOUT_S, what='code'):
    """Call ``fn`` with a wall-clock budget, in a dedicated traced thread.

    The worker installs a per-line trace hook that raises ``_Deadline`` once
    the budget is spent, which interrupts pure-Python loops. If the worker is
    still alive after the budget plus a grace period (possible when the code
    is blocked inside a C call that cannot be traced), it is abandoned as a
    daemon thread and SandboxTimeout is raised anyway, so the caller never
    hangs. ``timeout_s=None`` disables guarding (faster; use only for trusted
    code or long offline backtests).
    """
    kwargs = kwargs or {}
    if timeout_s is None:
        return fn(*args, **kwargs)

    box = {}

    def worker():
        deadline = time.monotonic() + timeout_s

        def tracer(frame, event, arg):
            if time.monotonic() > deadline:
                raise _Deadline()
            return tracer

        sys.settrace(tracer)
        try:
            box['value'] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box['error'] = exc
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=worker, daemon=True, name='world-model-exec')
    thread.start()
    thread.join(timeout_s + 1.0)
    if thread.is_alive():
        raise SandboxTimeout(
            f'{what} exceeded its {timeout_s}s budget and did not stop when interrupted '
            '(abandoned; it was likely blocked inside a C call)'
        )
    if 'error' in box:
        err = box['error']
        if isinstance(err, _Deadline):
            raise SandboxTimeout(f'{what} exceeded its {timeout_s}s wall-clock budget')
        raise err
    return box.get('value')


def _exec_code(code, timeout_s=DEFAULT_LOAD_TIMEOUT_S):
    """Static-check then execute world-model source in a restricted namespace.

    Returns the resulting namespace dict. Mirrors Schema's ``_exec_code`` with
    the globals it preloads (``np``, ``ENTRY_OBS``, ``CURRENT_LEVEL``,
    ``CURRENT_MODE``); ``CURRENT_LEVEL``/``CURRENT_MODE`` are kept for source
    parity and are simply None in a TextQuests run.
    """
    _static_check(code)
    ns = {
        '__builtins__': {**_SAFE_BUILTINS, '__import__': _safe_import},
        'np': _np,
        'numpy': _np,
        'ENTRY_OBS': None,
        'CURRENT_LEVEL': None,
        'CURRENT_MODE': None,
    }
    try:
        compiled = compile(code, '<world_model>', 'exec')
    except (SyntaxError, ValueError) as exc:
        raise SandboxError(f'world-model code failed to compile: {exc}') from exc
    try:
        _call_guarded(exec, (compiled, ns), timeout_s=timeout_s, what='module-level code')
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxError(
            f'world-model code raised at import time: {type(exc).__name__}: {exc}'
        ) from exc
    return ns


# ---------------------------------------------------------------------------
# Observation / signature helpers (ported from Schema world.py)
# ---------------------------------------------------------------------------

def _to_text(obs):
    if obs is None:
        return ''
    if isinstance(obs, str):
        return obs
    if isinstance(obs, (list, tuple)) and all(isinstance(x, str) for x in obs):
        return '\n'.join(obs)
    return str(obs)


def _arity(fn):
    """Count positional parameters, so a predicate may be f(obs) or f(state, obs)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    n = 0
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind is p.VAR_POSITIONAL:
            return 99
    return n


_WS_RE = re.compile(r'\s+')


def normalize_obs(text):
    """Whitespace/case-insensitive form used for the 'soft match' warning."""
    return _WS_RE.sub(' ', _to_text(text).strip().lower())


def diff_summary(pred, actual):
    """One-line human-readable difference report (Schema world.py)."""
    if pred is None or actual is None:
        return 'no prediction' if pred is None else 'no actual observation'
    if pred == actual:
        return 'identical'
    if len(pred) != len(actual):
        head = f'length {len(pred)} (predicted) vs {len(actual)} (actual)'
    else:
        head = f'same length {len(actual)}'
    n = min(len(pred), len(actual))
    first = next((i for i in range(n) if pred[i] != actual[i]), n)
    diffs = sum(1 for i in range(n) if pred[i] != actual[i]) + abs(len(pred) - len(actual))
    return (f'{head}; {diffs} char position(s) differ; first difference at index {first} '
            f'(predicted {pred[first:first + 8]!r} vs actual {actual[first:first + 8]!r})')


def match_level(pred, actual):
    """Return 'exact', 'normalized', or None.

    Schema accepts a prediction that matches the current frame but differs in
    the embedded-history prefix and downgrades it to a warning. TextQuests
    observations have no such prefix, so the analogous leniency here is a
    whitespace/case-insensitive match: scored as a warning, not a mismatch.
    """
    if pred is None or actual is None:
        return None
    if pred == actual:
        return 'exact'
    if normalize_obs(pred) == normalize_obs(actual):
        return 'normalized'
    return None


# ---------------------------------------------------------------------------
# The world model wrapper
# ---------------------------------------------------------------------------

class WorldModel:
    """Agent-authored environment program, loaded and callable under sandbox.

    The model must define exactly one transition function:

        def step(obs, action):
            # -> next_obs  |  (next_obs, info)

        def predict(state, obs, action):
            # -> (next_obs, info, next_state); pair with optional
            #    init_state(entry_obs) for the initial state value.

    ``info`` is an optional dict over FLAG_NAMES. Optional predicates, either
    ``f(obs)`` or ``f(state, obs)``, returning bool:

        is_win_condition(...)   the model's hypothesis about what ends the game
                                in a win. CHECKED by backtest().
        is_bfs_goal(...)        a self-chosen planning waypoint. NEVER checked
                                (kept for source parity with Schema; this port
                                ships no planner, see README).

    Every call into model code is wrapped in the wall-clock guard.
    """

    def __init__(self, code, call_timeout_s=DEFAULT_CALL_TIMEOUT_S,
                 load_timeout_s=DEFAULT_LOAD_TIMEOUT_S):
        ns = _exec_code(code, timeout_s=load_timeout_s)
        predict = ns.get('predict')
        step = ns.get('step')
        self.stateful = callable(predict)
        if not self.stateful and not callable(step):
            raise SandboxError(
                'world-model code must define predict(state, obs, action) OR step(obs, action)'
            )
        self.code = code
        self.call_timeout_s = call_timeout_s
        self._ns = ns
        self._predict = predict if self.stateful else None
        self._step = step if callable(step) else None
        init_state = ns.get('init_state')
        self._init_state = init_state if callable(init_state) else None

        def _pred(name):
            fn = ns.get(name)
            return (fn, _arity(fn)) if callable(fn) else (None, 0)

        self._win, self._win_arity = _pred('is_win_condition')
        self._bfs_goal, self._bfs_goal_arity = _pred('is_bfs_goal')

    # -- sandboxed invocation ------------------------------------------------

    def _guard(self, fn, *args):
        return _call_guarded(fn, args, timeout_s=self.call_timeout_s,
                             what=f'{getattr(fn, "__name__", "model code")}()')

    # -- segment / entry context --------------------------------------------

    def set_entry(self, entry):
        """Publish the current segment's entry observation as ``ENTRY_OBS``.

        Schema also publishes CURRENT_LEVEL / CURRENT_MODE per segment; both
        are None for TextQuests but are set for source parity.
        """
        entry = entry or {}
        obs = entry.get('obs')
        self._ns['ENTRY_OBS'] = None if obs is None else _to_text(obs)
        self._ns['CURRENT_LEVEL'] = entry.get('level')
        self._ns['CURRENT_MODE'] = entry.get('mode')

    def init_state(self, entry_obs):
        if not self.stateful:
            return None
        if self._init_state is None:
            return {}
        return self._guard(self._init_state, None if entry_obs is None else _to_text(entry_obs))

    # -- contract ------------------------------------------------------------

    def predict(self, state, obs, action):
        """-> (next_obs: str, info: dict, next_state). Works for both contracts."""
        if self.stateful:
            out = self._guard(self._predict, state, _to_text(obs), str(action))
            return self._unpack(out, state)
        out = self._guard(self._step, _to_text(obs), str(action))
        obs_out, info, _ = self._unpack(out, None)
        return obs_out, info, None

    def step(self, obs, action):
        """-> (next_obs: str, info: dict), stateless convenience form."""
        st = self.init_state(None) if self.stateful else None
        obs_out, info, _ = self.predict(st, obs, action)
        return obs_out, info

    @staticmethod
    def _unpack(out, prev_state):
        info = {}
        next_state = prev_state
        if isinstance(out, tuple):
            obs = out[0] if out else ''
            if len(out) > 1 and isinstance(out[1], dict):
                info = out[1]
            if len(out) > 2:
                next_state = out[2]
        else:
            obs = out
        return _to_text(obs), dict(info or {}), next_state

    def _call_pred(self, fn, arity, obs, state):
        if arity >= 2:
            return bool(self._guard(fn, state, _to_text(obs)))
        return bool(self._guard(fn, _to_text(obs)))

    @property
    def has_win_condition(self):
        return self._win is not None

    @property
    def has_bfs_goal(self):
        return self._bfs_goal is not None

    def is_win_condition(self, obs, state=None):
        """-> bool, or None when the model defines no win rule."""
        if self._win is None:
            return None
        return self._call_pred(self._win, self._win_arity, obs, state)

    def is_bfs_goal(self, obs, state=None):
        """-> bool, or None. Planning hint only; never scored by backtest()."""
        if self._bfs_goal is None:
            return None
        return self._call_pred(self._bfs_goal, self._bfs_goal_arity, obs, state)


def compile_world_model(code, **kwargs):
    """Load agent-authored code into a WorldModel, or raise SandboxError."""
    try:
        return WorldModel(code, **kwargs)
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxError(f'{type(exc).__name__}: {exc}') from exc


# ---------------------------------------------------------------------------
# Trajectory representation + adapters
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One recorded (obs, action) -> next_obs transition to backtest against.

    index:    position in the trajectory (used in mismatch reports).
    obs:      observation the agent acted on.
    action:   action string actually submitted.
    next_obs: observation the environment actually returned (None = unknown,
              e.g. the last recorded step; such a step is skipped).
    flags:    ground truth over FLAG_NAMES, as far as the recording knows.
    segment:  episode/segment id. State is re-initialised at every boundary,
              exactly as Schema re-inits per level / life / mode segment.
    meta:     free-form provenance (step number, progress, score, ...).
    """

    index: int
    obs: str
    action: str
    next_obs: str = None
    flags: dict = field(default_factory=dict)
    segment: int = 0
    meta: dict = field(default_factory=dict)


def trajectory_from_history(history, segment=0):
    """Build a trajectory from ``runner._run_loop``'s in-memory ``history``.

    The runner appends ``{'obs', 'action', 'reasoning', 'progress'}`` where
    ``obs`` is the observation the agent acted ON and ``progress`` is the
    progress AFTER the action. The resulting observation of step i is therefore
    the ``obs`` of step i+1, which is what this adapter pairs up. The final
    recorded step has no successor and is dropped.
    """
    out = []
    for i in range(len(history) - 1):
        cur, nxt = history[i], history[i + 1]
        prev_progress = history[i - 1].get('progress', 0) if i else 0
        out.append(Transition(
            index=i,
            obs=cur.get('obs') or '',
            action=cur.get('action') or '',
            next_obs=nxt.get('obs'),
            flags={'progress_increased': bool((cur.get('progress') or 0) > (prev_progress or 0))},
            segment=segment,
            meta={'progress': cur.get('progress')},
        ))
    return out


def trajectory_from_log(log):
    """Build a trajectory from a saved ``logs/<run_id>.json`` transcript.

    ``log`` is the parsed JSON dict (or just its ``steps`` list). Flags are
    derived from the recorded progress/score deltas, which is all the
    transcript carries — see the README on what the runner would need to log
    for the ``game_over`` / ``win`` flags to be checkable.
    """
    steps = log.get('steps', []) if isinstance(log, dict) else list(log or [])
    out = []
    for i in range(len(steps) - 1):
        cur, nxt = steps[i], steps[i + 1]
        prev = steps[i - 1] if i else {}
        out.append(Transition(
            index=i,
            obs=cur.get('obs') or '',
            action=cur.get('action') or '',
            next_obs=nxt.get('obs'),
            flags={
                'progress_increased': bool((cur.get('progress') or 0) > (prev.get('progress') or 0)),
                'score_increased': bool((cur.get('score') or 0) > (prev.get('score') or 0)),
            },
            segment=0,
            meta={'step': cur.get('step'), 'progress': cur.get('progress'),
                  'score': cur.get('score')},
        ))
    return out


def segment_entries(trajectory):
    """-> {segment_id: {'obs': first observation of that segment}}."""
    entries = {}
    for t in trajectory:
        entries.setdefault(t.segment, {'obs': t.obs})
    return entries


# ---------------------------------------------------------------------------
# Backtest: replay the induced program against recorded history
# ---------------------------------------------------------------------------

def _flag_mismatches(info, truth):
    """Compare predicted flags against ground truth.

    Only flags the recording actually asserts are checked: a flag absent from
    ``truth`` is unknown, not False, so predicting it is never penalised.
    This is deliberately more lenient than Schema, whose bench reports every
    flag on every step (see README).
    """
    errors, kinds = [], []
    for flag in FLAG_NAMES:
        if flag not in truth:
            continue
        pf, af = bool((info or {}).get(flag)), bool(truth.get(flag))
        if pf == af:
            continue
        errors.append(f'{flag} prediction error: predicted {pf}, actual {af}')
        kinds.append(flag)
    return errors, kinds


def _score_step(model, pred, info, next_state, tr):
    """Score one replayed transition -> {errors, kinds, warnings, win_verified, match}."""
    errors, kinds, warnings = [], [], []
    win_verified = False

    flag_errors, flag_kinds = _flag_mismatches(info, tr.flags)
    errors += flag_errors
    kinds += flag_kinds

    match = None
    if tr.next_obs is not None:
        match = match_level(pred, tr.next_obs)
        if match is None:
            errors.append('observation prediction error: ' + diff_summary(pred, tr.next_obs))
            kinds.append('obs')
        elif match == 'normalized':
            warnings.append('observation matches apart from whitespace/case '
                            '(not counted as a mismatch)')
            kinds.append('obs_soft')

    if model.has_win_condition:
        really_won = bool(tr.flags.get('win'))
        try:
            if tr.next_obs is not None:
                says_win = model.is_win_condition(tr.next_obs, next_state)
                if says_win and not really_won and 'win' in tr.flags:
                    errors.append('win_condition error: is_win_condition says this reached state '
                                  'wins the game, but in reality it did not')
                    kinds.append('win_cond')
                elif really_won and not says_win:
                    errors.append('win_condition error: is_win_condition is False on the REAL '
                                  'observation that ended the game in a win — the win rule does '
                                  'not explain this actual completion')
                    kinds.append('win_cond')
                elif really_won and says_win:
                    win_verified = True
        except SandboxError as exc:
            errors.append(f'is_win_condition sandbox failure: {exc}')
            kinds.append('win_cond')
        except Exception as exc:
            errors.append(f'is_win_condition raised {type(exc).__name__}: {exc}')
            kinds.append('win_cond')

    return {'errors': errors, 'kinds': kinds, 'warnings': warnings,
            'win_verified': win_verified, 'match': match}


def backtest(model, trajectory, max_details=5):
    """Replay ``model`` against a recorded ``trajectory`` and score it.

    This is the falsification step: ``runner.py`` never asks the model whether
    it is right, it replays what actually happened and checks.

    Args:
        model: a WorldModel (or anything with the same duck type).
        trajectory: list of Transition (see trajectory_from_history /
            trajectory_from_log). Transitions with ``next_obs is None`` are
            skipped as unscoreable.
        max_details: how many mismatches to render into ``report``.

    Returns a dict:
        {
          'total':        int,   # transitions actually scored
          'skipped':      int,   # unscoreable (no recorded next_obs)
          'exact':        int,   # reproduced with no errors
          'accuracy':     float, # exact / total, 0.0 when total == 0
          'green':        bool,  # accuracy == 1.0 and total > 0
          'win_verified': int,   # real wins the win rule explained
          'warnings':     int,
          'steps': [             # per-step predicted vs actual, in order
            {'index', 'segment', 'action', 'predicted', 'actual', 'match',
             'predicted_flags', 'actual_flags', 'errors', 'kinds',
             'warnings', 'ok'}
          ],
          'mismatches': [...],   # the subset of 'steps' with errors, in order
          'first_mismatch': dict|None,   # earliest falsifying transition
          'report': str,         # human/LLM-readable summary for the next round
        }
    """
    entries = segment_entries(trajectory)
    steps = []
    skipped = 0
    cur_segment = object()
    state = None

    for tr in trajectory:
        if tr.next_obs is None:
            skipped += 1
            continue

        entry = entries.get(tr.segment)
        if tr.segment != cur_segment:
            cur_segment = tr.segment
            try:
                model.set_entry(entry)
                state = model.init_state(entry.get('obs') if entry else None)
            except Exception as exc:
                steps.append(_failed_step(tr, f'init_state raised {type(exc).__name__}: {exc}'))
                state = None
                continue
        else:
            model.set_entry(entry)

        try:
            pred, info, next_state = model.predict(state, tr.obs, tr.action)
        except SandboxError as exc:
            steps.append(_failed_step(tr, f'sandbox failure: {exc}'))
            continue
        except Exception as exc:
            steps.append(_failed_step(tr, f'predict() raised {type(exc).__name__}: {exc}'))
            continue

        scored = _score_step(model, pred, info, next_state, tr)
        state = next_state
        steps.append({
            'index': tr.index,
            'segment': tr.segment,
            'action': tr.action,
            'predicted': pred,
            'actual': tr.next_obs,
            'match': scored['match'],
            'predicted_flags': {k: bool(v) for k, v in (info or {}).items() if k in FLAG_NAMES},
            'actual_flags': dict(tr.flags),
            'errors': scored['errors'],
            'kinds': scored['kinds'],
            'warnings': scored['warnings'],
            'win_verified': scored['win_verified'],
            'ok': not scored['errors'],
        })

    total = len(steps)
    mismatches = [s for s in steps if not s['ok']]
    exact = total - len(mismatches)
    accuracy = (exact / total) if total else 0.0
    result = {
        'total': total,
        'skipped': skipped,
        'exact': exact,
        'accuracy': accuracy,
        'green': bool(total) and not mismatches,
        'win_verified': sum(1 for s in steps if s.get('win_verified')),
        'warnings': sum(len(s['warnings']) for s in steps),
        'steps': steps,
        'mismatches': mismatches,
        'first_mismatch': mismatches[0] if mismatches else None,
    }
    result['report'] = format_backtest(result, max_details=max_details)
    return result


def _failed_step(tr, message):
    return {
        'index': tr.index, 'segment': tr.segment, 'action': tr.action,
        'predicted': None, 'actual': tr.next_obs, 'match': None,
        'predicted_flags': {}, 'actual_flags': dict(tr.flags),
        'errors': [message], 'kinds': ['error'], 'warnings': [],
        'win_verified': False, 'ok': False,
    }


def format_backtest(result, max_details=5):
    """Render a backtest result as the text fed back to the authoring model."""
    if not result['total']:
        return ('backtest: nothing to check yet (no transition had a recorded next '
                'observation). Take a few steps first.')
    head = (f"backtest — {'GREEN' if result['green'] else 'RED'}: "
            f"{result['exact']}/{result['total']} transitions reproduced exactly "
            f"(accuracy {result['accuracy']:.2f}"
            + (f", {len(result['mismatches'])} mismatch(es)" if result['mismatches'] else '')
            + (f", {result['skipped']} skipped" if result['skipped'] else '') + ').')
    lines = [head]
    if result['warnings']:
        lines.append(f"{result['warnings']} step(s) matched only after whitespace/case "
                     'normalisation (accepted).')
    if result['win_verified']:
        lines.append(f"Win rule: is_win_condition confirmed on {result['win_verified']} real win(s).")
    if result['green']:
        lines.append('Verdict: the model explains every transition it has seen. A green backtest '
                     'only covers actions actually taken.')
        return '\n'.join(lines)
    for m in result['mismatches'][:max_details]:
        lines.append(f"\n#{m['index']} action={m['action']!r} kinds={m['kinds'] or ['-']}")
        for err in m['errors']:
            lines.append(f'  - {err}')
        if m['predicted'] is not None:
            lines.append(f"  predicted: {m['predicted']!r}")
        lines.append(f"  actual:    {m['actual']!r}")
    hidden = len(result['mismatches']) - max_details
    if hidden > 0:
        lines.append(f'\n(+{hidden} more mismatch(es) not shown)')
    lines.append('\nNext: fix the RULE behind the EARLIEST mismatch (not that one transition), '
                 'then re-run the backtest.')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Induction: the LLM authors the program; this module only verifies it
# ---------------------------------------------------------------------------

_INDUCE_PROMPT = (
    'You are modelling an unknown text-adventure environment ({game}) as an executable '
    'Python program.\n\n'
    'Write a WORLD MODEL: Python source defining ONE of\n'
    '    def step(obs, action):              # -> next_obs, or (next_obs, info)\n'
    '    def predict(state, obs, action):    # -> (next_obs, info, next_state)\n'
    'where obs and the returned next observation are the exact observation strings the game '
    'prints, and info is an optional dict over {flags}.\n'
    'With predict() you may also define init_state(entry_obs) for the initial state value.\n'
    'Optionally define is_win_condition(obs) (or is_win_condition(state, obs)) -> bool: your '
    'hypothesis about which reached state ends the game in a win. It is part of the model and '
    'the backtest checks it.\n\n'
    'You may ONLY import: {imports}. No file, network, OS or process access; the history '
    'reaches your model exclusively through the function arguments. A global ENTRY_OBS holds '
    "the segment's first observation — ground the layout in it rather than hard-coding.\n\n"
    'RECORDED HISTORY (what actually happened; this is the only evidence):\n{history}\n'
    '{feedback}'
    '\nReturn ONLY one ```python code block containing the whole program.'
)

_FEEDBACK_TEMPLATE = (
    '\nYOUR PREVIOUS MODEL WAS FALSIFIED by replaying it against that history:\n'
    '{report}\n'
    'Previous program:\n```python\n{code}\n```\n'
)

_CODE_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def extract_code(text):
    """Pull the world-model source out of an LLM reply (fenced block, else raw)."""
    if not text:
        return ''
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def render_history(trajectory, max_steps=40, max_obs_chars=600):
    """Render a trajectory as the evidence block of the induction prompt."""
    tail = trajectory[-max_steps:]
    lines = []
    for tr in tail:
        def clip(s):
            s = _to_text(s)
            return s if len(s) <= max_obs_chars else s[:max_obs_chars] + '...<clipped>'
        lines.append(f'#{tr.index} seg={tr.segment} action={tr.action!r}')
        lines.append(f'  obs:      {clip(tr.obs)!r}')
        lines.append(f'  next_obs: {clip(tr.next_obs)!r}')
        if tr.flags:
            lines.append(f'  flags:    {dict(tr.flags)}')
    hidden = len(trajectory) - len(tail)
    if hidden > 0:
        lines.insert(0, f'({hidden} earlier transition(s) omitted)')
    return '\n'.join(lines) if lines else '(no transitions recorded yet)'


def llm_author(llm_client, max_tokens=2048, reasoning_enabled=False):
    """Adapt a repo LLM client into the ``author_fn(prompt) -> (text, tokens)`` dependency.

    Duck-typed on ``generate(prompt, max_tokens=..., reasoning_enabled=...)`` +
    ``parse_completion(resp)``, which is what ``llm.OpenRouterClient`` provides
    and what ``tests/test_harness.FakeLLMClient`` fakes. This module does not
    import ``llm.py``: pass the client in. Calling the returned callable is the
    ONLY thing in this module that can reach the network.
    """

    def author_fn(prompt):
        resp = llm_client.generate(prompt, max_tokens=max_tokens,
                                   reasoning_enabled=reasoning_enabled)
        parsed = llm_client.parse_completion(resp)
        if isinstance(parsed, dict):
            return parsed.get('content', ''), parsed.get('tokens', 0) or 0
        return str(parsed), 0

    return author_fn


def _call_author(author_fn, prompt):
    """Normalise author_fn's return into (text, tokens)."""
    out = author_fn(prompt)
    if isinstance(out, tuple):
        text = out[0] if out else ''
        tokens = out[1] if len(out) > 1 else 0
        return _to_text(text), int(tokens or 0)
    if isinstance(out, dict):
        return _to_text(out.get('content', '')), int(out.get('tokens', 0) or 0)
    return _to_text(out), 0


class WorldModelMechanism:
    """M7 (candidate): induce an executable world model, then falsify it on history.

    One round is: ask ``author_fn`` for a program, load it under the sandbox,
    backtest it against the recorded trajectory, and — if the replay falsified
    it — hand the earliest mismatch back and ask again. The LLM dependency is
    injected, so this class is fully testable offline (see
    tests/test_world_model.py).

    Usage (after the runner refactor lands; see README_world_model.md):

        wm = WorldModelMechanism(game='zork1')
        model, result, tokens = wm.refine(
            author_fn=llm_author(self.llm),
            trajectory=trajectory_from_history(history),
            max_rounds=3,
        )
    """

    def __init__(self, game=None, max_rounds=3, max_history_steps=40,
                 call_timeout_s=DEFAULT_CALL_TIMEOUT_S,
                 load_timeout_s=DEFAULT_LOAD_TIMEOUT_S):
        self.game = game
        self.max_rounds = max_rounds
        self.max_history_steps = max_history_steps
        self.call_timeout_s = call_timeout_s
        self.load_timeout_s = load_timeout_s
        self.model = None          # best model loaded so far
        self.best = None           # its backtest result
        self.rounds = []           # [{round, code, error, accuracy, result}]

    def build_prompt(self, trajectory, previous_code=None, previous_report=None):
        feedback = ''
        if previous_code and previous_report:
            feedback = _FEEDBACK_TEMPLATE.format(report=previous_report, code=previous_code)
        return _INDUCE_PROMPT.format(
            game=self.game or 'unknown game',
            flags=', '.join(FLAG_NAMES),
            imports=', '.join(sorted(_ALLOWED_IMPORTS)),
            history=render_history(trajectory, max_steps=self.max_history_steps),
            feedback=feedback,
        )

    def induce(self, author_fn, trajectory, previous_code=None, previous_report=None):
        """One authoring round -> (model|None, code, error|None, tokens).

        Never raises on a bad program: a SandboxError becomes the ``error``
        string, which is exactly the feedback the next round needs.
        """
        prompt = self.build_prompt(trajectory, previous_code, previous_report)
        try:
            text, tokens = _call_author(author_fn, prompt)
        except Exception as exc:
            return None, '', f'author_fn failed: {type(exc).__name__}: {exc}', 0
        code = extract_code(text)
        if not code:
            return None, '', 'author_fn returned no code', tokens
        try:
            model = compile_world_model(code, call_timeout_s=self.call_timeout_s,
                                        load_timeout_s=self.load_timeout_s)
        except SandboxError as exc:
            return None, code, str(exc), tokens
        return model, code, None, tokens

    def refine(self, author_fn, trajectory, max_rounds=None):
        """Run the induce -> backtest -> falsify loop.

        Returns ``(model|None, result|None, tokens)`` for the most accurate
        model found, and stops early on a green backtest. ``self.rounds``
        keeps the per-round record for logging.
        """
        rounds = self.max_rounds if max_rounds is None else max_rounds
        total_tokens = 0
        best_model, best_result = None, None
        code, report = None, None

        for i in range(max(1, rounds)):
            model, code_out, error, tokens = self.induce(author_fn, trajectory, code, report)
            total_tokens += tokens
            code = code_out or code
            if model is None:
                self.rounds.append({'round': i, 'code': code_out, 'error': error,
                                    'accuracy': None, 'result': None})
                report = f'The program could not even be loaded: {error}'
                continue
            result = backtest(model, trajectory)
            self.rounds.append({'round': i, 'code': code_out, 'error': None,
                                'accuracy': result['accuracy'], 'result': result})
            if best_result is None or result['accuracy'] > best_result['accuracy']:
                best_model, best_result = model, result
            if result['green']:
                break
            report = result['report']

        self.model, self.best = best_model, best_result
        return best_model, best_result, total_tokens
