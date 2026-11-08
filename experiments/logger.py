"""Run logger.

Accumulates step records in memory during a run, then writes a single
structured JSON transcript when save() is called.

Output file: logs/<run_id>.json

Schema
------
{
  "run_id":    str,
  "game":      str,
  "summary": {
    "total_steps":    int,
    "final_progress": float,   # 0–100 %
    "final_score":    int,
    "game_over":      bool,
    "stop_reason":    str,     # one of STOP_REASONS; REQUIRED (see below)
    "model":          str,     # resolved model id actually sent to the provider
    "model_source":   str,     # which source that id came from (env var / config.yaml)
    "context_evictions": int,  # oldest history pairs dropped by truncation this run
    "truncated":      bool,    # context_evictions > 0
    "overflow_recoveries": int,# provider context-overflow errors recovered from
    "started_at":     str,     # ISO-8601
    "completed_at":   str
  },
  "milestones": [              # one entry each time progress increases
    { "step": int, "progress_before": float, "progress_after": float,
      "score": int, "action": str }
  ],
  "steps": [                   # one entry per game step
    { "step": int, "obs": str, "action": str, "reasoning": str,
      "progress": float, "score": int, "reward": int|float,
      "tokens": int,          # main generation/intervention tokens + all judge/replan tokens this step
      "judge_tokens": int,    # breakdown: portion of "tokens" spent on coherence/integration/replan judges
      "goal": str|None, "subgoal": str|None,
      "belief": {"type": str, "item": str, "location": str}|None,
      "confidence": float|None,
      "mechanisms_used": [str, ...] }   # e.g. ["m1_info_seeking"], ["m6_integration_gap", "m6_planning"]
  ],
  "drift_events": [              # omitted if empty; one entry each time check_drift() flags
    { "step": int, "off_path_rate": float, "steps_since_on_path": int,
      "window": int, "threshold": float, "recent_actions": [...] }
  ],
  "ale_events": [               # omitted if empty; one entry each time check_ale() confirms
    { "step": int, "belief": [...], "contradiction": bool, "evidence": {...} }
  ],
  "budget_events": [            # omitted if empty; one entry each time check_budget() flags
    { "step": int, "budget_risk": float, "step_ratio": float, "token_ratio": float,
      "stall_ratio": float, "cumulative_tokens": int, "tokens_since_progress": int, "threshold": float }
  ],
  "integration_events": [       # omitted if empty; one entry each time the integration judge confirms a gap
    { "step": int, "unused": [...], "gap": bool, "suggested_combination": str|None }
  ],
  "truncation_events": [        # omitted if empty; one entry per rolling-window truncation
    { "step": int, "event": str, "kept_pairs": int }   # or per overflow-recovery eviction round
  ]
}
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Stop-reason taxonomy
# ---------------------------------------------------------------------------
# Every run records exactly one of these. Without it a run that died on an API
# failure at step 100 is indistinguishable in the aggregates from one that
# honestly exhausted its step budget — which is how "Max Steps 100.0" came to be
# reported for every configuration in Figure 3. Names follow dig-bench
# (core/harness.py) where they map; the three dig-bench names this repo renames
# are noted inline.
STOP_REASONS = (
    'solved',             # game_over with final_progress >= 100 (dig-bench: "done")
    'game_over',          # game_over below 100% — death/quit, a terminal episode that is not a win
    'step_cap',           # ran out of the configured step budget (dig-bench: "max_steps")
    'cost_cap',           # hit experiments.max_total_tokens
    'api_failure',        # provider call failed and was not a recoverable overflow
    'context_exhausted',  # context overflow that eviction could no longer shrink
    'invalid_action_cap', # too many consecutive unusable actions (dig-bench: "blocked")
    'env_failure',        # env.step() raised (dig-bench: "bench_failure")
    'interrupted',        # KeyboardInterrupt (Ctrl-C)
)


class RunLogger:
    """Accumulate records for one run and flush to a structured JSON file."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self._reset()

    def _reset(self):
        self._steps = []
        self._milestones = []
        self._errors = []
        self._ale_events = []
        self._drift_events = []
        self._budget_events = []
        self._integration_events = []
        self._truncation_events = []
        self._started_at = datetime.now(timezone.utc).isoformat()

    def add_step(self, record):
        """Record one game step. Expected keys: step, obs, action, reasoning, progress,
        score, reward, tokens, judge_tokens, goal, subgoal, belief, confidence, mechanisms_used."""
        self._steps.append(record)

    def add_milestone(self, record):
        """Record a progress increase. Expected keys: step, progress_before, progress_after, score, action."""
        self._milestones.append(record)

    def add_error(self, record):
        """Record an error. Expected keys: step, error."""
        self._errors.append(record)

    def add_ale(self, record):
        """Record a confirmed Accepted-Local-Error. Expected keys: step, belief, contradiction, evidence."""
        self._ale_events.append(record)

    def add_drift_event(self, record):
        """Record a confirmed drift detection. Expected keys: step, off_path_rate, steps_since_on_path, ..."""
        self._drift_events.append(record)

    def add_budget_event(self, record):
        """Record a confirmed budget/liveness risk. Expected keys: step, budget_risk, step_ratio, ..."""
        self._budget_events.append(record)

    def add_integration_event(self, record):
        """Record a confirmed integration gap. Expected keys: step, unused, gap, suggested_combination."""
        self._integration_events.append(record)

    def add_truncation_event(self, record):
        """Record a context truncation or overflow recovery. Expected keys: step, event."""
        self._truncation_events.append(record)

    def save(self, run_id, game_name, summary):
        """Write the transcript and reset for the next run.

        Args:
            run_id:    hex run identifier.
            game_name: e.g. 'zork1'.
            summary:   dict with keys total_steps, final_progress, final_score,
                       game_over, stop_reason, model.

        Raises:
            ValueError: if stop_reason or model is missing or unrecognised. This
                is deliberately a hard failure at the write boundary: it is the
                one place that can guarantee no transcript is ever produced
                without a recorded reason for stopping, or without the model id
                that produced it.

        Returns:
            Path to the written file.
        """
        stop_reason = (summary or {}).get('stop_reason')
        if stop_reason not in STOP_REASONS:
            raise ValueError(
                f'run {run_id}: stop_reason={stop_reason!r} is not one of {STOP_REASONS}. '
                'Every exit path out of the run loop must set exactly one.'
            )
        if not (summary or {}).get('model'):
            raise ValueError(
                f'run {run_id}: summary is missing the resolved model id. '
                'Pass the id returned by runner.resolve_model().'
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            'run_id': run_id,
            'game': game_name,
            'summary': {
                **summary,
                'started_at': self._started_at,
                'completed_at': datetime.now(timezone.utc).isoformat(),
            },
            'milestones': self._milestones,
            'steps': self._steps,
        }
        if self._errors:
            transcript['errors'] = self._errors
        if self._ale_events:
            transcript['ale_events'] = self._ale_events
        if self._drift_events:
            transcript['drift_events'] = self._drift_events
        if self._budget_events:
            transcript['budget_events'] = self._budget_events
        if self._integration_events:
            transcript['integration_events'] = self._integration_events
        if self._truncation_events:
            transcript['truncation_events'] = self._truncation_events

        path = self.out_dir / f'{run_id}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, default=str)

        self._reset()
        return path
