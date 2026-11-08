"""LLM client and response utilities.

OpenRouterClient wraps the OpenRouter chat-completions API with automatic
retries. extract_action_and_reasoning parses the JSON {action, reasoning}
format that game agents are prompted to return.

Typical usage
-------------
from llm import OpenRouterClient, extract_action_and_reasoning

client = OpenRouterClient(api_key=..., model='google/gemma-4-26b-a4b-it')
resp   = client.generate('What is 2+2?', max_tokens=64)
parsed = client.parse_completion(resp)
action, reasoning = extract_action_and_reasoning(parsed['content'])

Error handling
--------------
Failures are raised as LLMError, which keeps the provider's response *body* in
str(exc). requests' own HTTPError stringifies to just the status line, and the
text the runner has to classify on — the context-overflow wording, with the
token counts it needs to size an eviction — only ever appears in the body.
is_context_overflow() / parse_overflow_tokens() do that classification; the
runner's overflow-recovery loop drives them.
"""

import json
import os
import re
import time

import requests


class LLMError(RuntimeError):
    """A failed provider call, carrying the HTTP status and the response body.

    ``status`` is None for network/timeout errors (no HTTP status was reached).
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.message = message
        self.status = status


def _is_retryable(exc):
    """Transient (retry) vs deterministic (fail fast).

    Mirrors dig-bench's core.clientutil.is_retryable: 5xx / 408 / 429 and
    status-less network errors are transient; every other 4xx is deterministic.
    Failing fast on 4xx matters for the context-overflow path — retrying a 400
    three times with backoff just delays the eviction that would fix it.
    """
    status = getattr(exc, 'status', None)
    if status is None:
        return True
    if status in (408, 429):
        return True
    if 400 <= status < 500:
        return False
    return True


# Context-window overflow markers — a 400 that EVICTING history can fix, unlike
# a generic fatal 4xx. Ported from dig-bench core/clientutil.py so the two
# harnesses classify the same provider wording identically.
_CONTEXT_OVERFLOW_MARKERS = (
    'context_length_exceeded', 'maximum context length', 'exceed context limit',
    'context window', 'reduce the length', 'longer than the maximum',
    'exceeds the maximum number of tokens allowed',
)


def is_context_overflow(err):
    """True if ``err`` (an exception or an error dict) is a context-window overflow.

    Recoverable by evicting history and retrying, unlike a generic fatal 4xx.
    """
    if not err:
        return False
    blob = (str(err.get('message', '')) if isinstance(err, dict) else str(err)).lower()
    return any(marker in blob for marker in _CONTEXT_OVERFLOW_MARKERS)


# Overflow wording usually carries the exact numbers — "maximum context length of
# 262144 tokens. You requested a total of 264084 tokens" (litellm/vLLM) and
# "maximum context length is 8193 tokens, however you requested 10001 tokens".
_OVERFLOW_LIMIT_RE = re.compile(r'maximum context length (?:is |of )?(\d+)')
_OVERFLOW_TOTAL_RE = re.compile(r'requested (?:a total of )?(\d+) tokens')


def parse_overflow_tokens(err):
    """``(requested_total, limit)`` from an overflow error, or None.

    None means the wording carried no usable counts, and the caller must fall
    back to blind eviction. Only a coherent pair (total > limit > 0) is ever
    returned, so a deficit computed from it is never negative.
    """
    if not err:
        return None
    blob = (str(err.get('message', '')) if isinstance(err, dict) else str(err)).lower()
    limit_m = _OVERFLOW_LIMIT_RE.search(blob)
    total_m = _OVERFLOW_TOTAL_RE.search(blob)
    if not (limit_m and total_m):
        return None
    limit, total = int(limit_m.group(1)), int(total_m.group(1))
    if not 0 < limit < total:
        return None
    return total, limit


def _normalize_base_url(base_url):
    base_url = (base_url or '').strip().rstrip('/')
    if not base_url:
        return 'https://openrouter.ai/api/v1'
    if base_url.endswith('/api'):
        return base_url + '/v1'
    return base_url


class OpenRouterClient:
    """HTTP client for the OpenRouter chat-completions endpoint."""

    def __init__(self, api_key=None, model=None, base_url=None, request_timeout=15):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not set. '
                'Export it in your shell or add it to config.yaml under openrouter.api_key.'
            )
        # No silent default. The model id is resolved once, by
        # runner.resolve_model(), and passed in explicitly; a fallback here
        # would be a fourth, invisible source of truth (it is how published
        # runs ended up on a model config.yaml never named).
        if not model:
            raise RuntimeError(
                'OpenRouterClient requires an explicit model id. '
                'Resolve it with runner.resolve_model(cfg) so the id is recorded '
                'in the run log and cross-checked against the environment.'
            )
        self.model = model
        self.request_timeout = request_timeout
        self.base_url = _normalize_base_url(
            base_url or os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1'
        )
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        })

    def generate(self, prompt, temperature=None, max_tokens=512, timeout=None, reasoning_enabled=False):
        """Send a prompt and return the raw API response dict.

        Args:
            prompt: str (single user message) or list of {role, content} dicts.
            temperature: left None on purpose — see the fairness note below.
            reasoning_enabled: pass True to enable chain-of-thought on supported models.

        Raises LLMError (status + provider body in the message) on failure.
        """
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
        }
        # FAIRNESS: no sampling parameters. temperature/top_p are never put in
        # the payload, so every model runs its own defaults. Forcing one value
        # across providers would be a different intervention per model (several
        # reasoning models fix their sampling and reject or ignore an override).
        # See dig-bench baseline-harness README, "Fairness". The kwarg exists so
        # a caller can deliberately opt in and have that show up in the diff —
        # no caller in this repo does, and none should without a note in the
        # paper's methods section.
        if temperature is not None:
            payload['temperature'] = temperature
        if reasoning_enabled:
            payload['reasoning'] = {'enabled': True}

        url = self.base_url.rstrip('/') + '/chat/completions'
        effective_timeout = timeout if timeout is not None else self.request_timeout
        last_error = None
        for attempt in range(3):
            try:
                r = self.session.post(url, json=payload, timeout=effective_timeout)
                if r.status_code >= 400:
                    # Keep the body: the overflow wording and its token counts live there.
                    raise LLMError(f'HTTP {r.status_code}: {r.text[:2000]}', status=r.status_code)
                body = r.json()
            except LLMError as exc:
                if not _is_retryable(exc):
                    raise
                last_error = exc
            except Exception as exc:
                last_error = LLMError(f'{type(exc).__name__}: {exc}')
            else:
                # OpenRouter also reports upstream failures as a 200 carrying an
                # `error` object, so a status check alone would let an overflow
                # through as an empty completion.
                error = body.get('error') if isinstance(body, dict) else None
                if not error:
                    return body
                code = error.get('code') if isinstance(error, dict) else None
                message = error.get('message') if isinstance(error, dict) else str(error)
                exc = LLMError(f'HTTP {code}: {message}',
                               status=code if isinstance(code, int) else None)
                if not _is_retryable(exc):
                    raise exc
                last_error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
        raise last_error

    def parse_completion(self, resp):
        """Extract content and reasoning_details from a raw API response.

        Returns a dict with keys: content (str), reasoning_details, tokens,
        prompt_tokens, raw. ``prompt_tokens`` is the provider's own
        server-reported prompt size and is the trigger for the runner's rolling
        -window truncation (0 when the provider did not report one).
        """
        if not resp:
            return {'content': '', 'reasoning_details': None, 'tokens': 0,
                    'prompt_tokens': 0, 'raw': resp}
        if isinstance(resp, dict):
            usage = resp.get('usage') or {}
            prompt_tokens = usage.get('prompt_tokens') or 0
            tokens = usage.get('total_tokens') or (
                prompt_tokens + (usage.get('completion_tokens') or 0)
            )
            choices = resp.get('choices') or []
            if choices:
                msg = choices[0].get('message') or {}
                return {
                    'content': msg.get('content') or choices[0].get('text') or '',
                    'reasoning_details': msg.get('reasoning_details'),
                    'tokens': tokens,
                    'prompt_tokens': prompt_tokens,
                    'raw': resp,
                }
            if 'output' in resp:
                return {'content': str(resp['output']), 'reasoning_details': None,
                        'tokens': tokens, 'prompt_tokens': prompt_tokens, 'raw': resp}
        return {'content': str(resp), 'reasoning_details': None, 'tokens': 0,
                'prompt_tokens': 0, 'raw': resp}


def extract_action_and_reasoning(content, default_action='look'):
    """Parse a model response into (action, reasoning).

    Expects a JSON object with keys "action" and "reasoning". Falls back to
    using the last non-empty line as the action if JSON parsing fails.
    """
    text = (content or '').strip()
    if not text:
        return default_action, ''

    # collect JSON candidates in order of preference
    candidates = []
    fenced = re.search(r'```json\s*(.*?)\s*```', text, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if text.startswith('{') and text.endswith('}'):
        candidates.append(text)
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            action = str(parsed.get('action', '')).strip()
            reasoning = str(parsed.get('reasoning', '')).strip()
            if action:
                return action, reasoning or text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1], text) if lines else (default_action, text)


_VALID_BELIEF_TYPES = ('have_item', 'item_in_loc', 'none')


def _coerce_belief(raw):
    """Normalize a self-reported belief dict; return None if unusable."""
    if not isinstance(raw, dict):
        return None
    btype = str(raw.get('type', 'none')).strip().lower()
    if btype not in _VALID_BELIEF_TYPES or btype == 'none':
        return None
    item = str(raw.get('item') or '').strip() or None
    location = str(raw.get('location') or '').strip() or None
    if btype == 'have_item' and not item:
        return None
    if btype == 'item_in_loc' and not location:
        return None
    return {'type': btype, 'item': item, 'location': location}


def extract_step_fields(content, default_action='look'):
    """Parse a model response into the full structured step schema.

    Superset of extract_action_and_reasoning: also pulls goal, subgoal,
    belief, and confidence when the model reports them. Any field the model
    omits or reports malformed is simply absent from the result (None /
    empty) rather than reconstructed via a text-parsing fallback — mechanisms
    that depend on a field just get no signal that step.

    Returns a dict with keys: action, reasoning, goal, subgoal, belief
    (None or {type, item, location}), confidence (float or None).
    """
    text = (content or '').strip()
    result = {
        'action': default_action,
        'reasoning': '',
        'goal': None,
        'subgoal': None,
        'belief': None,
        'confidence': None,
    }
    if not text:
        return result

    candidates = []
    fenced = re.search(r'```json\s*(.*?)\s*```', text, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    if text.startswith('{') and text.endswith('}'):
        candidates.append(text)
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        action = str(parsed.get('action', '')).strip()
        if not action:
            continue
        result['action'] = action
        result['reasoning'] = str(parsed.get('reasoning', '')).strip() or text
        result['goal'] = str(parsed.get('goal') or '').strip() or None
        result['subgoal'] = str(parsed.get('subgoal') or '').strip() or None
        result['belief'] = _coerce_belief(parsed.get('belief'))
        try:
            confidence = parsed.get('confidence')
            result['confidence'] = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            result['confidence'] = None
        return result

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result['action'] = lines[-1] if lines else default_action
    result['reasoning'] = text
    return result
