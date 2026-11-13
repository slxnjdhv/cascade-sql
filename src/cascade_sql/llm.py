"""Model backends.

* OpenAICompatible: any server that speaks the OpenAI /chat/completions API:
  - `vllm serve <model>` on a GPU box or Colab (recommended for the real runs: supports n>1 + logprobs)
  - Ollama (`http://localhost:11434/v1`) for a laptop pilot with a small model
  - cloud providers that expose an OpenAI-compatible endpoint (escalation target A)
  The API key is read from an environment variable named in the config; never put keys in files.
* Mock: deterministic fake model for tests and the demo.
"""
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests


@dataclass
class Completion:
    text: str
    sql_logprob_mean: Optional[float] = None   # mean log-prob of tokens inside the SQL span
    sql_n_tokens: Optional[int] = None
    output_tokens: int = 0
    prompt_tokens: int = 0
    latency_s: float = 0.0
    # per generated token: {"token": str, "logprob": float, "top": {alt_token: prob}} (top-k alternatives;
    # needed for SAC). Filled by backends that expose full distributions (MLX) or top_logprobs (OpenAI).
    token_data: Optional[List[dict]] = None


class Backend:
    name = "base"

    def generate(self, messages: List[Dict[str, str]], n: int = 1, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: Optional[int] = None, meta: Optional[dict] = None
                 ) -> List[Completion]:
        raise NotImplementedError


# ---------------------------------------------------------------- OpenAI-compatible
_SQL_FENCE = re.compile(r"```(?:sqlite|sql)\b\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


def _sql_span(text: str):
    """Character span of the final code block (```sql preferred, else any ``` block,
    which is what OmniSQL emits)."""
    for rx in (_SQL_FENCE, _ANY_FENCE):
        m = list(rx.finditer(text))
        if m:
            return m[-1].start(1), m[-1].end(1)
    return None


def sql_logprob_from_tokens(tokens: List[dict], text: str):
    """Mean log-prob of the tokens that overlap the final ```sql block."""
    span = _sql_span(text)
    if not tokens:
        return None, None
    lps, pos = [], 0
    for t in tokens:
        tok, lp = t.get("token", ""), t.get("logprob")
        start, end = pos, pos + len(tok)
        pos = end
        if lp is None:
            continue
        if span is None or (end > span[0] and start < span[1]):
            lps.append(lp)
    if not lps:
        return None, None
    return sum(lps) / len(lps), len(lps)


class OpenAICompatible(Backend):
    def __init__(self, base_url: str, model: str, api_key_env: Optional[str] = None,
                 logprobs: bool = True, supports_n: bool = True, timeout_s: float = 300,
                 max_retries: int = 5, extra_body: Optional[dict] = None):
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not self.key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self.logprobs = logprobs
        self.supports_n = supports_n
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout_s)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def _one_call(self, messages, n, temperature, max_tokens, seed):
        body = {"model": self.model, "messages": messages, "n": n,
                "temperature": temperature, "max_tokens": max_tokens, **self.extra_body}
        if seed is not None:
            body["seed"] = seed
        if self.logprobs:
            body["logprobs"] = True
        t0 = time.monotonic()
        data = self._post(body)
        lat = time.monotonic() - t0
        usage = data.get("usage") or {}
        out = []
        for ch in data.get("choices", []):
            text = (ch.get("message") or {}).get("content") or ""
            toks = ((ch.get("logprobs") or {}).get("content")) or []
            lp, nt = sql_logprob_from_tokens(toks, text)
            out.append(Completion(text=text, sql_logprob_mean=lp, sql_n_tokens=nt,
                                  output_tokens=len(toks) or (usage.get("completion_tokens", 0) // max(1, n)),
                                  prompt_tokens=usage.get("prompt_tokens", 0), latency_s=lat))
        return out

    def generate(self, messages, n=1, temperature=0.0, max_tokens=1024, seed=None, meta=None):
        if n == 1 or self.supports_n:
            return self._one_call(messages, n, temperature, max_tokens, seed)
        res = []
        for i in range(n):  # servers without `n` support (e.g. some Ollama versions)
            res += self._one_call(messages, 1, temperature, max_tokens,
                                  None if seed is None else seed + i)
        return res


# ---------------------------------------------------------------- Mock
class Mock(Backend):
    """Fake model driven by `meta` from the dataset (demo/tests only).

    For each question, meta must contain `gold_sql` and `wrong_sqls` (list) and
    `difficulty` in [0, 1]. A sample is correct with probability
    skill * (1 - difficulty); confidence-like log-probs are higher for correct samples,
    with noise, so the gate has a real but imperfect signal.
    """

    def __init__(self, name: str = "mock", skill: float = 0.8, seed: int = 0):
        self.name = name
        self.skill = skill
        self.rng = random.Random(seed)

    def generate(self, messages, n=1, temperature=0.0, max_tokens=1024, seed=None, meta=None):
        meta = meta or {}
        is_repair = "failed" in messages[-1]["content"]  # same test as prompts.is_repair_prompt
        rng = random.Random(f"{self.name}|{meta.get('qid')}|{seed}|{temperature}|{is_repair}")
        p = max(0.02, min(0.98, self.skill * (1.0 - meta.get("difficulty", 0.3))))
        if temperature == 0.0:  # greedy: deterministic choice per question
            p = 1.0 if p >= 0.5 else 0.0
            p = p if rng.random() > 0.1 else 1 - p
        outs = []
        for _ in range(n):
            if is_repair:
                sql = meta["gold_sql"] if rng.random() < p else rng.choice(meta["wrong_sqls"])
            elif rng.random() < p:
                sql = meta["gold_sql"]
            elif rng.random() < 0.15:
                sql = "SELECT nonexistent_col FROM " + meta.get("some_table", "t")  # will error
            else:
                sql = rng.choice(meta["wrong_sqls"])
            correct = sql == meta["gold_sql"]
            lp = (-0.15 if correct else -0.45) + rng.gauss(0, 0.15)
            text = f"Reasoning omitted.\n```sql\n{sql}\n```"
            outs.append(Completion(text=text, sql_logprob_mean=lp, sql_n_tokens=max(1, len(sql) // 4),
                                   output_tokens=len(text) // 4, prompt_tokens=800,
                                   latency_s=0.05 if "large" not in self.name else 0.6))
        return outs


# ---------------------------------------------------------------- MLX (Apple Silicon, in-process)
class MLX(Backend):
    """Runs an MLX model in-process on Apple Silicon (M1/M2/M3...).

    Speed trick: the prompt is pre-filled ONCE into a KV cache; every sample (greedy and
    sampled) rewinds the cache to the end of the prompt and only generates. With BIRD
    prompts of ~3k tokens this removes most of the per-sample cost. The cache is reused
    across calls with the identical prompt (greedy call, then sampling call).
    """

    def __init__(self, model: str, prefill_step: int = 512, top_k: int = 10):
        from mlx_lm import load  # imported lazily: only needed on Macs
        self.name = model
        self.model, self.tok = load(model)
        self.prefill_step = prefill_step
        self.top_k = top_k
        self._dec = {}
        # stop on every special end marker (e.g. Qwen's <|im_end|> and <|endoftext|>); some
        # tokenizer configs list only one of them as EOS, and the other then leaks into the text
        self.stop_ids = set(getattr(self.tok, "eos_token_ids", None) or [])
        for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            try:
                tid = self.tok.convert_tokens_to_ids(name)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(self.tok, "unk_token_id", None):
                    self.stop_ids.add(tid)
            except Exception:
                pass
        self._key = None
        self._cache = None
        self._offset = 0
        self._last = None

    def _decode1(self, tid: int) -> str:
        s = self._dec.get(tid)
        if s is None:
            s = self.tok.decode([tid])
            self._dec[tid] = s
        return s

    def _prefill(self, tokens: List[int]) -> float:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.cache import trim_prompt_cache
        key = tuple(tokens)
        if key == self._key:
            return 0.0
        t0 = time.monotonic()
        pre = tokens[:-1]
        # reuse the longest common prefix with the cached prompt (e.g. a repair prompt that
        # extends the original prompt): rewind the cache to it and only pre-fill the rest
        start = 0
        if self._key is not None and self._cache is not None:
            old = self._key[:-1]
            n = min(len(old), len(pre))
            while start < n and old[start] == pre[start]:
                start += 1
            if start >= 64:
                trim_prompt_cache(self._cache, self._cache[0].offset - start)
            else:
                start = 0
        if start == 0:
            self._cache = make_prompt_cache(self.model)
        for i in range(start, len(pre), self.prefill_step):
            self.model(mx.array(pre[i:i + self.prefill_step])[None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
        self._offset = self._cache[0].offset
        self.prefix_reused = start
        self._last = tokens[-1]
        self._key = key
        return time.monotonic() - t0

    def generate(self, messages, n=1, temperature=0.0, max_tokens=1024, seed=None, meta=None):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokens = self.tok.encode(prompt, add_special_tokens=False) if hasattr(self.tok, "encode") else prompt
        prefill_s = self._prefill(list(tokens))
        sampler = make_sampler(temp=temperature)
        outs = []
        for i in range(n):
            if seed is not None:
                mx.random.seed(int(seed) * 1000 + i)
            trim_prompt_cache(self._cache, self._cache[0].offset - self._offset)
            t0 = time.monotonic()
            text, toks = "", []
            for r in stream_generate(self.model, self.tok, mx.array([self._last]), max_tokens=max_tokens,
                                     sampler=sampler, prompt_cache=self._cache):
                if int(r.token) in self.stop_ids:
                    break
                text += r.text
                tok_str = self._decode1(int(r.token))
                entry = {"token": tok_str, "logprob": float(r.logprobs[r.token].item())}
                if self.top_k:
                    lpv = r.logprobs
                    idx = mx.argpartition(-lpv, kth=self.top_k - 1)[: self.top_k]
                    vals = mx.exp(lpv[idx])
                    entry["top"] = {self._decode1(int(i)): float(v) for i, v in zip(idx.tolist(), vals.tolist())}
                toks.append(entry)
            lp, nt = sql_logprob_from_tokens(toks, text)
            outs.append(Completion(text=text, sql_logprob_mean=lp, sql_n_tokens=nt, output_tokens=len(toks),
                                   token_data=toks,
                                   # prompt tokens actually processed (0 when the cached prefill was reused)
                                   prompt_tokens=len(tokens) if (i == 0 and prefill_s > 0) else 0,
                                   latency_s=(time.monotonic() - t0) + (prefill_s if i == 0 else 0.0)))
        return outs


def make_backend(cfg: dict) -> Backend:
    kind = cfg.get("kind", "openai")
    if kind == "mlx":
        return MLX(model=cfg["model"], prefill_step=cfg.get("prefill_step", 512), top_k=cfg.get("top_k", 10))
    if kind == "mock":
        return Mock(name=cfg.get("name", "mock"), skill=cfg.get("skill", 0.8), seed=cfg.get("seed", 0))
    if kind == "openai":
        return OpenAICompatible(base_url=cfg["base_url"], model=cfg["model"],
                                api_key_env=cfg.get("api_key_env"), logprobs=cfg.get("logprobs", True),
                                supports_n=cfg.get("supports_n", True),
                                timeout_s=cfg.get("timeout_s", 300), extra_body=cfg.get("extra_body"))
    raise ValueError(f"unknown backend kind: {kind}")
