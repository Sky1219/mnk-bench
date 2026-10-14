"""LLM players: a language model picks the move from a text rendering of the board.

The agent sends the rules, the board (with coordinates) and the list of
legal moves, and asks for an answer of the form ``MOVE: b3``. Replies are
checked against the legal moves; a bad reply (unparseable, illegal, refused,
or an API error) is answered with feedback and retried up to ``max_retries``
times, after which a fallback agent (random by default) makes the move so the
game can always finish. Every call's tokens, cost and latency are tracked,
which gives the *cost per move* metric.

Providers are pluggable: anything with a ``complete(system, messages)``
method returning an :class:`LLMResponse` works. Built in:

* ``anthropic`` (default): Claude via the Anthropic SDK, model
  ``claude-sonnet-5-5`` unless ``model=`` is given. The key is read from the
  ``ANTHROPIC_API_KEY`` environment variable by the SDK; it is never stored
  or logged here. Server-side refusal fallback is enabled by default.
* ``openai``: any OpenAI-compatible chat API via the ``openai`` SDK
  (``OPENAI_API_KEY``).
* ``mock``: an offline fake used by tests and CI, with configurable
  behaviour (random/first/illegal-first/garbage) and token accounting.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..game import GameConfig, State
from .base import Agent, Budget, CostEstimate, FeasibilityReport, Status

DEFAULT_MODEL = "claude-sonnet-5-5"

# USD per million tokens (input, output). Used for the cost-per-move metric.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
    "mock": (2.0, 10.0),
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    refused: bool = False


class LLMError(RuntimeError):
    """A provider call failed (network, auth, rate limit...)."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict[str, str]]) -> LLMResponse: ...


# ---------------------------------------------------------------- providers
class AnthropicClient:
    """Claude through the official Anthropic SDK."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 16000, effort: str | None = None,
                 fallbacks: bool = True, timeout: float = 120.0, client: Any = None):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("the anthropic provider needs: pip install 'mnk-bench[llm]'") from exc
            # Credentials come from the environment (ANTHROPIC_API_KEY or an
            # `ant auth login` profile); nothing is hard-coded.
            client = anthropic.Anthropic(timeout=timeout)
        self.client = client

    def complete(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                # Server-side fallback: a request declined by the safety
                # classifiers is retried on a recommended fallback model.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                response = self.client.messages.create(**kwargs)
        except Exception as exc:  # anthropic.APIError and friends
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0)
            + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            + int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            model=getattr(response, "model", self.model),
            refused=getattr(response, "stop_reason", None) == "refusal",
        )


class OpenAIClient:
    """Any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 4000, base_url: str | None = None,
                 client: Any = None):
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("the openai provider needs: pip install 'mnk-bench[openai]'") from exc
            client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        self.client = client

    def complete(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system}, *messages])
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        usage = getattr(r, "usage", None)
        return LLMResponse(
            text=r.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=getattr(r, "model", self.model),
        )


class MockClient:
    """Offline stand-in for an LLM.

    ``behavior``: ``"random"`` (a random legal move), ``"first"`` (first legal
    move), ``"illegal_first"`` (an occupied/illegal cell, then legal moves on
    retry), ``"garbage"`` (never a parseable move), ``"error"`` (raises
    LLMError), ``"refuse"`` (a refusal). Token counts are estimated as
    characters / 4, so cost accounting can be tested without network access.
    """

    def __init__(self, behavior: str = "random", seed: int = 0, model: str = "mock",
                 output_tokens: int = 40):
        self.behavior = behavior
        self.rng = random.Random(seed)
        self.model = model
        self.out_tokens = output_tokens
        self.calls = 0
        self.last_prompt = ""

    def complete(self, system: str, messages: list[dict[str, str]]) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1]["content"]
        self.last_prompt = prompt
        tokens_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4
        legal = re.findall(r"\b([a-z]\d+)\b", prompt.split("Legal moves:")[-1]) if "Legal moves:" in prompt else []
        if self.behavior == "error":
            raise LLMError("mock failure")
        if self.behavior == "refuse":
            return LLMResponse("", tokens_in, 0, self.model, refused=True)
        if self.behavior == "garbage" or not legal:
            return LLMResponse("I think the centre is nice.", tokens_in, self.out_tokens, self.model)
        if self.behavior == "illegal_first" and len(messages) == 1:
            return LLMResponse("MOVE: z99", tokens_in, self.out_tokens, self.model)
        move = legal[0] if self.behavior == "first" else self.rng.choice(legal)
        return LLMResponse(f"I'll take {move}.\nMOVE: {move}", tokens_in, self.out_tokens, self.model)


PROVIDERS = {"anthropic": AnthropicClient, "openai": OpenAIClient, "mock": MockClient}


def make_client(provider: str, model: str | None = None, **kwargs: Any) -> LLMClient:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    cls = PROVIDERS[provider]
    if model:
        kwargs["model"] = model
    return cls(**kwargs)


# ------------------------------------------------------------------ prompts
SYSTEM_PROMPT = (
    "You are an expert player of m,n,k games (the family that includes tic-tac-toe, gomoku and, with "
    "gravity, Connect Four). Think about immediate wins, forced blocks and double threats. "
    "End your reply with a single line of the form `MOVE: <cell>`, for example `MOVE: b3`."
)

MOVE_RE = re.compile(r"MOVE:\s*\**\s*([a-zA-Z])\s*(\d{1,2})\b", re.IGNORECASE)


def rules_text(config: GameConfig) -> str:
    parts = [
        f"The board has {config.m} rows (numbered 1-{config.m} from the top) and {config.n} columns "
        f"(lettered a-{chr(ord('a') + config.n - 1)} from the left). A cell is named by column letter then "
        f"row number, so b3 is column b, row 3.",
        f"Players alternate placing one stone on an empty cell; X moves first. The first player to get "
        f"{config.k} of their stones in a row horizontally, vertically or diagonally wins. If the board "
        f"fills up first, the game is a draw.",
    ]
    if config.gravity:
        parts.append("Gravity applies: a stone always drops to the lowest empty cell of its column, so only "
                     "the lowest empty cell in each column is a legal move.")
    if config.exact_k:
        parts.append(f"Only exactly {config.k} in a row wins; longer lines (overlines) do not count.")
    return " ".join(parts)


def build_prompt(state: State) -> str:
    game = state.game
    me = "XO"[state.to_move - 1]
    legal = " ".join(game.cell_name(m) for m in state.legal_moves())
    last = f"Your opponent's last move was {game.cell_name(state.last_move)}. " if state.history else ""
    return (
        f"{rules_text(game.config)}\n\n"
        f"You are playing {me}. {last}It is your move (move {state.move_count + 1}).\n\n"
        f"Current board ('.' is empty):\n```\n{state.render()}\n```\n\n"
        f"Legal moves: {legal}\n\n"
        f"Choose the best move for {me}. End with `MOVE: <cell>`."
    )


def parse_move(text: str, state: State) -> tuple[int | None, str]:
    """Return (move, problem). The last ``MOVE:`` line wins."""
    game = state.game
    matches = MOVE_RE.findall(text or "")
    if not matches:
        return None, "I could not find a line of the form `MOVE: <cell>` in your reply."
    col, row = matches[-1]
    name = f"{col.lower()}{int(row)}"
    try:
        mv = game.parse_cell(name)
    except ValueError:
        return None, f"{name} is not a cell on this {game.m}x{game.n} board."
    if not state.is_legal(mv):
        if state.cells[mv] != 0:
            return None, f"{name} is already occupied."
        return None, f"{name} is not playable (under gravity only the lowest empty cell of a column is)."
    return mv, ""


# -------------------------------------------------------------------- agent
class LLMAgent(Agent):
    name = "llm"

    def __init__(self, seed: int | None = None, provider: str = "anthropic", model: str | None = None,
                 max_retries: int = 2, fallback: str = "random", effort: str | None = None,
                 client: LLMClient | None = None, price_in: float | None = None,
                 price_out: float | None = None, **client_kwargs: Any):
        super().__init__(seed)
        self.provider = provider
        self.model = model or {"anthropic": DEFAULT_MODEL, "mock": "mock"}.get(provider)
        self.max_retries = max_retries
        self.fallback_spec = fallback
        self.effort = effort
        self.client_kwargs = client_kwargs
        self._client = client
        self.price_in = price_in
        self.price_out = price_out
        self._fallback: Agent | None = None
        self.log: list[dict] = []  # per-move records (tokens, cost, retries, fallback)

    def params(self) -> dict:
        return {"provider": self.provider, "model": self.model, "max_retries": self.max_retries,
                "fallback": self.fallback_spec, "effort": self.effort}

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            kwargs = dict(self.client_kwargs)
            if self.provider == "anthropic" and self.effort:
                kwargs["effort"] = self.effort
            if self.provider == "mock":
                kwargs.setdefault("seed", self.rng.randrange(2**31))
            self._client = make_client(self.provider, self.model, **kwargs)
        return self._client

    def prices(self) -> tuple[float, float] | None:
        if self.price_in is not None and self.price_out is not None:
            return self.price_in, self.price_out
        model = getattr(self.client, "model", self.model) if self._client else self.model
        return PRICES.get(model or "")

    def cost(self, tokens_in: int, tokens_out: int) -> float | None:
        p = self.prices()
        if p is None:
            return None
        return (tokens_in * p[0] + tokens_out * p[1]) / 1e6

    def _fallback_agent(self) -> Agent:
        if self._fallback is None:
            from . import make_agent

            self._fallback = make_agent(self.fallback_spec, seed=self.rng.randrange(2**31))
        return self._fallback

    def select_move(self, state: State) -> int:
        messages = [{"role": "user", "content": build_prompt(state)}]
        rec = {"move_number": state.move_count + 1, "calls": 0, "input_tokens": 0, "output_tokens": 0,
               "retries": 0, "fallback": False, "errors": [], "latency": 0.0}
        move = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.complete(SYSTEM_PROMPT, messages)
            except LLMError as exc:
                rec["latency"] += time.perf_counter() - t0
                rec["errors"].append(str(exc)[:200])
                rec["retries"] += attempt > 0
                continue
            rec["latency"] += time.perf_counter() - t0
            rec["calls"] += 1
            rec["input_tokens"] += resp.input_tokens
            rec["output_tokens"] += resp.output_tokens
            if attempt:
                rec["retries"] += 1
            if resp.refused:
                rec["errors"].append("refusal")
                problem = "Please just choose a move."
            else:
                move, problem = parse_move(resp.text, state)
                if move is not None:
                    break
                rec["errors"].append(problem)
            messages = messages + [
                {"role": "assistant", "content": resp.text or "(no answer)"},
                {"role": "user", "content": f"{problem} Legal moves: "
                 + " ".join(state.game.cell_name(m) for m in state.legal_moves())
                 + "\nReply with `MOVE: <cell>` using one of the legal moves."},
            ]
        if move is None:
            rec["fallback"] = True
            move = self._fallback_agent().select_move(state)
        cost = self.cost(rec["input_tokens"], rec["output_tokens"])
        rec["cost_usd"] = cost
        self.log.append(rec)
        self.add_stat("moves", 1)
        self.add_stat("calls", rec["calls"])
        self.add_stat("input_tokens", rec["input_tokens"])
        self.add_stat("output_tokens", rec["output_tokens"])
        self.add_stat("retries", rec["retries"])
        self.add_stat("fallbacks", rec["fallback"])
        self.add_stat("cost_usd", cost or 0.0)
        return move

    def summary(self) -> dict:
        moves = self.stats.get("moves", 0) or 1
        return {
            "moves": int(self.stats.get("moves", 0)),
            "calls": int(self.stats.get("calls", 0)),
            "input_tokens_per_move": self.stats.get("input_tokens", 0) / moves,
            "output_tokens_per_move": self.stats.get("output_tokens", 0) / moves,
            "cost_per_move_usd": self.stats.get("cost_usd", 0) / moves,
            "retry_rate": self.stats.get("retries", 0) / moves,
            "fallback_rate": self.stats.get("fallbacks", 0) / moves,
        }

    # ------------------------------------------------------------ feasibility
    def _credentials(self) -> bool:
        if self._client is not None or self.provider == "mock":
            return True
        if self.provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if self.provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        return True

    def estimate(self, config: GameConfig) -> CostEstimate:
        legal_chars = 4 * config.size
        board_chars = 2 * config.size + 4 * config.m
        tokens_in = (len(SYSTEM_PROMPT) + 700 + legal_chars + board_chars) / 3.5
        tokens_out = 600.0  # reasoning + answer; varies a lot by model and effort
        retries = 1.2
        cost = self.cost(int(tokens_in * retries), int(tokens_out * retries)) or 0.0
        latency = 0.0 if self.provider == "mock" else 4.0 + tokens_out / 80.0
        return CostEstimate(time_per_move=latency, memory_bytes=2e6, cost_per_move=cost,
                            notes=f"~{tokens_in:.0f} input / ~{tokens_out:.0f} output tokens per move")

    def feasibility(self, config: GameConfig, budget: Budget) -> FeasibilityReport:
        if not self._credentials():
            return FeasibilityReport(self.name, config.label, Status.UNAVAILABLE, self.estimate(config),
                                     reason=f"no credentials for provider {self.provider!r}")
        return super().feasibility(config, budget)
