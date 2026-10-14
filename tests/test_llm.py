import os
from types import SimpleNamespace

import pytest

from mnkbench.agents import make_agent
from mnkbench.agents.base import Budget, Status
from mnkbench.agents.llm import (
    AnthropicClient,
    LLMAgent,
    LLMError,
    MockClient,
    build_prompt,
    parse_move,
)
from mnkbench.game import CONNECT_FOUR, TIC_TAC_TOE, GameConfig, State, get_game
from mnkbench.tournament import play_match


def test_prompt_contains_board_rules_and_legal_moves():
    s = get_game(TIC_TAC_TOE).new_state().play(4)
    p = build_prompt(s)
    assert "You are playing O" in p and "b2" in p
    assert "Legal moves: a1 b1 c1 a2 c2 a3 b3 c3" in p
    assert "3 of their stones in a row" in p
    g = build_prompt(get_game(CONNECT_FOUR).new_state())
    assert "Gravity applies" in g and "Legal moves: a6 b6 c6 d6 e6 f6 g6" in g
    e = build_prompt(get_game(GameConfig(5, 5, 4, exact_k=True)).new_state())
    assert "overlines" in e


def test_parse_move():
    s = State.from_string(TIC_TAC_TOE, "X../.O./...")
    assert parse_move("thinking...\nMOVE: c3", s) == (8, "")
    assert parse_move("MOVE: a1 ... no wait MOVE: **B3**", s)[0] == 7
    mv, why = parse_move("MOVE: a1", s)
    assert mv is None and "occupied" in why
    mv, why = parse_move("MOVE: d9", s)
    assert mv is None and "not a cell" in why
    mv, why = parse_move("I like corners", s)
    assert mv is None and "MOVE" in why
    c4 = get_game(CONNECT_FOUR).new_state()
    mv, why = parse_move("MOVE: a1", c4)
    assert mv is None and "gravity" in why


def test_retry_then_legal_move():
    a = LLMAgent(provider="mock", behavior="illegal_first", seed=0)
    s = get_game(TIC_TAC_TOE).new_state()
    mv = a.select_move(s)
    assert s.is_legal(mv)
    rec = a.log[-1]
    assert rec["calls"] == 2 and rec["retries"] == 1 and not rec["fallback"]
    assert rec["input_tokens"] > 0 and rec["cost_usd"] > 0


@pytest.mark.parametrize("behavior", ["garbage", "error", "refuse"])
def test_fallback_after_retries(behavior):
    a = LLMAgent(provider="mock", behavior=behavior, max_retries=2, seed=0)
    s = get_game(TIC_TAC_TOE).new_state()
    mv = a.select_move(s)
    assert s.is_legal(mv)
    assert a.log[-1]["fallback"]
    assert a.summary()["fallback_rate"] == 1.0


def test_full_games_with_mock_and_cost_tracking():
    a = make_agent("llm:provider=mock")
    r = play_match(TIC_TAC_TOE, a, make_agent("random"), 4, seed=1)
    assert r.n_games == 4 and r.aborted == 0
    s = a.summary()
    assert s["moves"] > 0 and s["cost_per_move_usd"] > 0
    assert s["fallback_rate"] == 0


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0),
            model="claude-sonnet-5-5", stop_reason=self.stop_reason)


def test_anthropic_client_request_and_usage():
    fake_messages = FakeMessages("MOVE: b2")
    fake = SimpleNamespace(beta=SimpleNamespace(messages=fake_messages), messages=fake_messages)
    client = AnthropicClient(client=fake, effort="low")
    agent = LLMAgent(client=client, seed=0)
    s = get_game(TIC_TAC_TOE).new_state()
    assert agent.select_move(s) == 4
    call = fake_messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"][0]["role"] == "user"
    rec = agent.log[-1]
    assert rec["input_tokens"] == 100 and rec["output_tokens"] == 50
    assert rec["cost_usd"] == pytest.approx((100 * 2.0 + 50 * 10.0) / 1e6)


def test_anthropic_refusal_and_errors_trigger_retry():
    fake_messages = FakeMessages("", stop_reason="refusal")
    fake = SimpleNamespace(beta=SimpleNamespace(messages=fake_messages), messages=fake_messages)
    agent = LLMAgent(client=AnthropicClient(client=fake), max_retries=1, seed=0)
    s = get_game(TIC_TAC_TOE).new_state()
    assert s.is_legal(agent.select_move(s))
    assert agent.log[-1]["fallback"] and len(fake_messages.calls) == 2

    class Boom:
        def create(self, **kw):
            raise RuntimeError("network down")

    boom = SimpleNamespace(beta=SimpleNamespace(messages=Boom()), messages=Boom())
    c = AnthropicClient(client=boom)
    with pytest.raises(LLMError):
        c.complete("sys", [{"role": "user", "content": "hi"}])


def test_feasibility_without_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    rep = make_agent("llm").feasibility(TIC_TAC_TOE, Budget())
    assert rep.status == Status.UNAVAILABLE
    rep = make_agent("llm:provider=mock").feasibility(TIC_TAC_TOE, Budget())
    assert rep.status == Status.OK
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    rep = make_agent("llm").feasibility(TIC_TAC_TOE, Budget(time_per_move=1.0))
    assert rep.status == Status.REFUSED and "time/move" in rep.reason
    rep = make_agent("llm").feasibility(TIC_TAC_TOE, Budget(time_per_move=60.0))
    assert rep.status == Status.OK and rep.estimate.cost_per_move > 0


def test_llm_experiment_mock(tmp_path, monkeypatch):
    from mnkbench.experiments.llm_exp import exp_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    d = exp_llm(tmp_path, quick=True, log=lambda s: None)
    assert not d["meta"]["live"] and "mock" in d["note"]
    assert d["rows"][0]["cost_per_move_usd"] > 0


def test_unknown_provider():
    with pytest.raises(ValueError):
        LLMAgent(provider="nope").select_move(get_game(TIC_TAC_TOE).new_state())


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_claude_plays_a_legal_move():
    agent = make_agent("llm", seed=0)
    s = State.from_string(TIC_TAC_TOE, "XX./OO./...")
    mv = agent.select_move(s)
    assert s.is_legal(mv)
    assert agent.log[-1]["input_tokens"] > 0


def test_mock_client_parses_legal_list():
    c = MockClient(behavior="first")
    r = c.complete("s", [{"role": "user", "content": "Legal moves: b2 c3\nEnd with MOVE"}])
    assert r.text.endswith("MOVE: b2")
