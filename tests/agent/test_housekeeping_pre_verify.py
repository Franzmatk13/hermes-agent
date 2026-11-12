"""With agent.pre_verify_always, an empty reply after a housekeeping round asks pre_verify
before the earlier line becomes the final answer (that exit used to skip the stop gates)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import turn_empty_response, verify_hooks
from agent.session_persistence import _EPHEMERAL_SCAFFOLDING_FLAGS

REPLY = "Saved the skill."


@pytest.fixture
def hook(monkeypatch):
    state = {"calls": [], "message": "keep going", "always": True, "max": 3}

    def fake_message(**kw):
        state["calls"].append(kw)
        return state["message"]

    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_verify")
    monkeypatch.setattr(plugins, "get_pre_verify_continue_message", fake_message)
    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda config=None: state["always"])
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda config=None: state["max"])
    return state


def _agent():
    a = SimpleNamespace(
        _current_streamed_assistant_text="", _last_content_with_tools=REPLY,
        _last_content_tools_all_housekeeping=True, _hold_stream=False, _empty_content_retries=0,
        _pre_verify_nudges=0, _turn_file_mutation_paths=set(), session_id="s1", platform="cli",
        model="m", _resolved_is_coding=False, status=[],
    )
    a._has_content_after_think_block = lambda text: bool((text or "").strip())
    a._strip_think_blocks = lambda text: text or ""
    a._build_assistant_message = lambda msg, finish: {"role": "assistant", "content": "", "finish_reason": finish}
    a._emit_diagnostic_status = a.status.append
    a._buffer_diagnostic_status = a.status.append
    return a


def _history():
    return [
        {"role": "user", "content": "make a skill"},
        {"role": "assistant", "content": REPLY, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "saved"},
    ]


def _recover(agent, messages):
    return turn_empty_response.recover_empty_response(
        agent, SimpleNamespace(), None, "stop", final_response="", messages=messages,
        api_messages=[], conversation_history=[], active_system_prompt="sys", api_call_count=1,
        turn_exit_reason=None, preflight_compression_blocked=False,
    )


def test_option_off_keeps_the_upstream_fallback(hook):
    hook["always"] = False
    agent, messages = _agent(), _history()
    v = _recover(agent, messages)
    assert (v.action, v.final_response, v.turn_exit_reason) == ("break", REPLY, "fallback_prior_turn_content")
    assert hook["calls"] == [] and messages == _history()


def test_option_on_hook_continues_the_turn(hook):
    agent, messages = _agent(), _history()
    v = _recover(agent, messages)
    assert v.action == "continue"
    assert hook["calls"][0]["final_response"] == REPLY and hook["calls"][0]["changed_paths"] == []
    empty_row, nudge_row = messages[-2:]
    assert empty_row["role"] == "assistant" and empty_row["content"] == "(empty)"
    nudge_row.pop("timestamp", None)  # stamped by append_message
    assert nudge_row == {"role": "user", "content": "keep going", "_pre_verify_synthetic": True,
                         "_housekeeping_pre_verify": True}
    # Both rows are ephemeral scaffolding: never persisted to the session store.
    assert any(empty_row.get(f) for f in _EPHEMERAL_SCAFFOLDING_FLAGS)
    assert any(nudge_row.get(f) for f in _EPHEMERAL_SCAFFOLDING_FLAGS)
    # The reply line stays available as the fallback answer.
    assert agent._pre_verify_nudges == 1 and agent._last_content_with_tools == REPLY


def test_option_on_hook_silent_ends_on_the_reply(hook):
    hook["message"] = None
    agent, messages = _agent(), _history()
    v = _recover(agent, messages)
    assert (v.action, v.final_response) == ("break", REPLY)
    assert len(hook["calls"]) == 1 and messages == _history()


def test_budget_exhausted_ends_on_the_reply_and_drops_nudges(hook):
    hook["max"] = 1
    agent, messages = _agent(), _history()
    assert _recover(agent, messages).action == "continue"
    assert len(messages) == 5
    v = _recover(agent, messages)  # the model went quiet again; budget of 1 is spent
    assert (v.action, v.final_response) == ("break", REPLY)
    assert len(hook["calls"]) == 1 and messages == _history()
    assert agent._last_content_with_tools is None


def test_no_housekeeping_line_never_asks(hook):
    agent, messages = _agent(), _history()
    agent._last_content_tools_all_housekeeping = False
    agent._post_tool_empty_retried = False
    v = _recover(agent, messages)
    assert v.action == "continue" and hook["calls"] == []  # the upstream post-tool nudge
    assert messages[-1].get("_empty_recovery_synthetic") and not messages[-1].get("_pre_verify_synthetic")
