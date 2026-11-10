"""agent.pre_verify_always opens the pre_verify gate on turns that edited no files."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import turn_stop_gates, verify_hooks


def test_default_is_off():
    assert verify_hooks.pre_verify_always({}) is False
    assert verify_hooks.pre_verify_always({"agent": {}}) is False


def test_reads_truthy_config():
    assert verify_hooks.pre_verify_always({"agent": {"pre_verify_always": True}}) is True
    assert verify_hooks.pre_verify_always({"agent": {"pre_verify_always": "yes"}}) is True
    assert verify_hooks.pre_verify_always({"agent": {"pre_verify_always": False}}) is False


@pytest.fixture
def gate(monkeypatch):
    calls = []

    def fake_message(**kw):
        calls.append(kw)
        return "keep going"

    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_verify")
    monkeypatch.setattr(plugins, "get_pre_verify_continue_message", fake_message)
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda config=None: 3)
    return calls


def _agent(paths):
    return SimpleNamespace(_turn_file_mutation_paths=set(paths), session_id="s1", platform="cli",
                           model="m", _resolved_is_coding=False)


def test_no_edits_and_option_off_never_asks(gate, monkeypatch):
    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda config=None: False)
    assert turn_stop_gates._pre_verify_nudge(_agent([]), "done", 0) is None
    assert gate == []


def test_no_edits_and_option_on_asks_with_empty_paths(gate, monkeypatch):
    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda config=None: True)
    assert turn_stop_gates._pre_verify_nudge(_agent([]), "done", 0) == "keep going"
    assert gate[0]["changed_paths"] == []


def test_edits_still_ask_with_option_off(gate, monkeypatch):
    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda config=None: False)
    assert turn_stop_gates._pre_verify_nudge(_agent(["/x/a.py"]), "done", 0) == "keep going"
    assert gate[0]["changed_paths"] == ["/x/a.py"]


def test_budget_still_bounds_the_loop(gate, monkeypatch):
    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda config=None: True)
    assert turn_stop_gates._pre_verify_nudge(_agent([]), "done", 3) is None
