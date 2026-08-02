"""Tests for graphify.llm._label_batch_with_retry — adaptive split-and-retry
on JSON parse failure during community labeling (#1278).
"""
from __future__ import annotations

import json
import re

from graphify import llm as llm_mod


def test_label_batch_recovers_via_split_on_invalid_json(monkeypatch):
    """Demonstrates the bug fix.

    The full batch of 4 communities triggers malformed JSON from the LLM.
    The helper splits in half (2+2) and retries each half. Both sub-batches
    succeed. Every community ends up labeled — none silently dropped.
    """
    batch_cids = [42, 99, 137, 201]
    batch_lines = [
        "Community 42: validate_token, get_session",
        "Community 99: create_order, add_to_cart",
        "Community 137: build_graph, cluster_nodes",
        "Community 201: render_route, handle_request",
    ]
    call_count = {"n": 0}

    def fake_call_llm(prompt: str, **_kwargs) -> str:
        """First call (4 communities): returns broken JSON to trigger retry.
        Subsequent calls (<=2 communities): return a clean JSON object
        labeling whatever community IDs appear in the prompt.
        """
        call_count["n"] += 1
        cids_in_prompt = [int(m) for m in re.findall(r"Community (\d+):", prompt)]
        if call_count["n"] == 1:
            return "{this is not valid json, missing quotes"
        return json.dumps({str(cid): f"Label {cid}" for cid in cids_in_prompt})

    monkeypatch.setattr(llm_mod, "_call_llm", fake_call_llm)

    result = llm_mod._label_batch_with_retry(
        batch_cids, batch_lines, backend="gemini", model=None,
    )

    assert result == {42: "Label 42", 99: "Label 99", 137: "Label 137", 201: "Label 201"}
    assert call_count["n"] >= 2


def _capture_label_budget(monkeypatch, *, backend, model, cids):
    """Run one labeling call and return the max_tokens it budgeted."""
    monkeypatch.delenv("GRAPHIFY_MAX_OUTPUT_TOKENS", raising=False)
    seen: dict = {}

    def fake_call_llm(prompt: str, **kwargs) -> str:
        seen["max_tokens"] = kwargs.get("max_tokens")
        ids = [int(m) for m in re.findall(r"Community (\d+):", prompt)]
        return json.dumps({str(c): f"Label {c}" for c in ids})

    monkeypatch.setattr(llm_mod, "_call_llm", fake_call_llm)
    lines = [f"Community {c}: sym_a, sym_b" for c in cids]
    llm_mod._label_batch_with_retry(cids, lines, backend=backend, model=model)
    return seen["max_tokens"]


def test_reasoning_model_gets_generous_label_budget(monkeypatch):
    # gpt-5 / o-series spend completion tokens on hidden reasoning first, so the
    # tight 256+48*n budget starved them and the pass degraded to "Community N".
    # They now get the generous 16384 budget (mirrors extract).
    assert _capture_label_budget(
        monkeypatch, backend="openai", model="gpt-5.6-terra", cids=[1]
    ) == 16384


def test_classic_model_keeps_the_tight_label_budget(monkeypatch):
    # A non-reasoning model keeps the cheap, tuned formula.
    assert _capture_label_budget(
        monkeypatch, backend="gemini", model="gemini-2.5-flash", cids=[1, 2, 3]
    ) == 256 + 48 * 3
