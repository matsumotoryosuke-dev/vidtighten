"""Tests for llm_correct.py — the local-LLM brand-noun correction backend.

The fixtures below are REAL outputs captured from gemma4:12b-mlx during the
feasibility testing round (see 04_Context/vidtighten-llm-correct.md), so the
parser/validator is proven against the model's actual failure modes — doubled
key-quotes, markdown fences, multi-block replies, backwards suggestions — not
against idealised JSON.
"""

import json
from unittest.mock import patch

import pytest

from preprod import llm_correct as L


# Real captured model replies (verbatim from testing) ─────────────────────────

# Seeded brand pass — correct direction, but doubled `""` key-quotes + leading spaces.
REAL_SEEDED = (
    '{"fixes":[{""wrong":"空気デザイン",""correct":"クウキデザイン"},'
    '{""wrong":" クロードコード",""correct":" Claude Code"},'
    '{""wrong":" クロドコード",""correct":" Claude Code"},'
    '{""wrong":" ノーション",""correct":" Notion"}]}'
)

# Open-ended glossary pass — BACKWARDS direction, markdown fence, doubled quotes.
REAL_BACKWARDS = (
    '```json\n{"glossary": [{""wrong": "クウキデザイン", "correct": "空気デザイン", '
    '"evidence": "hallucinated"}]}\n```'
)

# Grammar pass — multiple ```json blocks with prose between them (only first used).
REAL_MULTIBLOCK = (
    '```json\n{"fixes":[{""wrong":"あるある","correct":"ある"}]}\n```\n\n'
    '*Self-correction: on reflection...*\n\n'
    '```json\n{"fixes":[{"wrong":"XXX","correct":"YYY"}]}\n```'
)


class TestRepairJson:
    def test_collapses_doubled_key_quotes(self):
        out = L._repair_json('{""wrong":"a",""correct":"b"}')
        assert json.loads(out) == {"wrong": "a", "correct": "b"}

    def test_strips_markdown_fence(self):
        out = L._repair_json('```json\n{"x":1}\n```')
        assert json.loads(out) == {"x": 1}

    def test_takes_first_block_of_multiblock(self):
        out = L._repair_json(REAL_MULTIBLOCK)
        parsed = json.loads(out)
        assert parsed["fixes"][0]["wrong"] == "あるある"

    def test_already_clean_json_unchanged_semantically(self):
        out = L._repair_json('{"fixes":[]}')
        assert json.loads(out) == {"fixes": []}

    def test_bare_array_extracted(self):
        out = L._repair_json('leading junk [1,2,3] trailing')
        assert json.loads(out) == [1, 2, 3]

    def test_empty_input_returns_empty(self):
        assert L._repair_json("") == ""

    def test_no_json_container_returned_as_is(self):
        assert L._repair_json("no json here") == "no json here"


class TestParseSuggestions:
    def test_real_seeded_output_parses_to_four_pairs(self):
        got = L.parse_suggestions(REAL_SEEDED)
        assert len(got) == 4
        assert {"wrong": "空気デザイン", "correct": "クウキデザイン"} in got
        # leading spaces stripped
        assert {"wrong": "クロードコード", "correct": "Claude Code"} in got

    def test_glossary_key_and_backwards_pair_extracted(self):
        # parse only extracts; direction is caught later by seed design, not here.
        got = L.parse_suggestions(REAL_BACKWARDS)
        assert got == [{"wrong": "クウキデザイン", "correct": "空気デザイン"}]

    def test_original_corrected_key_spelling_accepted(self):
        got = L.parse_suggestions('{"fixes":[{"original":"a","corrected":"b"}]}')
        assert got == [{"wrong": "a", "correct": "b"}]

    def test_bare_array_accepted(self):
        got = L.parse_suggestions('[{"wrong":"a","correct":"b"}]')
        assert got == [{"wrong": "a", "correct": "b"}]

    def test_garbage_returns_empty_not_raise(self):
        assert L.parse_suggestions("total garbage no json") == []

    def test_empty_string_returns_empty(self):
        assert L.parse_suggestions("") == []

    def test_entry_missing_keys_is_skipped(self):
        got = L.parse_suggestions('{"fixes":[{"wrong":"a"},{"wrong":"b","correct":"c"}]}')
        assert got == [{"wrong": "b", "correct": "c"}]


class TestAnchorValidate:
    TEXT = "今日は空気デザインの話です。空気デザインはいい。クロドコードも使う。"

    def test_survivor_gets_count(self):
        got = L.anchor_validate(
            [{"wrong": "空気デザイン", "correct": "クウキデザイン"}], self.TEXT
        )
        assert got == [{"wrong": "空気デザイン", "correct": "クウキデザイン", "count": 2}]

    def test_drops_span_not_in_transcript(self):
        # hallucinated span the model invented but that isn't in the text
        got = L.anchor_validate(
            [{"wrong": "存在しない語", "correct": "何か"}], self.TEXT
        )
        assert got == []

    def test_drops_noop_where_wrong_equals_correct(self):
        got = L.anchor_validate(
            [{"wrong": "空気デザイン", "correct": "空気デザイン"}], self.TEXT
        )
        assert got == []

    def test_drops_empty_wrong(self):
        assert L.anchor_validate([{"wrong": "", "correct": "x"}], self.TEXT) == []

    def test_dedups_identical_pairs(self):
        got = L.anchor_validate(
            [
                {"wrong": "クロドコード", "correct": "Claude Code"},
                {"wrong": "クロドコード", "correct": "Claude Code"},
            ],
            self.TEXT,
        )
        assert len(got) == 1

    def test_backwards_suggestion_on_correct_text_is_dropped_when_absent(self):
        # If the model proposes クウキデザイン→空気デザイン but the transcript only
        # contains the (correct) 空気デザイン... here クウキデザイン is absent → dropped.
        got = L.anchor_validate(
            [{"wrong": "クウキデザイン", "correct": "空気デザイン"}], self.TEXT
        )
        assert got == []

    def test_full_real_seeded_pipeline(self):
        # parse → validate against a transcript that contains 3 of the 4 spans.
        text = "空気デザインとクロードコードとクロドコードの話"
        got = L.anchor_validate(L.parse_suggestions(REAL_SEEDED), text)
        wrongs = {g["wrong"] for g in got}
        assert wrongs == {"空気デザイン", "クロードコード", "クロドコード"}  # ノーション absent → dropped


class TestPickDefaultModel:
    def test_prefers_loaded(self):
        models = [{"name": "a", "loaded": False}, {"name": "b", "loaded": True}]
        assert L.pick_default_model(models) == "b"

    def test_falls_back_to_first_when_none_loaded(self):
        models = [{"name": "a", "loaded": False}, {"name": "b", "loaded": False}]
        assert L.pick_default_model(models) == "a"

    def test_none_when_empty(self):
        assert L.pick_default_model([]) is None


class TestBuildPrompt:
    def test_includes_names_examples_and_transcript(self):
        p = L.build_prompt(
            ["クウキデザイン", "Claude Code"],
            [{"wrong": "空気デザイン", "correct": "クウキデザイン"}],
            "今日は空気デザインの話",
        )
        assert "クウキデザイン" in p
        assert "Claude Code" in p
        assert '"空気デザイン" should be "クウキデザイン"' in p
        assert "今日は空気デザインの話" in p
        assert "STRICT JSON" in p

    def test_handles_no_known_names_without_crashing(self):
        p = L.build_prompt([], [], "text")
        assert "(none provided)" in p


class TestBuildChatBody:
    def test_think_is_false(self):
        body = L.build_chat_body("hi", "some-model")
        assert body["think"] is False  # mandatory — thinking mode times out
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert body["model"] == "some-model"


class TestSuggestBrandCorrections:
    def test_unavailable_when_ollama_down(self):
        with patch.object(L, "ollama_available", return_value=False):
            r = L.suggest_brand_corrections("text", ["X"], [], "m")
        assert r["status"] == "unavailable"
        assert r["fixes"] == []

    def test_empty_reply_when_model_returns_none(self):
        with patch.object(L, "ollama_available", return_value=True), \
             patch.object(L, "_post_chat", return_value=None):
            r = L.suggest_brand_corrections("text", ["X"], [], "m")
        assert r["status"] == "empty_reply"
        assert r["fixes"] == []

    def test_ok_path_validates_against_transcript(self):
        text = "空気デザインとクロドコードの話"
        with patch.object(L, "ollama_available", return_value=True), \
             patch.object(L, "_post_chat", return_value=REAL_SEEDED):
            r = L.suggest_brand_corrections(text, ["クウキデザイン"], [], "m")
        assert r["status"] == "ok"
        wrongs = {f["wrong"] for f in r["fixes"]}
        # only the spans actually present in `text` survive anchoring
        assert "空気デザイン" in wrongs
        assert "クロドコード" in wrongs
        assert "ノーション" not in wrongs

    def test_ok_with_no_valid_fixes_returns_empty_list(self):
        with patch.object(L, "ollama_available", return_value=True), \
             patch.object(L, "_post_chat", return_value='{"fixes":[]}'):
            r = L.suggest_brand_corrections("clean text", ["X"], [], "m")
        assert r["status"] == "ok"
        assert r["fixes"] == []
