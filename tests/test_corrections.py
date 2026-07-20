"""Tests for corrections.py — custom-vocabulary transcription fixes."""

import json

import pytest

from preprod import corrections as C
from preprod.corrections import BRAND_CORRECTIONS, correct_text, correct_words


class TestCorrectText:
    def test_brand_name_corrected(self):
        assert correct_text("実際空気デザインのロゴ") == "実際クウキデザインのロゴ"

    def test_bare_air_word_untouched(self):
        # 空気 meaning "air" (not followed by デザイン) must NOT be changed.
        assert correct_text("空気がきれいですね") == "空気がきれいですね"

    def test_no_match_returns_unchanged(self):
        assert correct_text("普通のテキスト") == "普通のテキスト"

    def test_empty_string(self):
        assert correct_text("") == ""

    def test_multiple_occurrences(self):
        assert correct_text("空気デザインと空気デザイン") == "クウキデザインとクウキデザイン"

    def test_correction_only_changes_brand_token(self):
        # The correction rewrites only the matched substring; surrounding text
        # (and therefore every other token's timing) is untouched.
        assert correct_text("これは空気デザインです") == "これはクウキデザインです"

    def test_custom_dict_longest_first(self):
        corr = {"AB": "X", "ABC": "Y"}
        # "ABC" must win over "AB" (longest key applied first)
        assert correct_text("ABC", corr) == "Y"


class TestCorrectWords:
    def test_word_key_corrected_in_place(self):
        words = [{"word": "実際空気デザイン", "start": 0.0, "end": 1.0}]
        n = correct_words(words)
        assert n == 1
        assert words[0]["word"] == "実際クウキデザイン"

    def test_text_key_corrected_in_place(self):
        words = [{"text": "空気デザイン", "start": 0.0, "end": 1.0}]
        n = correct_words(words)
        assert n == 1
        assert words[0]["text"] == "クウキデザイン"

    def test_unchanged_words_not_counted(self):
        words = [{"word": "こんにちは"}, {"word": "空気デザイン"}]
        n = correct_words(words)
        assert n == 1
        assert words[0]["word"] == "こんにちは"
        assert words[1]["word"] == "クウキデザイン"

    def test_word_without_text_key_skipped(self):
        words = [{"start": 0.0, "end": 1.0}]   # malformed, no text/word key
        assert correct_words(words) == 0

    def test_both_keys_corrects_word_only(self):
        # T0264: when a token has BOTH 'word' and 'text', 'word' wins (it is the
        # raw source); 'text' is left untouched and the token counts once.
        words = [{"word": "空気デザイン", "text": "other", "start": 0.0, "end": 1.0}]
        n = correct_words(words)
        assert n == 1
        assert words[0]["word"] == "クウキデザイン"   # corrected
        assert words[0]["text"] == "other"            # untouched

    def test_default_dict_is_brand_corrections(self):
        assert "空気デザイン" in BRAND_CORRECTIONS


class TestUserGlossary:
    """User-grown glossary (~/.preprod/corrections.json) merged over defaults."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        # Redirect the glossary file to a temp path and clear the merged-view
        # cache before AND after each test so nothing leaks across tests or
        # touches the real ~/.preprod/corrections.json.
        monkeypatch.setattr(C, "_USER_CORRECTIONS_PATH", tmp_path / "corrections.json")
        C.invalidate_corrections_cache()
        yield
        C.invalidate_corrections_cache()

    def test_missing_file_returns_empty(self):
        assert C.load_user_corrections() == {}

    def test_active_falls_back_to_builtins_when_no_user_file(self):
        assert C.active_corrections() == BRAND_CORRECTIONS

    def test_save_then_load_roundtrip(self):
        C.save_user_correction("クロドコード", "Claude Code")
        assert C.load_user_corrections() == {"クロドコード": "Claude Code"}

    def test_save_merges_into_active_and_applies(self):
        C.save_user_correction("クロドコード", "Claude Code")
        # default-arg correct_text now picks up the freshly-saved mapping
        assert correct_text("今日はクロドコードを使う") == "今日はClaude Codeを使う"
        # built-in still works too
        assert correct_text("空気デザイン") == "クウキデザイン"

    def test_user_entry_overrides_builtin(self):
        C.save_user_correction("空気デザイン", "KUUKI")   # user overrides default target
        assert C.active_corrections()["空気デザイン"] == "KUUKI"

    def test_malformed_file_ignored(self):
        C._USER_CORRECTIONS_PATH.write_text("{ not json", encoding="utf-8")
        assert C.load_user_corrections() == {}
        assert C.active_corrections() == BRAND_CORRECTIONS

    def test_non_dict_json_ignored(self):
        C._USER_CORRECTIONS_PATH.write_text("[1,2,3]", encoding="utf-8")
        assert C.load_user_corrections() == {}

    def test_non_string_entries_filtered(self):
        C._USER_CORRECTIONS_PATH.write_text(
            json.dumps({"good": "ok", "bad": 5, "": "empty-key"}), encoding="utf-8"
        )
        assert C.load_user_corrections() == {"good": "ok"}

    def test_save_ignores_empty_wrong(self):
        C.save_user_correction("", "x")
        assert C.load_user_corrections() == {}

    def test_cache_invalidated_on_save(self):
        assert C.active_corrections() == BRAND_CORRECTIONS   # populate cache
        C.save_user_correction("foo", "bar")
        assert C.active_corrections().get("foo") == "bar"    # cache refreshed
