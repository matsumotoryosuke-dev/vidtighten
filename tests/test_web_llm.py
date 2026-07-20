"""Tests for the local-LLM correction routes in web.py.

Ollama is never actually contacted — llm_correct's HTTP-facing functions are
patched, so these tests cover the route wiring, task lifecycle, and glossary
persistence deterministically and offline.
"""

import time

import pytest
from unittest.mock import patch


@pytest.fixture()
def client():
    import preprod.web  # warm the module cache before any patching
    with patch("preprod.web.extract_audio"), patch("preprod.web.probe_media"):
        from preprod.web import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


class TestLlmModels:
    def test_available_lists_models_and_default(self, client):
        models = [{"name": "a", "loaded": False}, {"name": "b", "loaded": True}]
        with patch("preprod.web.llm_correct.ollama_available", return_value=True), \
             patch("preprod.web.llm_correct.list_models", return_value=models):
            r = client.get("/api/llm/models")
        assert r.status_code == 200
        body = r.get_json()
        assert body["available"] is True
        assert body["default"] == "b"          # loaded model preferred
        assert len(body["models"]) == 2

    def test_unavailable_returns_empty(self, client):
        with patch("preprod.web.llm_correct.ollama_available", return_value=False):
            r = client.get("/api/llm/models")
        body = r.get_json()
        assert body["available"] is False
        assert body["models"] == []
        assert body["default"] is None


class TestLlmSuggest:
    def test_missing_transcript_text_is_400(self, client):
        r = client.post("/api/llm/suggest", json={"model": "m"})
        assert r.status_code == 400

    def test_missing_model_is_400(self, client):
        r = client.post("/api/llm/suggest", json={"transcript_text": "hi"})
        assert r.status_code == 400

    def test_suggest_runs_and_status_returns_fixes(self, client):
        fake = {"status": "ok",
                "fixes": [{"wrong": "空気デザイン", "correct": "クウキデザイン", "count": 2}],
                "error": None}
        with patch("preprod.web.llm_correct.suggest_brand_corrections", return_value=fake):
            start = client.post("/api/llm/suggest",
                                json={"transcript_text": "空気デザイン", "model": "m"})
            assert start.status_code == 200
            task_id = start.get_json()["task_id"]
            # worker runs in a daemon thread; poll briefly for completion
            for _ in range(50):
                st = client.get(f"/api/analyze/status/{task_id}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.02)
        assert st["status"] == "done"
        assert st["result"]["fixes"][0]["wrong"] == "空気デザイン"

    def test_suggest_unavailable_marks_task_error(self, client):
        fake = {"status": "unavailable", "fixes": [], "error": "Ollama not reachable"}
        with patch("preprod.web.llm_correct.suggest_brand_corrections", return_value=fake):
            task_id = client.post("/api/llm/suggest",
                                  json={"transcript_text": "x", "model": "m"}).get_json()["task_id"]
            for _ in range(50):
                st = client.get(f"/api/analyze/status/{task_id}").get_json()
                if st["status"] != "running":
                    break
                time.sleep(0.02)
        assert st["status"] == "error"
        assert st["error"] == "Ollama not reachable"


class TestLlmGlossaryAdd:
    def test_add_persists_via_save_user_correction(self, client):
        with patch("preprod.web.save_user_correction") as save:
            r = client.post("/api/llm/glossary/add",
                            json={"wrong": "クロドコード", "correct": "Claude Code"})
        assert r.status_code == 200
        assert r.get_json()["ok"] is True
        save.assert_called_once_with("クロドコード", "Claude Code")

    def test_add_missing_fields_is_400(self, client):
        assert client.post("/api/llm/glossary/add", json={"wrong": "x"}).status_code == 400
        assert client.post("/api/llm/glossary/add", json={"correct": "y"}).status_code == 400
