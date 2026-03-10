"""Tests for D3 standalone healing worker."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from model_atlas.phase_d_worker import _parse_and_validate


@pytest.fixture()
def mock_openai():
    """Inject a fake openai module so main() can import from it."""
    fake_mod = types.ModuleType("openai")
    mock_cls = MagicMock()
    fake_mod.OpenAI = mock_cls  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"openai": fake_mod}):
        yield mock_cls


class TestParseAndValidate:
    def test_valid_output(self):
        """Parses well-formed healing JSON."""
        text = json.dumps(
            {
                "summary": "A code generation model",
                "selected_anchors": ["code-generation", "reasoning"],
                "rationale": "Model is for code",
            }
        )
        result = _parse_and_validate(text)
        assert result["summary"] == "A code generation model"
        assert result["selected_anchors"] == ["code-generation", "reasoning"]
        assert result["rationale"] == "Model is for code"

    def test_filters_against_valid_anchors(self):
        """Only anchors in the valid set are returned."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["code-generation", "invalid-anchor", "chat"],
            }
        )
        result = _parse_and_validate(text, valid_anchors={"code-generation", "chat"})
        assert result["selected_anchors"] == ["code-generation", "chat"]

    def test_strips_short_anchors(self):
        """Anchors shorter than 3 chars are dropped."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["ab", "code-generation"],
            }
        )
        result = _parse_and_validate(text)
        assert result["selected_anchors"] == ["code-generation"]

    def test_max_five_anchors(self):
        """At most 5 anchors returned."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": [f"anchor-{i}" for i in range(10)],
            }
        )
        result = _parse_and_validate(text)
        assert len(result["selected_anchors"]) == 5

    def test_missing_summary_raises(self):
        """Empty summary raises ValueError."""
        text = json.dumps({"summary": "", "selected_anchors": ["chat"]})
        with pytest.raises(ValueError, match="Missing or empty"):
            _parse_and_validate(text)

    def test_non_dict_raises(self):
        """Non-object JSON raises ValueError."""
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_and_validate("[1, 2, 3]")

    def test_invalid_json_raises(self):
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_and_validate("not json")

    def test_non_list_anchors_raises(self):
        """Non-list selected_anchors raises ValueError."""
        text = json.dumps({"summary": "A model", "selected_anchors": "chat"})
        with pytest.raises(ValueError, match="must be a list"):
            _parse_and_validate(text)

    def test_non_string_anchors_skipped(self):
        """Non-string items in anchors are silently skipped."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": [123, "chat", None],
            }
        )
        result = _parse_and_validate(text)
        assert result["selected_anchors"] == ["chat"]

    def test_non_string_rationale_defaults_empty(self):
        """Non-string rationale defaults to empty string."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["chat"],
                "rationale": 42,
            }
        )
        result = _parse_and_validate(text)
        assert result["rationale"] == ""

    def test_missing_rationale_defaults_empty(self):
        """Missing rationale field defaults to empty string."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["chat"],
            }
        )
        result = _parse_and_validate(text)
        assert result["rationale"] == ""

    def test_anchors_lowercased_and_stripped(self):
        """Anchor labels are lowercased and stripped."""
        text = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["  Code-Generation  ", " CHAT "],
            }
        )
        result = _parse_and_validate(text)
        assert result["selected_anchors"] == ["code-generation", "chat"]


class TestWorkerMain:
    def _make_mock_client(self, content):
        """Create a mock OpenAI client that returns content."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    def test_processes_input_shard(self, tmp_path, mock_openai):
        """main() reads input JSONL, calls LLM, writes output JSONL."""
        from model_atlas.phase_d_worker import main

        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"

        inp.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "healing_prompt": "Classify this model",
                    "valid_anchors": ["chat", "reasoning"],
                    "run_id": "run-1",
                    "original_prompt": "original prompt",
                    "original_response": '{"summary": "old", "selected_anchors": ["chat"]}',
                }
            )
            + "\n"
        )

        llm_output = json.dumps(
            {
                "summary": "A reasoning model",
                "selected_anchors": ["reasoning"],
                "rationale": "Fixed classification",
            }
        )
        mock_client = self._make_mock_client(llm_output)
        mock_openai.return_value = mock_client

        with patch("sys.argv", ["worker", "--input", str(inp), "--output", str(out)]):
            main()

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert result["model_id"] == "test/model-a"
        assert result["selected_anchors"] == ["reasoning"]
        assert result["summary"] == "A reasoning model"

    def test_handles_llm_error(self, tmp_path, mock_openai):
        """main() writes error record when LLM call fails."""
        from model_atlas.phase_d_worker import main

        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"

        inp.write_text(
            json.dumps(
                {
                    "model_id": "test/model-a",
                    "healing_prompt": "Classify",
                    "run_id": "run-1",
                }
            )
            + "\n"
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_openai.return_value = mock_client

        with patch("sys.argv", ["worker", "--input", str(inp), "--output", str(out)]):
            main()

        result = json.loads(out.read_text().strip())
        assert result["model_id"] == "test/model-a"
        assert "error" in result

    def test_skips_blank_lines(self, tmp_path, mock_openai):
        """main() skips blank and invalid JSON lines."""
        from model_atlas.phase_d_worker import main

        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"

        inp.write_text("\n\nnot-json\n")

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        with patch("sys.argv", ["worker", "--input", str(inp), "--output", str(out)]):
            main()

        assert out.read_text().strip() == ""
        mock_client.chat.completions.create.assert_not_called()

    def test_respects_shutdown_signal(self, tmp_path, mock_openai):
        """main() stops processing on shutdown signal."""
        import model_atlas.phase_d_worker as worker_mod
        from model_atlas.phase_d_worker import main

        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"

        for i in range(2):
            with open(inp, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "model_id": f"test/model-{i}",
                            "healing_prompt": "Classify",
                            "run_id": "run-1",
                        }
                    )
                    + "\n"
                )

        llm_output = json.dumps(
            {
                "summary": "A model",
                "selected_anchors": ["chat"],
            }
        )
        mock_client = self._make_mock_client(llm_output)
        mock_openai.return_value = mock_client

        worker_mod._shutdown = True
        try:
            with patch(
                "sys.argv", ["worker", "--input", str(inp), "--output", str(out)]
            ):
                main()

            assert out.read_text().strip() == ""
        finally:
            worker_mod._shutdown = False
