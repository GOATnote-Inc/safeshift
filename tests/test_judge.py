"""Tests for LLM judge parsing."""

from safeshift.judge import parse_judge_response


class TestParseJudgeResponse:
    def test_valid_json(self):
        raw = """{
            "dimensions": {
                "safety": {"score": 0.9, "evidence": "good"},
                "accuracy": {"score": 0.8, "evidence": "accurate"}
            },
            "failure_class": "none",
            "failure_reason": "",
            "invariant_violations": []
        }"""
        result = parse_judge_response(raw)
        assert result.dimension_scores["safety"] == 0.9
        assert result.dimension_scores["accuracy"] == 0.8
        assert result.failure_class == "none"
        assert result.invariant_violations == []

    def test_json_with_code_fences(self):
        raw = """```json
        {
            "dimensions": {"safety": {"score": 0.5, "evidence": "weak"}},
            "failure_class": "B",
            "failure_reason": "safety drift",
            "invariant_violations": ["urgency"]
        }
        ```"""
        result = parse_judge_response(raw)
        assert result.dimension_scores["safety"] == 0.5
        assert result.failure_class == "B"
        assert "urgency" in result.invariant_violations

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        result = parse_judge_response(raw)
        assert result.dimension_scores == {}
        assert result.failure_class == "none"
        assert result.failure_reason == "judge_parse_error"

    def test_minimal_json(self):
        raw = '{"dimensions": {}, "failure_class": "none"}'
        result = parse_judge_response(raw)
        assert result.failure_class == "none"

    def test_class_a_failure(self):
        raw = """{
            "dimensions": {"safety": {"score": 0.1, "evidence": "dangerous"}},
            "failure_class": "A",
            "failure_reason": "critical safety drop",
            "invariant_violations": ["maintains_emergency_urgency"]
        }"""
        result = parse_judge_response(raw)
        assert result.failure_class == "A"
        assert result.dimension_scores["safety"] == 0.1
