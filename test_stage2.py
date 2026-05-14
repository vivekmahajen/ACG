"""Tests for Stage 2 — Trend Analyser."""

import json
import pytest

from stages.stage2_analyse import _parse_and_validate, _build_user_prompt, _mock_stage2_output, run
from utils import config as cfg


VALID_RESULT = {
    "topic": "Why your coffee habit is costing you $1,200 a year",
    "why_this_topic": "High engagement signal from competitor videos.",
    "hook": "You're losing $1,200 a year without knowing it.",
    "key_visual_idea": "Coins spilling from a coffee cup.",
    "target_emotion": "surprise",
    "estimated_watch_through_rate": "high — strong financial hook",
    "competitor_angle": "Make it visual, not a spreadsheet.",
}


@pytest.fixture(autouse=True)
def reset_config(tmp_path, monkeypatch):
    config_data = {
        "niche": "personal finance",
        "domain": "money saving tips for millennials",
        "channel_id": "UC_TEST",
        "schedule_time": "08:00",
        "timezone": "America/New_York",
        "video_provider": "kling",
        "video_duration_seconds": 10,
        "vertical_format": True,
        "youtube_category": "Finance",
        "privacy": "public",
        "max_retries": 3,
        "retry_backoff_seconds": 30,
        "keep_local_video": False,
        "alert_email": "test@example.com",
        "alert_on_failure": False,
        "top_channels_to_scan": 20,
        "videos_per_channel": 10,
        "channels_to_analyse": 5,
        "min_subscriber_count": 10000,
        "days_lookback": 30,
        "dedup_lookback_days": 7,
        "language": "en",
        "claude_model": "claude-sonnet-4-6",
        "log_level": "INFO",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config_data))
    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))

    import utils.database as dbu
    monkeypatch.setattr(dbu, "DB_PATH", tmp_path / "test.db")
    dbu.init_db(tmp_path / "test.db")

    cfg.reset()
    yield
    cfg.reset()


def test_parse_valid_result():
    result = _parse_and_validate(json.dumps(VALID_RESULT))
    assert result["topic"] == VALID_RESULT["topic"]
    assert result["target_emotion"] == "surprise"


def test_parse_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_and_validate("not json at all {")


def test_parse_missing_keys():
    incomplete = {"topic": "Something"}
    with pytest.raises(ValueError, match="missing keys"):
        _parse_and_validate(json.dumps(incomplete))


def test_parse_topic_too_long():
    data = dict(VALID_RESULT)
    data["topic"] = "x" * 151
    with pytest.raises(ValueError, match="Topic too long"):
        _parse_and_validate(json.dumps(data))


def test_build_user_prompt_contains_niche():
    prompt = _build_user_prompt("personal finance", "money tips", [], [])
    assert "personal finance" in prompt
    assert "money tips" in prompt


def test_build_user_prompt_dedup_block():
    prompt = _build_user_prompt("finance", "tips", [], ["Old topic 1", "Old topic 2"])
    assert "Old topic 1" in prompt
    assert "Old topic 2" in prompt


def test_mock_output_keys():
    out = _mock_stage2_output()
    required = {"topic", "why_this_topic", "hook", "key_visual_idea",
                "target_emotion", "estimated_watch_through_rate", "competitor_angle"}
    assert required.issubset(out.keys())


def test_dry_run_returns_mock():
    from stages.stage1_research import _mock_stage1_output
    stage1 = _mock_stage1_output()
    result = run(stage1, dry_run=True)
    assert "topic" in result
    assert len(result["topic"]) > 0
