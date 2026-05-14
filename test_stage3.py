"""Tests for Stage 3 — Video Prompt & Metadata Generator."""

import json
import pytest

from stages.stage3_prompt import _validate, _mock_stage3_output, run
from utils import config as cfg


_SCENE_PROMPT = (
    "A young woman sits at a kitchen table in warm golden morning light, her hands wrapped "
    "around a coffee mug. The camera begins in a tight close-up and slowly pulls back in a "
    "smooth tracking shot, revealing coins scattered across the table around her. Her expression "
    "shifts from relaxed to quietly stunned as she looks down at the money. The colour grade is "
    "warm amber with soft shadows evoking comfort turning into realisation. A single shaft of "
    "natural window light falls diagonally across the scene making the coins glint. The 9:16 "
    "vertical frame keeps her face centred creating an intimate confessional mood. The overall "
    "aesthetic is cinematic and premium — warm human and emotionally resonant throughout."
)

VALID_DATA = {
    "video_prompt_1": _SCENE_PROMPT,
    "video_prompt_2": _SCENE_PROMPT,
    "video_prompt_3": _SCENE_PROMPT,
    "title": "Your coffee habit costs $1,200/year",
    "description": "You buy this every day without thinking. We did the maths. Follow for more.",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11", "tag12"],
    "hashtags": ["PersonalFinance", "SavingMoney", "LatteFactor", "MoneyHacks", "Shorts"],
    "thumbnail_concept": "Coins overflowing from a coffee cup.",
    "pinned_comment": "How much do you spend on coffee monthly?",
    "category": "Finance",
    "language": "en",
}


@pytest.fixture(autouse=True)
def reset_config(tmp_path, monkeypatch):
    config_data = {
        "niche": "personal finance",
        "domain": "money saving tips",
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


def test_validate_passes_valid_data():
    _validate(VALID_DATA)  # should not raise


def test_validate_rejects_short_prompt():
    data = dict(VALID_DATA)
    data["video_prompt_1"] = "Short prompt."
    with pytest.raises(ValueError, match="too short"):
        _validate(data)


def test_validate_rejects_long_title():
    data = dict(VALID_DATA)
    data["title"] = "x" * 61
    with pytest.raises(ValueError, match="too long"):
        _validate(data)


def test_validate_rejects_too_few_tags():
    data = dict(VALID_DATA)
    data["tags"] = ["only", "seven", "tags", "here", "not", "enough", "ok"]
    with pytest.raises(ValueError, match="Not enough tags"):
        _validate(data)


def test_validate_rejects_disqualified_words():
    data = dict(VALID_DATA)
    data["video_prompt_1"] = VALID_DATA["video_prompt_1"] + " A subtitle appears at the bottom."
    with pytest.raises(ValueError, match="disqualified word"):
        _validate(data)


def test_mock_output_valid():
    out = _mock_stage3_output()
    _validate(out)  # mock should pass its own validation
    assert out["category_id"] == "27"  # Finance maps to 27


def test_dry_run_returns_mock():
    from stages.stage2_analyse import _mock_stage2_output
    stage2 = _mock_stage2_output()
    result = run(stage2, dry_run=True)
    assert "video_prompt" in result
    assert "title" in result
    assert "tags" in result
    assert len(result["tags"]) >= 8
