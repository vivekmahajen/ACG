"""Tests for Stage 1 — YouTube Research Agent."""

import json
import pytest

from stages.stage1_research import (
    _engagement_rate,
    _parse_duration,
    _mock_stage1_output,
    run,
)
from utils import config as cfg


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

    # Use an in-memory database so no file is created
    import utils.database as dbu
    monkeypatch.setattr(dbu, "DB_PATH", tmp_path / "test.db")
    dbu.init_db(tmp_path / "test.db")

    cfg.reset()
    yield
    cfg.reset()


def test_parse_duration_standard():
    assert _parse_duration("PT1M30S") == 90
    assert _parse_duration("PT10S") == 10
    assert _parse_duration("PT1H") == 3600


def test_parse_duration_invalid():
    assert _parse_duration("NOT_A_DURATION") == 0


def test_engagement_rate_normal():
    rate = _engagement_rate(likes=100, comments=20, views=1000)
    assert abs(rate - 0.12) < 1e-9


def test_engagement_rate_zero_views():
    assert _engagement_rate(0, 0, 0) == 0.0


def test_mock_output_structure():
    out = _mock_stage1_output()
    assert "channels" in out
    assert "videos" in out
    assert len(out["channels"]) >= 1
    assert len(out["videos"]) >= 1

    video = out["videos"][0]
    required = {"title", "video_id", "channel_name", "channel_id", "view_count",
                "like_count", "comment_count", "duration_seconds", "tags",
                "description_snippet", "published_at", "engagement_rate"}
    assert required.issubset(video.keys())


def test_dry_run_returns_mock():
    result = run(dry_run=True)
    assert "channels" in result
    assert "videos" in result
    assert len(result["videos"]) > 0
