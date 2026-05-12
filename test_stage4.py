"""Tests for Stage 4 — Video Generation."""

import json
from pathlib import Path

import pytest

from stages.stage4_generate import _mock_stage4_output, run
from utils import config as cfg


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


def test_mock_output_keys():
    out = _mock_stage4_output()
    assert "video_provider" in out
    assert "video_file" in out
    assert "video_duration" in out
    assert out["video_provider"] == "kling"
    assert out["video_duration"] == 10.0


def test_dry_run_returns_mock():
    from stages.stage3_prompt import _mock_stage3_output
    stage3 = _mock_stage3_output()
    result = run(stage3, dry_run=True)
    assert "video_file" in result
    assert Path(result["video_file"]).exists()


def test_dry_run_creates_placeholder_file():
    from stages.stage3_prompt import _mock_stage3_output
    stage3 = _mock_stage3_output()
    result = run(stage3, dry_run=True)
    p = Path(result["video_file"])
    assert p.exists()
    assert p.stat().st_size > 0
