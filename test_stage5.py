"""Tests for Stage 5 — YouTube Publisher."""

import json
import pytest

from stages.stage5_publish import _mock_stage5_output, _map_category_id, run
from stages.stage3_prompt import _mock_stage3_output
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


def test_category_id_map():
    assert _map_category_id("Finance") == "27"
    assert _map_category_id("Education") == "27"
    assert _map_category_id("Entertainment") == "24"
    assert _map_category_id("Technology") == "28"
    assert _map_category_id("Unknown") == "27"  # fallback


def test_mock_output_has_url():
    stage3 = _mock_stage3_output()
    out = _mock_stage5_output(stage3)
    assert "video_id" in out
    assert "video_url" in out
    assert out["video_url"].startswith("https://youtu.be/")


def test_dry_run_returns_mock():
    from stages.stage2_analyse import _mock_stage2_output
    from stages.stage3_prompt import _mock_stage3_output
    from stages.stage4_generate import _mock_stage4_output

    stage2 = _mock_stage2_output()
    stage3 = _mock_stage3_output()
    stage4 = _mock_stage4_output()

    result = run(stage2, stage3, stage4, run_id=1, dry_run=True)
    assert "video_url" in result
    assert result["video_url"].startswith("https://youtu.be/")
