import json
import os
from pathlib import Path
from typing import Any


REQUIRED_KEYS = [
    "niche",
    "domain",
    "channel_id",
    "schedule_time",
    "timezone",
    "video_provider",
    "video_duration_seconds",
    "youtube_category",
    "privacy",
    "max_retries",
    "retry_backoff_seconds",
    "top_channels_to_scan",
    "videos_per_channel",
    "channels_to_analyse",
    "min_subscriber_count",
    "days_lookback",
    "dedup_lookback_days",
    "language",
    "claude_model",
    "log_level",
]

_config: dict[str, Any] | None = None


def load_config(path: str | None = None) -> dict[str, Any]:
    global _config
    if _config is not None:
        return _config

    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.json")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path) as f:
        cfg = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    _config = cfg
    return _config


def get(key: str, default: Any = None) -> Any:
    cfg = load_config()
    return cfg.get(key, default)


def reset() -> None:
    """Reset cached config — used in tests."""
    global _config
    _config = None
