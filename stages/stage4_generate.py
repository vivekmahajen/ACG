"""Stage 4 — Video Generation.

Submits the prompt to a text-to-video API and downloads the resulting MP4.
Cascades through providers: Kling → Runway → Higgsfield.
Post-processes the video (trim, scale, fade) and validates the result.
"""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests as _requests

from providers import higgsfield, kling, runway
from utils import config as cfg
from utils import ffmpeg
from utils.logger import get_logger

logger = get_logger(__name__)

PROVIDER_ORDER = ["kling", "runway", "higgsfield"]
PROVIDER_MAP = {
    "kling": kling,
    "runway": runway,
    "higgsfield": higgsfield,
}


def _output_path(suffix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    Path("generated_videos").mkdir(exist_ok=True)
    return f"generated_videos/{ts}{suffix}.mp4"


def _try_provider(name: str, prompt: str, duration: int, raw_path: str) -> str:
    provider = PROVIDER_MAP.get(name)
    if provider is None:
        raise ValueError(f"Unknown provider: {name}")
    logger.info("Stage 4 | Trying provider: %s", name)
    return provider.generate(prompt, duration=duration, output_path=raw_path)


def run(stage3_output: dict, dry_run: bool = False) -> dict:
    """Execute Stage 4. Returns dict with video_file, video_provider, video_duration."""
    conf = cfg.load_config()
    preferred = conf.get("video_provider", "kling")
    duration = conf.get("video_duration_seconds", 10)

    prompt: str = stage3_output["video_prompt"]
    logger.info("Stage 4 | Generating video (preferred=%s, duration=%ds)", preferred, duration)

    if dry_run:
        logger.info("Stage 4 | DRY-RUN — returning mock video path")
        return _mock_stage4_output()

    # Build provider order: preferred first, then the others
    order = [preferred] + [p for p in PROVIDER_ORDER if p != preferred]
    raw_path = _output_path("_raw")
    final_path = _output_path("_final")

    used_provider: str | None = None
    raw_file: str | None = None

    for provider_name in order:
        try:
            raw_file = _try_provider(provider_name, prompt, duration, raw_path)
            used_provider = provider_name
            logger.info("Stage 4 | Generation succeeded with %s", provider_name)
            break
        except _requests.exceptions.HTTPError as e:
            logger.warning("Stage 4 | Provider %s HTTP error: %s — trying next", provider_name, e)
        except EnvironmentError as e:
            if any(k in str(e) for k in ("not set", "credentials", "API key", "KLING", "RUNWAY", "HIGGSFIELD")):
                logger.warning("Stage 4 | Provider %s skipped (not configured): %s", provider_name, e)
            else:
                logger.warning("Stage 4 | Provider %s failed: %s — trying next", provider_name, e)
        except (RuntimeError, TimeoutError, Exception) as e:
            logger.warning("Stage 4 | Provider %s failed: %s — trying next", provider_name, e)

    if not raw_file or not used_provider:
        raise RuntimeError("Stage 4 failed: all providers failed or are unconfigured")

    # Post-process
    logger.info("Stage 4 | Post-processing video")
    try:
        ffmpeg.post_process(raw_file, final_path)
    except Exception as e:
        logger.warning("Stage 4 | Post-processing failed (%s) — using raw file", e)
        shutil.copy(raw_file, final_path)

    # Validate
    try:
        ffmpeg.validate_video(final_path)
    except Exception as e:
        # Try once more with raw file if post-processing created a bad file
        logger.warning("Stage 4 | Validation failed on processed file (%s) — trying raw", e)
        ffmpeg.validate_video(raw_file)
        final_path = raw_file

    # Clean up intermediate raw file (if different from final)
    if raw_file != final_path and Path(raw_file).exists():
        try:
            os.remove(raw_file)
        except OSError:
            pass

    info = ffmpeg.probe(final_path)
    duration_actual = float(info.get("duration", 0))

    logger.info("Stage 4 | Final video: %s (%.1fs)", final_path, duration_actual)
    return {
        "video_provider": used_provider,
        "video_file": final_path,
        "video_duration": duration_actual,
    }


def _mock_stage4_output() -> dict:
    mock_path = "generated_videos/dry_run_mock.mp4"
    Path("generated_videos").mkdir(exist_ok=True)
    # Create a tiny valid placeholder so downstream checks don't fail on file existence
    if not Path(mock_path).exists():
        Path(mock_path).write_bytes(b"\x00" * 1024)
    return {
        "video_provider": "kling",
        "video_file": mock_path,
        "video_duration": 10.0,
    }
