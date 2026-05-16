"""Stage 4 — Video Generation.

Generates five 10-second clips (one per scene prompt) and stitches them into
a 50-second MP4. Falls back to fewer clips if any scene fails.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SUBMIT_STAGGER = 3  # seconds between parallel job submissions to avoid burst 429s


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


def _generate_clip(prompt: str, scene_num: int, duration: int, order: list[str]) -> str | None:
    clip_path = _output_path(f"_scene{scene_num}_raw")
    for provider_name in order:
        try:
            clip = _try_provider(provider_name, prompt, duration, clip_path)
            logger.info("Stage 4 | Scene %d generated with %s", scene_num, provider_name)
            return clip
        except _requests.exceptions.HTTPError as e:
            logger.warning("Stage 4 | Scene %d provider %s HTTP error: %s", scene_num, provider_name, e)
        except EnvironmentError as e:
            if any(k in str(e) for k in ("not set", "credentials", "API key", "KLING", "RUNWAY", "HIGGSFIELD")):
                logger.warning("Stage 4 | Provider %s skipped (not configured): %s", provider_name, e)
            else:
                logger.warning("Stage 4 | Scene %d provider %s failed: %s", scene_num, provider_name, e)
        except (RuntimeError, TimeoutError, Exception) as e:
            logger.warning("Stage 4 | Scene %d provider %s failed: %s", scene_num, provider_name, e)
    logger.error("Stage 4 | Scene %d failed on all providers", scene_num)
    return None


def run(stage3_output: dict, dry_run: bool = False) -> dict:
    """Execute Stage 4. Returns dict with video_file, video_provider, video_duration."""
    conf = cfg.load_config()
    preferred = conf.get("video_provider", "kling")
    duration = conf.get("video_duration_seconds", 10)

    logger.info("Stage 4 | Generating 5-scene video (preferred=%s, duration=%ds each)", preferred, duration)

    if dry_run:
        logger.info("Stage 4 | DRY-RUN — returning mock video path")
        return _mock_stage4_output()

    order = [preferred] + [p for p in PROVIDER_ORDER if p != preferred]

    prompts = [
        stage3_output.get("video_prompt_1"),
        stage3_output.get("video_prompt_2"),
        stage3_output.get("video_prompt_3"),
        stage3_output.get("video_prompt_4"),
        stage3_output.get("video_prompt_5"),
    ]
    if not any(prompts):
        prompts = [stage3_output.get("video_prompt")]
    prompts = [p for p in prompts if p]
    logger.info("Stage 4 | Generating %d scene clip(s) in parallel", len(prompts))

    clips_by_scene: dict[int, str] = {}
    used_provider: str | None = preferred

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {}
        for i, prompt in enumerate(prompts, 1):
            if i > 1:
                time.sleep(SUBMIT_STAGGER)
            future = executor.submit(_generate_clip, prompt, i, duration, order)
            futures[future] = i
            logger.info("Stage 4 | Scene %d submitted", i)

        for future in as_completed(futures):
            scene_num = futures[future]
            try:
                clip = future.result()
                if clip:
                    clips_by_scene[scene_num] = clip
                    logger.info("Stage 4 | Scene %d complete (%d/%d done)",
                                scene_num, len(clips_by_scene), len(prompts))
            except Exception as e:
                logger.error("Stage 4 | Scene %d raised exception: %s", scene_num, e)

    clips = [clips_by_scene[i] for i in sorted(clips_by_scene)]

    if not clips:
        raise RuntimeError("Stage 4 failed: all scene generations failed")

    if len(clips) > 1:
        stitched_path = _output_path("_stitched")
        try:
            ffmpeg.concatenate(clips, stitched_path)
            logger.info("Stage 4 | Stitched %d clips into %s", len(clips), stitched_path)
            for clip in clips:
                try:
                    os.remove(clip)
                except OSError:
                    pass
            raw_file = stitched_path
        except Exception as e:
            logger.warning("Stage 4 | Concatenation failed (%s) — using first clip only", e)
            raw_file = clips[0]
    else:
        raw_file = clips[0]

    try:
        ffmpeg.validate_video(raw_file)
    except Exception as e:
        logger.warning("Stage 4 | Validation skipped (%s) — proceeding", e)

    try:
        info = ffmpeg.probe(raw_file)
        duration_actual = float(info.get("duration", 0))
    except Exception:
        duration_actual = float(duration * len(clips))

    logger.info("Stage 4 | Final video: %s (%.1fs, %d scenes)", raw_file, duration_actual, len(clips))
    return {
        "video_provider": used_provider,
        "video_file": raw_file,
        "video_duration": duration_actual,
        "scenes_generated": len(clips),
    }


def _mock_stage4_output() -> dict:
    mock_path = "generated_videos/dry_run_mock.mp4"
    Path("generated_videos").mkdir(exist_ok=True)
    if not Path(mock_path).exists():
        Path(mock_path).write_bytes(b"\x00" * 1024)
    return {
        "video_provider": "kling",
        "video_file": mock_path,
        "video_duration": 50.0,
        "scenes_generated": 5,
    }
