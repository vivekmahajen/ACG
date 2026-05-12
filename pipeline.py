#!/usr/bin/env python3
"""YT-AutoPilot — main pipeline entry point.

Usage:
    python pipeline.py                     # full run
    python pipeline.py --dry-run           # all stages, no real API calls
    python pipeline.py --config path.json  # custom config file
    python pipeline.py --stage 1           # run only up to stage N (for debugging)
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stages import stage1_research, stage2_analyse, stage3_prompt, stage4_generate, stage5_publish
from utils import config as cfg
from utils import database as db
from utils.alerts import send_alert
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def _write_daily_log(log_data: dict) -> None:
    Path("logs").mkdir(exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = f"logs/{date_str}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info("Daily log written: %s", log_path)


def _alert_on_failure(subject: str, body: str, conf: dict) -> None:
    if conf.get("alert_on_failure"):
        send_alert(subject, body, conf.get("alert_email"))


def run_pipeline(dry_run: bool = False, stop_after_stage: int = 5) -> dict:
    pipeline_start = time.monotonic()
    conf = cfg.load_config()
    niche: str = conf["niche"]
    domain: str = conf["domain"]

    ran_at = datetime.now(timezone.utc).isoformat()
    run_id = db.create_run(ran_at, niche, domain)
    logger.info("=" * 60)
    logger.info("YT-AutoPilot | run_id=%d | niche=%s | dry_run=%s", run_id, niche, dry_run)
    logger.info("=" * 60)

    log_data: dict = {
        "date": ran_at[:10],
        "ran_at": ran_at,
        "niche": niche,
        "domain": domain,
        "dry_run": dry_run,
    }

    stage1_out: dict = {}
    stage2_out: dict = {}
    stage3_out: dict = {}
    stage4_out: dict = {}
    stage5_out: dict = {}

    try:
        # ── Stage 1: Research ────────────────────────────────────────────
        logger.info(">>> Stage 1: YouTube Research")
        stage1_out = _run_with_retry(
            lambda: stage1_research.run(dry_run=dry_run),
            max_retries=conf["max_retries"],
            backoff=conf["retry_backoff_seconds"],
            stage_name="Stage 1",
        )
        db.update_run(
            run_id,
            stage_reached="stage1",
            channels_found=len(stage1_out.get("channels", [])),
            videos_collected=len(stage1_out.get("videos", [])),
        )
        if stop_after_stage == 1:
            return _finish(run_id, "success", pipeline_start, log_data, stage1_out=stage1_out)

        # ── Stage 2: Trend Analysis ──────────────────────────────────────
        logger.info(">>> Stage 2: Trend Analysis")
        stage2_out = _run_with_retry(
            lambda: stage2_analyse.run(stage1_out, dry_run=dry_run),
            max_retries=conf["max_retries"],
            backoff=conf["retry_backoff_seconds"],
            stage_name="Stage 2",
        )
        db.update_run(
            run_id,
            stage_reached="stage2",
            trend_topic=stage2_out.get("topic"),
            trend_hook=stage2_out.get("hook"),
            trend_emotion=stage2_out.get("target_emotion"),
        )
        log_data["trend_identified"] = {
            "topic": stage2_out.get("topic"),
            "hook": stage2_out.get("hook"),
            "emotion": stage2_out.get("target_emotion"),
        }
        if stop_after_stage == 2:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 3: Prompt & Metadata ────────────────────────────────────
        logger.info(">>> Stage 3: Video Prompt & Metadata")
        stage3_out = _run_with_retry(
            lambda: stage3_prompt.run(stage2_out, dry_run=dry_run),
            max_retries=conf["max_retries"],
            backoff=conf["retry_backoff_seconds"],
            stage_name="Stage 3",
        )
        db.update_run(
            run_id,
            stage_reached="stage3",
            video_prompt=stage3_out.get("video_prompt"),
            youtube_title=stage3_out.get("title"),
            youtube_tags=json.dumps(stage3_out.get("tags", [])),
            thumbnail_concept=stage3_out.get("thumbnail_concept"),
        )
        if stop_after_stage == 3:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 4: Video Generation ─────────────────────────────────────
        logger.info(">>> Stage 4: Video Generation")
        gen_start = time.monotonic()
        stage4_out = _run_with_retry(
            lambda: stage4_generate.run(stage3_out, dry_run=dry_run),
            max_retries=conf["max_retries"],
            backoff=conf["retry_backoff_seconds"],
            stage_name="Stage 4",
        )
        gen_elapsed = time.monotonic() - gen_start
        db.update_run(
            run_id,
            stage_reached="stage4",
            video_provider=stage4_out.get("video_provider"),
            video_file=stage4_out.get("video_file"),
            video_duration=stage4_out.get("video_duration"),
        )
        log_data["video"] = {
            "provider_used": stage4_out.get("video_provider"),
            "duration_seconds": stage4_out.get("video_duration"),
            "file_size_kb": _file_size_kb(stage4_out.get("video_file", "")),
            "generation_time_seconds": round(gen_elapsed),
        }
        if stop_after_stage == 4:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 5: Publish ──────────────────────────────────────────────
        logger.info(">>> Stage 5: YouTube Publish")
        upload_start = time.monotonic()
        stage5_out = _run_with_retry(
            lambda: stage5_publish.run(stage2_out, stage3_out, stage4_out, run_id, dry_run=dry_run),
            max_retries=conf["max_retries"],
            backoff=conf["retry_backoff_seconds"],
            stage_name="Stage 5",
        )
        upload_elapsed = time.monotonic() - upload_start
        db.update_run(
            run_id,
            stage_reached="stage5",
            youtube_video_id=stage5_out.get("video_id"),
            youtube_url=stage5_out.get("video_url"),
        )
        log_data["upload"] = {
            "video_id": stage5_out.get("video_id"),
            "url": stage5_out.get("video_url"),
            "title": stage3_out.get("title"),
            "upload_time_seconds": round(upload_elapsed),
        }

        video_url = stage5_out.get("video_url", "")
        logger.info("=" * 60)
        logger.info("Pipeline complete! Video URL: %s", video_url)
        logger.info("=" * 60)
        print(video_url)

        return _finish(run_id, "success", pipeline_start, log_data)

    except SystemExit as e:
        # Quota exceeded — do not retry
        msg = str(e)
        logger.error("Pipeline aborted: %s", msg)
        db.update_run(run_id, status="failed", error_message=msg)
        _alert_on_failure("Pipeline aborted", msg, conf)
        log_data["error"] = msg
        _write_daily_log({**log_data, "pipeline": {"status": "aborted", "error": msg}})
        raise

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Pipeline failed: %s\n%s", exc, tb)
        db.update_run(run_id, status="failed", error_message=str(exc))
        _alert_on_failure("Pipeline failed", f"{exc}\n\n{tb}", conf)
        log_data["error"] = str(exc)
        elapsed = time.monotonic() - pipeline_start
        _write_daily_log({
            **log_data,
            "pipeline": {
                "total_duration_seconds": round(elapsed),
                "status": "failed",
                "stages_completed": _count_stages(stage1_out, stage2_out, stage3_out, stage4_out, stage5_out),
            },
        })
        raise


def _finish(run_id: int, status: str, start: float, log_data: dict, **_) -> dict:
    elapsed = time.monotonic() - start
    db.update_run(run_id, status=status, duration_seconds=round(elapsed, 2))
    _write_daily_log({
        **log_data,
        "pipeline": {
            "total_duration_seconds": round(elapsed),
            "status": status,
        },
    })
    return {"run_id": run_id, "status": status, "elapsed_seconds": elapsed}


def _run_with_retry(fn, max_retries: int, backoff: int, stage_name: str):
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except SystemExit:
            raise  # quota exceeded — do not retry
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning("%s attempt %d/%d failed: %s — retrying in %ds",
                               stage_name, attempt, max_retries, e, wait)
                time.sleep(wait)
            else:
                logger.error("%s failed after %d attempts: %s", stage_name, max_retries, e)
    raise RuntimeError(f"{stage_name} exhausted {max_retries} retries") from last_exc


def _file_size_kb(path: str) -> int:
    try:
        return int(Path(path).stat().st_size / 1024)
    except OSError:
        return 0


def _count_stages(*outputs) -> int:
    return sum(1 for o in outputs if o)


def main() -> None:
    parser = argparse.ArgumentParser(description="YT-AutoPilot pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run all stages without real API calls")
    parser.add_argument("--config", default=None, help="Path to config.json (default: ./config.json)")
    parser.add_argument("--stage", type=int, default=5, choices=range(1, 6),
                        help="Stop after this stage number (1–5)")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    cfg.load_config(args.config)
    conf = cfg.load_config()

    db.init_db()
    logger.setLevel(conf.get("log_level", "INFO"))

    run_pipeline(dry_run=args.dry_run, stop_after_stage=args.stage)


if __name__ == "__main__":
    main()
