#!/usr/bin/env python3
"""YT-AutoPilot — main pipeline entry point.

Usage:
    python pipeline.py                        # full run
    python pipeline.py --dry-run              # all stages, no real API calls
    python pipeline.py --config path.json     # custom config file
    python pipeline.py --stage 4              # stop after stage 4
    python pipeline.py --resume-from 5        # skip to stage 5 using saved state
"""

import argparse
import base64
import json
import os
import sys
import time

# CI: decode credentials from env vars so OAuth works without local files
if os.environ.get("TOKEN_JSON_B64"):
    with open("token.json", "w") as _f:
        _f.write(base64.b64decode(os.environ["TOKEN_JSON_B64"]).decode())

if os.environ.get("CLIENT_SECRET_JSON_B64"):
    with open("client_secret.json", "w") as _f:
        _f.write(base64.b64decode(os.environ["CLIENT_SECRET_JSON_B64"]).decode())

import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stages import stage1_research, stage2_analyse, stage2b_resources, stage3_prompt, stage4_generate, stage4b_audio, stage5_publish
from utils import config as cfg
from utils import database as db
from utils.alerts import send_alert
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

STATE_FILE = "logs/pipeline_state.json"


def _save_state(stage1_out, stage2_out, stage3_out, stage4_out, resources=None) -> None:
    Path("logs").mkdir(exist_ok=True)
    state = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stage1_out": stage1_out,
        "stage2_out": stage2_out,
        "stage3_out": stage3_out,
        "stage4_out": stage4_out,
        "resources": resources or [],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_state() -> tuple[dict, dict, dict, dict, list]:
    if not Path(STATE_FILE).exists():
        raise FileNotFoundError(
            f"No saved state found at {STATE_FILE}. Run the full pipeline first."
        )
    with open(STATE_FILE) as f:
        state = json.load(f)
    logger.info("Loaded saved state from %s (saved at %s)", STATE_FILE, state.get("saved_at", "unknown"))
    return (
        state.get("stage1_out", {}),
        state.get("stage2_out", {}),
        state.get("stage3_out", {}),
        state.get("stage4_out", {}),
        state.get("resources", []),
    )


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


def run_pipeline(dry_run: bool = False, stop_after_stage: int = 5, resume_from: int = 1) -> dict:
    pipeline_start = time.monotonic()
    conf = cfg.load_config()
    niche: str = conf["niche"]
    domain: str = conf["domain"]

    ran_at = datetime.now(timezone.utc).isoformat()
    run_id = db.create_run(ran_at, niche, domain)
    logger.info("=" * 60)
    logger.info("YT-AutoPilot | run_id=%d | niche=%s | dry_run=%s | resume_from=%d",
                run_id, niche, dry_run, resume_from)
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
    stage2b_resources_out: list = []
    stage3_out: dict = {}
    stage4_out: dict = {}
    stage5_out: dict = {}

    # Load saved state if resuming mid-pipeline
    if resume_from > 1:
        stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out = _load_state()
        logger.info("Resuming from Stage %d — skipping stages 1–%d", resume_from, resume_from - 1)

    try:
        # ── Stage 1: Research ────────────────────────────────────────────
        if resume_from <= 1:
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
            _save_state(stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out)
        if stop_after_stage == 1:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 2: Trend Analysis ──────────────────────────────────────
        if resume_from <= 2:
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
            _save_state(stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out)

        # ── Stage 2b: Resource Search ────────────────────────────────────
        if resume_from <= 2:
            logger.info(">>> Stage 2b: Resource Search")
            stage2b_resources_out = stage2b_resources.run(stage2_out, dry_run=dry_run)
            logger.info("Stage 2b | Resources found: %d", len(stage2b_resources_out))

        if stop_after_stage == 2:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 3: Prompt & Metadata ────────────────────────────────────
        if resume_from <= 3:
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
            _save_state(stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out)
        if stop_after_stage == 3:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 4: Video Generation ─────────────────────────────────────
        if resume_from <= 4:
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
            _save_state(stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out)

        # ── Stage 4b: Voiceover ───────────────────────────────────────────
        if resume_from <= 4:
            logger.info(">>> Stage 4b: Voiceover Generation")
            stage4_out = _run_with_retry(
                lambda: stage4b_audio.run(stage2_out, stage3_out, stage4_out, dry_run=dry_run, resources=stage2b_resources_out),
                max_retries=conf["max_retries"],
                backoff=conf["retry_backoff_seconds"],
                stage_name="Stage 4b",
            )
            _save_state(stage1_out, stage2_out, stage3_out, stage4_out, stage2b_resources_out)

        if stop_after_stage == 4:
            return _finish(run_id, "success", pipeline_start, log_data)

        # ── Stage 5: Publish ──────────────────────────────────────────────
        logger.info(">>> Stage 5: YouTube Publish")
        logger.info("Stage 5 | Uploading: %s", stage4_out.get("video_file"))
        upload_start = time.monotonic()
        stage5_out = _run_with_retry(
            lambda: stage5_publish.run(stage2_out, stage3_out, stage4_out, run_id, dry_run=dry_run, resources=stage2b_resources_out),
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
            raise
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
    parser.add_argument("--resume-from", type=int, default=1, choices=range(1, 6),
                        dest="resume_from",
                        help="Resume from stage N using saved state (skips earlier stages)")
    args = parser.parse_args()

    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    cfg.load_config(args.config)
    conf = cfg.load_config()

    db.init_db()
    logger.setLevel(conf.get("log_level", "INFO"))

    run_pipeline(dry_run=args.dry_run, stop_after_stage=args.stage, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
