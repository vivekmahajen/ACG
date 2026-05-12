"""Runway Gen-4 Turbo text-to-video provider (fallback)."""

import os
import time

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "https://api.dev.runwayml.com/v1/image_to_video"
POLL_BASE = "https://api.dev.runwayml.com/v1/tasks"
POLL_INTERVAL = 5
MAX_WAIT = 300


def generate(prompt: str, duration: int = 10, output_path: str = "generated_videos/output.mp4") -> str:
    api_key = os.environ.get("RUNWAY_API_KEY")
    if not api_key:
        raise EnvironmentError("RUNWAY_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gen4_turbo",
        "prompt_text": prompt,
        "duration": duration,
        "ratio": "768:1280",
    }

    logger.info("Runway | Submitting generation request")
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Runway did not return a task id. Response: {data}")

    logger.info("Runway | task_id=%s — polling", task_id)
    return _poll_and_download(task_id, headers, output_path)


def _poll_and_download(task_id: str, headers: dict, output_path: str) -> str:
    status_url = f"{POLL_BASE}/{task_id}"
    elapsed = 0

    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = requests.get(status_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "").lower()
        logger.info("Runway | task=%s status=%s elapsed=%ds", task_id, status, elapsed)

        if status == "succeeded":
            video_url = data.get("output", [None])[0] if data.get("output") else None
            if not video_url:
                raise RuntimeError(f"Runway succeeded but no output URL: {data}")
            return _download(video_url, output_path, headers)

        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Runway task {status}: {data.get('failure', data)}")

    raise TimeoutError(f"Runway timed out after {MAX_WAIT}s for task {task_id}")


def _download(url: str, output_path: str, headers: dict) -> str:
    import os as _os
    _os.makedirs(_os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("Runway | Downloading to %s", output_path)
    resp = requests.get(url, stream=True, headers=headers, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Runway | Download complete")
    return output_path
