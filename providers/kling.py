"""Kling AI text-to-video provider."""

import os
import time

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "https://api.qingque.cn/v1/videos/text2video"
POLL_INTERVAL = 5
MAX_WAIT = 300  # 5 minutes


def generate(prompt: str, duration: int = 10, output_path: str = "generated_videos/output.mp4") -> str:
    api_key = os.environ.get("KLING_API_KEY")
    if not api_key:
        raise EnvironmentError("KLING_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "kling-v2",
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": "9:16",
        "cfg_scale": 0.5,
        "mode": "std",
    }

    logger.info("Kling | Submitting generation request")
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError(f"Kling did not return a task_id. Response: {data}")

    logger.info("Kling | task_id=%s — polling for completion", task_id)
    return _poll_and_download(task_id, headers, output_path)


def _poll_and_download(task_id: str, headers: dict, output_path: str) -> str:
    status_url = f"https://api.qingque.cn/v1/videos/{task_id}"
    elapsed = 0

    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = requests.get(status_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "").lower()
        logger.info("Kling | task=%s status=%s elapsed=%ds", task_id, status, elapsed)

        if status in ("succeeded", "completed", "finished"):
            video_url = (
                data.get("output", {}).get("video_url")
                or data.get("video_url")
                or data.get("url")
            )
            if not video_url:
                raise RuntimeError(f"Kling succeeded but no video_url in response: {data}")
            return _download(video_url, output_path)

        if status in ("failed", "error"):
            raise RuntimeError(f"Kling generation failed: {data.get('error', data)}")

    raise TimeoutError(f"Kling timed out after {MAX_WAIT}s for task {task_id}")


def _download(url: str, output_path: str) -> str:
    import os as _os
    _os.makedirs(_os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("Kling | Downloading video to %s", output_path)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Kling | Download complete: %s", output_path)
    return output_path
