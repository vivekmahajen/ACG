"""Higgsfield video generation provider (second fallback)."""

import os
import time

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_API = "https://api.higgsfield.ai/v1/generate/image"
VIDEO_API = "https://api.higgsfield.ai/v1/generate/video"
POLL_INTERVAL = 5
MAX_WAIT = 300


def generate(prompt: str, duration: int = 10, output_path: str = "generated_videos/output.mp4") -> str:
    api_key = os.environ.get("HIGGSFIELD_API_KEY")
    if not api_key:
        raise EnvironmentError("HIGGSFIELD_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Step 1: generate a reference image
    logger.info("Higgsfield | Generating reference image")
    img_resp = requests.post(
        IMAGE_API,
        json={"model": "nano_banana_2", "prompt": prompt, "aspect_ratio": "9:16"},
        headers=headers,
        timeout=60,
    )
    img_resp.raise_for_status()
    img_data = img_resp.json()
    image_url = img_data.get("url") or img_data.get("image_url")
    if not image_url:
        raise RuntimeError(f"Higgsfield image API returned no URL: {img_data}")

    # Step 2: animate the image
    logger.info("Higgsfield | Animating image with seedance_2_0")
    vid_resp = requests.post(
        VIDEO_API,
        json={
            "model": "seedance_2_0",
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": "9:16",
        },
        headers=headers,
        timeout=30,
    )
    vid_resp.raise_for_status()
    vid_data = vid_resp.json()

    task_id = vid_data.get("task_id") or vid_data.get("id")
    if not task_id:
        raise RuntimeError(f"Higgsfield video API returned no task_id: {vid_data}")

    return _poll_and_download(task_id, headers, output_path)


def _poll_and_download(task_id: str, headers: dict, output_path: str) -> str:
    status_url = f"https://api.higgsfield.ai/v1/tasks/{task_id}"
    elapsed = 0

    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        resp = requests.get(status_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "").lower()
        logger.info("Higgsfield | task=%s status=%s elapsed=%ds", task_id, status, elapsed)

        if status in ("succeeded", "completed"):
            video_url = data.get("video_url") or data.get("url")
            if not video_url:
                raise RuntimeError(f"Higgsfield succeeded but no video_url: {data}")
            return _download(video_url, output_path, headers)

        if status in ("failed", "error"):
            raise RuntimeError(f"Higgsfield generation failed: {data}")

    raise TimeoutError(f"Higgsfield timed out after {MAX_WAIT}s for task {task_id}")


def _download(url: str, output_path: str, headers: dict) -> str:
    import os as _os
    _os.makedirs(_os.path.dirname(output_path) or ".", exist_ok=True)
    logger.info("Higgsfield | Downloading to %s", output_path)
    resp = requests.get(url, stream=True, headers=headers, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Higgsfield | Download complete")
    return output_path
