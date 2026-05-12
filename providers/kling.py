"""Kling AI text-to-video provider (official api.klingai.com)."""

import hashlib
import hmac
import os
import time

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.klingai.com"
POLL_INTERVAL = 10
MAX_WAIT = 600  # 10 minutes — Kling can be slow


def _make_jwt(access_key: str, secret_key: str) -> str:
    """Build a HS256 JWT for Kling API auth."""
    import base64, json as _json
    header = base64.urlsafe_b64encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(_json.dumps({
        "iss": access_key,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }).encode()).rstrip(b"=")
    msg = header + b"." + payload
    sig = base64.urlsafe_b64encode(
        hmac.new(secret_key.encode(), msg, hashlib.sha256).digest()
    ).rstrip(b"=")
    return (msg + b"." + sig).decode()


def generate(prompt: str, duration: int = 10, output_path: str = "generated_videos/output.mp4") -> str:
    access_key = os.environ.get("KLING_ACCESS_KEY")
    secret_key = os.environ.get("KLING_SECRET_KEY")

    # Also accept a single KLING_API_KEY in "access_key:secret_key" format for convenience
    if not access_key or not secret_key:
        combined = os.environ.get("KLING_API_KEY", "")
        if ":" in combined:
            access_key, secret_key = combined.split(":", 1)

    if not access_key or not secret_key:
        raise EnvironmentError(
            "Kling credentials not set. Set KLING_ACCESS_KEY and KLING_SECRET_KEY "
            "(or KLING_API_KEY=access_key:secret_key)"
        )

    token = _make_jwt(access_key, secret_key)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # duration must be "5" or "10" as a string per Kling API
    duration_str = "10" if duration >= 10 else "5"

    payload = {
        "model_name": "kling-v1-6",
        "prompt": prompt,
        "duration": duration_str,
        "aspect_ratio": "9:16",
        "cfg_scale": 0.5,
        "mode": "std",
    }

    logger.info("Kling | Submitting generation request")
    for attempt in range(3):
        resp = requests.post(f"{BASE_URL}/v1/videos/text2video", json=payload, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning("Kling | 429 rate-limited — waiting %ds (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
            token = _make_jwt(access_key, secret_key)
            headers["Authorization"] = f"Bearer {token}"
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()
    data = resp.json()

    task_id = (
        data.get("data", {}).get("task_id")
        or data.get("task_id")
        or data.get("id")
    )
    if not task_id:
        raise RuntimeError(f"Kling did not return a task_id. Response: {data}")

    logger.info("Kling | task_id=%s — polling for completion", task_id)
    return _poll_and_download(task_id, access_key, secret_key, output_path)


def _poll_and_download(task_id: str, access_key: str, secret_key: str, output_path: str) -> str:
    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        token = _make_jwt(access_key, secret_key)
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(
            f"{BASE_URL}/v1/videos/text2video/{task_id}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        task_data = data.get("data", data)
        status = (task_data.get("task_status") or task_data.get("status", "")).lower()
        logger.info("Kling | task=%s status=%s elapsed=%ds", task_id, status, elapsed)

        if status in ("succeed", "succeeded", "completed", "finished"):
            works = task_data.get("task_result", {}).get("videos", [])
            video_url = None
            if works:
                video_url = works[0].get("url") or works[0].get("video_url")
            if not video_url:
                video_url = (
                    task_data.get("output", {}).get("video_url")
                    or task_data.get("video_url")
                    or task_data.get("url")
                )
            if not video_url:
                raise RuntimeError(f"Kling succeeded but no video_url in response: {data}")
            return _download(video_url, output_path)

        if status in ("failed", "error"):
            raise RuntimeError(f"Kling generation failed: {task_data.get('task_status_msg', data)}")

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
