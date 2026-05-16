"""Stage 5 — YouTube Publisher.

Uploads the validated MP4 to the authenticated YouTube channel using the
resumable upload API, then logs results and cleans up.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httplib2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from utils import config as cfg
from utils import database as db
from utils.logger import get_logger

logger = get_logger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for commentThreads.insert
]
TOKEN_FILE = "token.json"

CATEGORY_ID_MAP = {
    "Education": "27",
    "Entertainment": "24",
    "HowTo": "26",
    "News": "25",
    "Finance": "27",
    "Health": "26",
    "Technology": "28",
    "Lifestyle": "22",
}


def _get_authenticated_service():
    import google.auth.exceptions as _gauth_exc
    import google.auth.transport.requests as grequests
    creds: Credentials | None = None

    if Path(TOKEN_FILE).exists():
        with open(TOKEN_FILE) as f:
            token_data = json.load(f)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

    # Always try to refresh if we have a refresh token — catches expired access tokens
    if creds and creds.refresh_token and (not creds.valid or creds.expired):
        try:
            logger.info("Stage 5 | Refreshing OAuth token")
            creds.refresh(grequests.Request())
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            logger.info("Stage 5 | Token refreshed and saved")
        except _gauth_exc.RefreshError as e:
            logger.warning("Stage 5 | Token refresh failed: %s", e)
            creds = None

    if not creds or not creds.valid:
        # In CI there is no browser — fail immediately with a clear message
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            raise RuntimeError(
                "Stage 5 | token.json is missing or invalid and cannot be refreshed in CI. "
                "Run 'python reauth.py' locally, then update TOKEN_JSON_B64 in GitHub Secrets."
            )
        client_secret_path = os.environ.get("YOUTUBE_CLIENT_SECRET", "client_secret.json")
        if not Path(client_secret_path).exists():
            raise FileNotFoundError(
                f"OAuth client secret not found: {client_secret_path}."
            )
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        logger.info("Stage 5 | OAuth token saved to %s", TOKEN_FILE)

    return build("youtube", "v3", credentials=creds)


def _map_category_id(category_str: str) -> str:
    return CATEGORY_ID_MAP.get(category_str, "27")


def _upload(youtube, video_file: str, stage3: dict, stage2: dict, conf: dict,
            resources: list | None = None) -> dict:
    from stages.stage2b_resources import format_pinned_comment
    category_id = stage3.get("category_id") or _map_category_id(stage3.get("category", "Finance"))
    privacy = conf.get("privacy", "public")

    base_description = stage3.get("description", "")
    disclaimer_block = format_pinned_comment(resources or [])
    full_description = f"{base_description}\n\n{disclaimer_block}" if base_description else disclaimer_block

    body = {
        "snippet": {
            "title": stage3["title"],
            "description": full_description,
            "tags": stage3["tags"],
            "categoryId": category_id,
            "defaultLanguage": stage3.get("language", "en"),
            "defaultAudioLanguage": stage3.get("language", "en"),
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "madeForKids": False,
        },
    }

    media = MediaFileUpload(video_file, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    logger.info("Stage 5 | Starting resumable upload: %s", video_file)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("Stage 5 | Upload progress: %d%%", int(status.progress() * 100))

    video_id: str = response["id"]
    video_url = f"https://youtu.be/{video_id}"
    logger.info("Stage 5 | Upload complete: %s", video_url)
    return {"video_id": video_id, "video_url": video_url}


def _retry_upload(youtube, video_file: str, stage3: dict, stage2: dict, conf: dict,
                  max_retries: int = 3, resources: list | None = None) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            return _upload(youtube, video_file, stage3, stage2, conf, resources=resources)
        except HttpError as e:
            status = e.resp.status
            if status == 403:
                logger.error("Stage 5 | YouTube quota exceeded (403) — cannot upload")
                raise
            if status == 400:
                reason = ""
                try:
                    import json as _json
                    details = _json.loads(e.content)
                    reason = details.get("error", {}).get("errors", [{}])[0].get("reason", "")
                except Exception:
                    pass
                if reason == "uploadLimitExceeded":
                    logger.error(
                        "Stage 5 | YouTube upload limit reached — verify your channel at "
                        "YouTube Studio → Settings → Channel → Feature eligibility"
                    )
                logger.error("Stage 5 | Bad request (400) — check metadata: %s", e)
                raise
            if status in (500, 503):
                wait = 60 * attempt
                logger.warning("Stage 5 | Server error %d attempt %d/%d — waiting %ds", status, attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
                raise
        except (httplib2.HttpLib2Error, TimeoutError) as e:
            wait = 30 * attempt
            logger.warning("Stage 5 | Network error attempt %d/%d: %s — waiting %ds", attempt, max_retries, e, wait)
            if attempt < max_retries:
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Stage 5 | All upload attempts exhausted")


def run(stage2_output: dict, stage3_output: dict, stage4_output: dict,
        run_id: int, dry_run: bool = False, resources: list | None = None) -> dict:
    """Execute Stage 5. Returns dict with video_id and video_url."""
    conf = cfg.load_config()

    video_file: str = stage4_output["video_file"]
    logger.info("Stage 5 | Publishing: %s", video_file)

    if dry_run:
        logger.info("Stage 5 | DRY-RUN — skipping upload")
        return _mock_stage5_output(stage3_output)

    youtube = _get_authenticated_service()

    result = _retry_upload(youtube, video_file, stage3_output, stage2_output, conf,
                           max_retries=conf.get("max_retries", 3), resources=resources or [])

    # Post pinned comment: disclaimer + resource list (falls back to generated pinned_comment)
    comment_text = _build_comment(stage3_output, resources or [])
    if comment_text:
        _post_comment(youtube, result["video_id"], comment_text)

    # Post-upload housekeeping
    published_at = datetime.now(timezone.utc).isoformat()
    db.log_published_topic(stage2_output["topic"], published_at, result["video_url"])
    logger.info("Stage 5 | Topic logged to published_topics")

    if not conf.get("keep_local_video", False):
        try:
            os.remove(video_file)
            logger.info("Stage 5 | Deleted local video: %s", video_file)
        except OSError as e:
            logger.warning("Stage 5 | Could not delete video file: %s", e)

    return result


def _post_comment(youtube, video_id: str, text: str) -> None:
    body = {
        "snippet": {
            "videoId": video_id,
            "topLevelComment": {
                "snippet": {"textOriginal": text},
            },
        }
    }
    try:
        response = youtube.commentThreads().insert(part="snippet", body=body).execute()
        comment_id = response["snippet"]["topLevelComment"]["id"]
        logger.info("Stage 5 | Comment posted (id=%s) — pin it manually in YouTube Studio", comment_id)
    except HttpError as e:
        if e.resp.status == 403:
            logger.error(
                "Stage 5 | Comment posting forbidden (403) — delete token.json and re-authenticate "
                "to pick up the youtube.force-ssl scope."
            )
        else:
            logger.error("Stage 5 | Could not post comment (HTTP %s): %s", e.resp.status, e)
    except Exception as e:
        logger.error("Stage 5 | Comment posting failed unexpectedly: %s", e)


def _build_comment(stage3: dict, resources: list[dict]) -> str:
    from stages.stage2b_resources import format_pinned_comment, DISCLAIMER
    if resources:
        return format_pinned_comment(resources)
    # No resources available (e.g. resumed from stage 5) — still wrap with disclaimer
    pinned = stage3.get("pinned_comment", "").strip()
    parts = [DISCLAIMER]
    if pinned:
        parts += ["", pinned]
    parts += ["", DISCLAIMER]
    return "\n".join(parts)


def _mock_stage5_output(stage3: dict) -> dict:
    mock_id = "dQw4w9WgXcQ"
    return {
        "video_id": mock_id,
        "video_url": f"https://youtu.be/{mock_id}",
    }
