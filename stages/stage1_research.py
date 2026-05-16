"""Stage 1 — YouTube Research Agent.

Finds the top N channels in the configured niche and collects recent videos
from the top channels_to_analyse channels by subscriber count.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import isodate
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from utils import config as cfg
from utils import database as db
from utils.logger import get_logger

logger = get_logger(__name__)

SEARCH_QUALIFIERS = ["tips", "2025", "beginners", "explained"]
BASE_URL = "https://www.googleapis.com/youtube/v3"


def _youtube_client():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError("YOUTUBE_API_KEY is not set")
    return build("youtube", "v3", developerKey=api_key)


def _parse_duration(duration_str: str) -> int:
    try:
        return int(isodate.parse_duration(duration_str).total_seconds())
    except Exception:
        return 0


def _engagement_rate(likes: int, comments: int, views: int) -> float:
    if views == 0:
        return 0.0
    return (likes + comments) / views


def _search_channels(youtube, query: str, max_results: int = 50) -> list[dict]:
    """Search for channels matching query. Returns list with id and snippet."""
    channels: list[dict] = []
    try:
        response = youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            order="viewCount",
            maxResults=min(max_results, 50),
        ).execute()
        for item in response.get("items", []):
            channels.append({
                "channel_id": item["snippet"]["channelId"],
                "channel_name": item["snippet"]["channelTitle"],
            })
    except HttpError as e:
        _handle_http_error(e, "search_channels")
    return channels


def _get_channel_stats(youtube, channel_ids: list[str]) -> dict[str, dict]:
    """Retrieve statistics for a list of channel IDs. Returns map channel_id → stats."""
    if not channel_ids:
        return {}
    stats: dict[str, dict] = {}
    try:
        response = youtube.channels().list(
            part="statistics,snippet",
            id=",".join(channel_ids),
            maxResults=50,
        ).execute()
        for item in response.get("items", []):
            cid = item["id"]
            s = item.get("statistics", {})
            stats[cid] = {
                "channel_id": cid,
                "channel_name": item["snippet"]["title"],
                "subscriber_count": int(s.get("subscriberCount", 0)),
                "view_count": int(s.get("viewCount", 0)),
            }
    except HttpError as e:
        _handle_http_error(e, "get_channel_stats")
    return stats


def _get_recent_videos(youtube, channel_id: str, days_lookback: int, max_results: int) -> list[str]:
    """Return video IDs for recent uploads from a channel.

    Uses playlistItems.list (1 quota unit) on the channel's uploads playlist
    instead of search.list (100 quota units). Falls back to search on failure.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days_lookback)

    # Uploads playlist ID: replace leading "UC" with "UU" (always valid for UC channels)
    if channel_id.startswith("UC"):
        playlist_id = "UU" + channel_id[2:]
        video_ids = _playlist_video_ids(youtube, playlist_id, since, max_results, channel_id)
        if video_ids is not None:
            return video_ids

    # Fallback: search.list (100 units) — only used if playlist lookup failed
    logger.warning("Stage 1 | Falling back to search.list for channel %s (costs 100 quota units)", channel_id)
    video_ids = []
    try:
        response = youtube.search().list(
            part="id",
            channelId=channel_id,
            type="video",
            order="date",
            publishedAfter=since.isoformat(),
            maxResults=min(max_results, 50),
        ).execute()
        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])
    except HttpError as e:
        _handle_http_error(e, f"get_recent_videos:{channel_id}")
    return video_ids


def _playlist_video_ids(youtube, playlist_id: str, since: datetime, max_results: int, channel_id: str) -> list[str] | None:
    """Fetch video IDs from a playlist, filtering to items published after `since`.

    Returns None if the playlist doesn't exist (caller should fall back to search).
    Costs 1 quota unit per page fetched.
    """
    video_ids: list[str] = []
    try:
        response = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=playlist_id,
            maxResults=min(max_results * 3, 50),  # fetch extra so date filter has room
        ).execute()
        for item in response.get("items", []):
            video_id = item.get("contentDetails", {}).get("videoId", "")
            if not video_id:
                continue
            published_str = item.get("snippet", {}).get("publishedAt", "")
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    if published_at < since:
                        continue
                except Exception:
                    pass
            video_ids.append(video_id)
            if len(video_ids) >= max_results:
                break
        return video_ids
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning("Stage 1 | Uploads playlist %s not found for channel %s", playlist_id, channel_id)
            return None
        _handle_http_error(e, f"playlist_video_ids:{channel_id}")
        return []


def _get_video_details(youtube, video_ids: list[str]) -> list[dict]:
    """Retrieve full stats+snippet for a list of video IDs."""
    if not video_ids:
        return []
    videos: list[dict] = []
    try:
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids),
        ).execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            cd = item.get("contentDetails", {})

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            duration_secs = _parse_duration(cd.get("duration", "PT0S"))

            videos.append({
                "title": snippet.get("title", ""),
                "video_id": item["id"],
                "channel_name": snippet.get("channelTitle", ""),
                "channel_id": snippet.get("channelId", ""),
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "duration_seconds": duration_secs,
                "tags": snippet.get("tags", []),
                "description_snippet": snippet.get("description", "")[:300],
                "published_at": snippet.get("publishedAt", ""),
                "engagement_rate": _engagement_rate(likes, comments, views),
            })
    except HttpError as e:
        _handle_http_error(e, "get_video_details")
    return videos


def _handle_http_error(e: HttpError, context: str) -> None:
    status = e.resp.status
    if status == 403:
        logger.error("[%s] YouTube quota exceeded (403). Stopping.", context)
        raise SystemExit("YouTube quota exceeded") from e
    if status == 404:
        logger.warning("[%s] Resource not found (404) — skipping.", context)
        return
    if status == 429:
        logger.warning("[%s] Rate limited (429) — waiting 60s.", context)
        time.sleep(60)
        return
    raise e


def run(dry_run: bool = False) -> dict:
    """Execute Stage 1. Returns dict with channels and videos lists."""
    conf = cfg.load_config()
    niche: str = conf["niche"]
    min_subs: int = conf["min_subscriber_count"]
    top_n: int = conf["top_channels_to_scan"]
    analyse_n: int = conf["channels_to_analyse"]
    vids_per_channel: int = conf["videos_per_channel"]
    days_lookback: int = conf["days_lookback"]

    logger.info("Stage 1 | niche=%s | min_subs=%d | top_n=%d", niche, min_subs, top_n)

    if not dry_run:
        cached = db.get_stage1_cache(niche)
        if cached:
            logger.info("Stage 1 | Using cached video data (%d videos) — skipping YouTube API calls",
                        len(cached.get("videos", [])))
            return cached

    # Check 24-hour channel cache first
    cached = db.get_cached_channels(niche)
    if cached:
        logger.info("Stage 1 | Using cached channels (%d entries)", len(cached))
        top_channels = sorted(cached, key=lambda c: c["subscriber_count"], reverse=True)[:analyse_n]
    else:
        if dry_run:
            logger.info("Stage 1 | DRY-RUN — returning mock channel data")
            return _mock_stage1_output()

        youtube = _youtube_client()

        # Build search queries
        queries = [niche] + [f"{niche} {q}" for q in SEARCH_QUALIFIERS]
        all_channel_ids: set[str] = set()
        all_channel_snippets: dict[str, str] = {}

        for query in queries:
            if len(all_channel_ids) >= top_n * 2:
                break
            logger.info("Stage 1 | Searching channels: %r", query)
            results = _search_channels(youtube, query, max_results=top_n)
            for r in results:
                all_channel_ids.add(r["channel_id"])
                all_channel_snippets[r["channel_id"]] = r["channel_name"]

        # Batch fetch channel stats (max 50 per call)
        ids_list = list(all_channel_ids)
        stats_map: dict[str, dict] = {}
        for i in range(0, len(ids_list), 50):
            batch = ids_list[i:i + 50]
            stats_map.update(_get_channel_stats(youtube, batch))

        # Filter by min subscribers and sort
        qualified = [
            s for s in stats_map.values()
            if s["subscriber_count"] >= min_subs
        ]
        qualified.sort(key=lambda c: c["subscriber_count"], reverse=True)
        top_channels = qualified[:analyse_n]

        if not top_channels:
            logger.warning("Stage 1 | No channels found matching criteria — broadening search")
            raise RuntimeError("Stage 1 failed: zero qualifying channels found")

        db.upsert_channel_cache(top_channels, niche)
        logger.info("Stage 1 | %d channels cached for niche=%s", len(top_channels), niche)

    logger.info("Stage 1 | Collecting videos from %d channels", len(top_channels))

    if dry_run:
        return _mock_stage1_output()

    youtube = _youtube_client()
    all_videos: list[dict] = []

    for ch in top_channels:
        logger.info("Stage 1 | Channel: %s (%s) subs=%d",
                    ch["channel_name"], ch["channel_id"], ch["subscriber_count"])
        video_ids = _get_recent_videos(youtube, ch["channel_id"], days_lookback, vids_per_channel)
        if not video_ids:
            logger.info("Stage 1 | No recent videos for channel %s", ch["channel_id"])
            continue
        videos = _get_video_details(youtube, video_ids)
        all_videos.extend(videos)
        logger.info("Stage 1 | Collected %d videos from %s", len(videos), ch["channel_name"])

    logger.info("Stage 1 | Total videos collected: %d", len(all_videos))
    result = {
        "channels": top_channels,
        "videos": all_videos,
    }
    db.save_stage1_cache(niche, result)
    logger.info("Stage 1 | Video data cached for niche=%s (reused by subsequent runs today)", niche)
    return result


def _mock_stage1_output() -> dict:
    """Return deterministic mock data for dry-run mode."""
    return {
        "channels": [
            {"channel_id": "UC_mock1", "channel_name": "Finance Wizards", "subscriber_count": 850000},
            {"channel_id": "UC_mock2", "channel_name": "Millennial Money", "subscriber_count": 620000},
        ],
        "videos": [
            {
                "title": "I saved $10,000 in 6 months — here's exactly how",
                "video_id": "mock_vid_001",
                "channel_name": "Finance Wizards",
                "channel_id": "UC_mock1",
                "view_count": 1_200_000,
                "like_count": 48000,
                "comment_count": 3200,
                "duration_seconds": 540,
                "tags": ["saving money", "personal finance", "budgeting"],
                "description_snippet": "In this video I break down exactly how I saved $10k...",
                "published_at": "2026-05-05T10:00:00Z",
                "engagement_rate": 0.043,
            },
            {
                "title": "Why your coffee habit costs you $1,200 a year",
                "video_id": "mock_vid_002",
                "channel_name": "Millennial Money",
                "channel_id": "UC_mock2",
                "view_count": 980_000,
                "like_count": 52000,
                "comment_count": 4100,
                "duration_seconds": 360,
                "tags": ["coffee", "saving money", "latte factor"],
                "description_snippet": "The latte factor is real and I did the math...",
                "published_at": "2026-05-08T14:00:00Z",
                "engagement_rate": 0.057,
            },
            {
                "title": "5 no-spend challenge tips that actually work",
                "video_id": "mock_vid_003",
                "channel_name": "Finance Wizards",
                "channel_id": "UC_mock1",
                "view_count": 750_000,
                "like_count": 31000,
                "comment_count": 2800,
                "duration_seconds": 420,
                "tags": ["no spend", "challenge", "saving", "frugal"],
                "description_snippet": "The no-spend challenge changed my life...",
                "published_at": "2026-05-10T09:00:00Z",
                "engagement_rate": 0.045,
            },
        ],
    }
