"""Stage 2b — Resource Finder.

Uses Claude to generate topic-specific resources, then validates each URL
with a live HTTP check before including it in the pinned comment.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import requests as _requests

from utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "⚠️ This content is for entertainment purposes only and does not constitute "
    "financial, legal, or medical advice. Always consult a qualified professional."
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ACG-resource-checker/1.0)"}

RESOURCE_TOOL = {
    "name": "submit_resources",
    "description": "Submit a list of real, specific resources directly relevant to the video topic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resources": {
                "type": "array",
                "description": "10–12 candidate resources. More candidates means more survive URL validation.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["title", "url"],
                },
                "minItems": 10,
                "maxItems": 12,
            }
        },
        "required": ["resources"],
    },
}


def _check_url(url: str, timeout: int = 8) -> bool:
    """Return True if the URL is reachable and returns a non-error status."""
    try:
        resp = _requests.head(url, timeout=timeout, headers=_HEADERS, allow_redirects=True)
        if resp.status_code == 405:
            # HEAD not supported — fall back to GET without downloading the body
            resp = _requests.get(url, timeout=timeout, headers=_HEADERS, stream=True)
            resp.close()
        return resp.status_code < 400
    except Exception:
        return False


def _validate(resources: list[dict], max_keep: int = 5) -> list[dict]:
    """Check all URLs in parallel; return only those that are reachable."""
    valid: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_check_url, r["url"]): r for r in resources}
        for future in as_completed(futures):
            r = futures[future]
            if future.result():
                valid.append(r)
                logger.info("Stage 2b | URL OK: %s", r["url"])
            else:
                logger.warning("Stage 2b | URL unreachable, dropping: %s", r["url"])
    return valid[:max_keep]


def run(stage2_out: dict, dry_run: bool = False) -> list[dict]:
    """Generate topic-specific resources via Claude, validate URLs, return working ones."""
    topic: str = stage2_out.get("topic", "")

    if dry_run:
        logger.info("Stage 2b | DRY-RUN — skipping resource generation")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("Stage 2b | ANTHROPIC_API_KEY not set — skipping resources")
        return []

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            tools=[RESOURCE_TOOL],
            tool_choice={"type": "any"},
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    "Suggest 10–12 real online resources that directly address this topic. "
                    "Prioritise official government pages (.gov), major nonprofits (aarp.org), "
                    "and well-known financial guidance sites. "
                    "Link to specific pages about this exact topic — not generic homepages. "
                    "Provide more candidates than needed because some URLs will be checked live "
                    "and dropped if unreachable — aim for at least 5 to survive."
                ),
            }],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == "submit_resources":
                candidates = block.input.get("resources", [])
                logger.info("Stage 2b | Claude suggested %d resources — validating URLs", len(candidates))
                valid = _validate(candidates)
                logger.info("Stage 2b | %d/%d resources passed URL validation", len(valid), len(candidates))
                return valid

        logger.warning("Stage 2b | Claude returned no resources")
        return []

    except Exception as exc:
        logger.warning("Stage 2b | Resource generation failed (%s) — skipping resources", exc)
        return []


def format_pinned_comment(resources: list[dict]) -> str:
    """Build the pinned comment: disclaimer → resource list → disclaimer."""
    lines = [DISCLAIMER, ""]
    if resources:
        lines.append("📚 Free resources on this topic:")
        for r in resources:
            lines.append(f"• {r['title']}")
            lines.append(f"  {r['url']}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
