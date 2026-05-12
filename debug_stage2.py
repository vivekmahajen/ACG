"""Run this to see exactly what Claude returns for Stage 2."""
import json, os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = "Return ONLY valid JSON. No commentary. No markdown. No preamble."

USER = """Niche: personal finance
Domain focus: money saving tips for millennials
Today's date: 2026-05-12

Recently published videos to analyse:
- Title: I saved $10,000 in 6 months | Views: 1,200,000 | Likes: 48000 | Engagement: 0.043

Return ONLY valid JSON with these exact keys:
{
  "topic": "...",
  "why_this_topic": "...",
  "hook": "...",
  "key_visual_idea": "...",
  "target_emotion": "...",
  "estimated_watch_through_rate": "...",
  "competitor_angle": "..."
}"""

msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=600,
    system=SYSTEM,
    messages=[{"role": "user", "content": USER}],
)

raw = msg.content[0].text
print("=== RAW RESPONSE ===")
print(repr(raw[:500]))
print()
print("=== DISPLAY ===")
print(raw[:500])
