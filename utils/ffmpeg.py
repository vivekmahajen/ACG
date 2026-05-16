import json
import subprocess
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def probe(file_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "json",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no video streams")
    return streams[0]


def validate_video(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    if path.stat().st_size < 500 * 1024:
        raise ValueError(f"Video file too small ({path.stat().st_size} bytes) — likely corrupt")

    info = probe(file_path)
    duration = float(info.get("duration", 0))
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))

    if not (5 <= duration <= 75):
        raise ValueError(f"Video duration {duration:.1f}s outside acceptable range 5–75s")
    if width < 720 or height < 1280:
        raise ValueError(f"Video resolution {width}x{height} below minimum 720x1280")

    logger.info("Video validated: %.1fs, %dx%d, %.1f KB", duration, width, height, path.stat().st_size / 1024)


def trim_to_50s(input_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-t", "50", "-c", "copy", output_path]
    _run(cmd, "trim_to_50s")


def scale_to_1080x1920(input_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", "scale=1080:1920", output_path]
    _run(cmd, "scale_to_1080x1920")


def add_fade(input_path: str, output_path: str, duration: float = 30.0) -> None:
    fade_out_start = max(0.0, duration - 0.5)
    vf = f"fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out_start:.2f}:d=0.5"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", vf, output_path]
    _run(cmd, "add_fade")


def post_process(input_path: str, output_path: str) -> None:
    info = probe(input_path)
    duration = float(info.get("duration", 30.0))
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))

    current = input_path
    tmp1 = input_path + ".trim.mp4"
    tmp2 = input_path + ".scaled.mp4"
    tmps: list[str] = []

    if duration > 50:
        logger.info("Trimming video to 50s")
        trim_to_50s(current, tmp1)
        current = tmp1
        tmps.append(tmp1)
        duration = 50.0

    if width < 1080 or height < 1920:
        logger.info("Scaling video to 1080x1920")
        scale_to_1080x1920(current, tmp2)
        current = tmp2
        tmps.append(tmp2)

    logger.info("Adding fade in/out")
    add_fade(current, output_path, duration)

    import os
    for t in tmps:
        try:
            os.remove(t)
        except OSError:
            pass


def concatenate(clips: list[str], output_path: str) -> str:
    import os as _os, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clips:
            f.write(f"file '{_os.path.abspath(clip)}'\n")
        list_file = f.name
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]
        _run(cmd, "concatenate")
    finally:
        try:
            _os.remove(list_file)
        except OSError:
            pass
    return output_path


_FONT_CANDIDATES = [
    # Windows
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/calibri.ttf",
    r"C:/Windows/Fonts/verdana.ttf",
    # Linux / GitHub Actions
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _find_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def add_ticker(input_path: str, output_path: str,
               text: str = "FOR ENTERTAINMENT PURPOSES ONLY - NOT FINANCIAL ADVICE") -> str:
    """Burn a scrolling disclaimer ticker along the bottom of the video."""
    import os as _os
    import tempfile

    font_path = _find_font()
    if font_path is None:
        raise RuntimeError("No suitable font found for drawtext ticker")

    scroll_text = f"  {text}  |  {text}  |  {text}  "

    # Write text to a temp file — avoids ALL ffmpeg drawtext escaping issues on Windows and Linux
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        tmp.write(scroll_text)
        tmp.close()
        text_file_arg = tmp.name.replace("\\", "/").replace(":", "\\:")
        font_arg = font_path.replace("\\", "/").replace(":", "\\:")
        drawtext = (
            f"drawtext=fontfile=’{font_arg}’:textfile=’{text_file_arg}’:"
            "fontsize=22:fontcolor=white:"
            "box=1:boxcolor=black@0.75:boxborderw=6:"
            "x=w-80*t:y=h-50"
        )
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", drawtext, "-c:a", "copy", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg add_ticker failed: {result.stderr[-800:]}")
        logger.info("Ticker added: %s", output_path)
        return output_path
    finally:
        try:
            _os.remove(tmp.name)
        except OSError:
            pass


def _run(cmd: list[str], name: str) -> None:
    logger.info("ffmpeg %s: %s", name, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {name} failed: {result.stderr[-500:]}")
