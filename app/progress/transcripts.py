"""Deterministic parsing for timestamped and plain-text meeting transcripts."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
VOICE_RE = re.compile(r"^<v(?:\.[^ >]+)?(?:\s+([^>]+))?>(.*)$", re.DOTALL)
SPEAKER_RE = re.compile(r"^([^:\n]{1,80}):\s+(.+)$", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class TranscriptCue:
    index: int
    start_seconds: float | None
    end_seconds: float | None
    speaker: str | None
    text: str
    language: str
    source_locator: str


def timestamp_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def detect_language(text: str) -> str:
    has_ja = bool(JAPANESE_RE.search(text))
    has_en = bool(LATIN_RE.search(text))
    if has_ja and has_en:
        return "mixed"
    if has_ja:
        return "ja"
    if has_en:
        return "en"
    return "unknown"


def clean_cue_text(raw: str) -> tuple[str | None, str]:
    value = html.unescape(raw.strip())
    speaker: str | None = None
    voice = VOICE_RE.match(value)
    if voice:
        speaker = (voice.group(1) or "").strip() or None
        value = voice.group(2)
    value = TAG_RE.sub("", value).strip()
    speaker_match = SPEAKER_RE.match(value)
    if not speaker and speaker_match:
        speaker = speaker_match.group(1).strip()
        value = speaker_match.group(2).strip()
    value = re.sub(r"\s+", " ", value).strip()
    return speaker, value


def parse_timestamped(text: str) -> list[TranscriptCue]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", normalized)
    cues: list[TranscriptCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if TIMING_RE.search(line)), None)
        if timing_index is None:
            continue
        timing = TIMING_RE.search(lines[timing_index])
        if not timing:
            continue
        speaker, cue_text = clean_cue_text("\n".join(lines[timing_index + 1 :]))
        if not cue_text:
            continue
        index = len(cues) + 1
        cues.append(
            TranscriptCue(
                index=index,
                start_seconds=timestamp_seconds(timing.group("start")),
                end_seconds=timestamp_seconds(timing.group("end")),
                speaker=speaker,
                text=cue_text,
                language=detect_language(cue_text),
                source_locator=f"cue:{index}",
            )
        )
    return cues


def parse_plain_text(text: str) -> list[TranscriptCue]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    cues: list[TranscriptCue] = []
    for paragraph in paragraphs:
        speaker, cue_text = clean_cue_text(paragraph)
        if not cue_text:
            continue
        index = len(cues) + 1
        cues.append(
            TranscriptCue(
                index=index,
                start_seconds=None,
                end_seconds=None,
                speaker=speaker,
                text=cue_text,
                language=detect_language(cue_text),
                source_locator=f"paragraph:{index}",
            )
        )
    return cues


def parse_transcript(text: str, filename: str) -> list[TranscriptCue]:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in {"vtt", "srt"} or TIMING_RE.search(text):
        return parse_timestamped(text)
    return parse_plain_text(text)
