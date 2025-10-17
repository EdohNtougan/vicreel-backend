#!/usr/bin/env python3
# sync_aliases.py - Génère config/resolved_aliases.json à partir de RAW et model speakers
from TTS.api import TTS
import json
from pathlib import Path
import logging
import unicodedata
import difflib
import re
logger = logging.getLogger("vicreel-sync")
logging.basicConfig(level=logging.INFO)

RAW_FILE = "config/speaker_aliases.json"
RESOLVED_FILE = "config/resolved_aliases.json"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match(raw_key: str, available: list) -> str or None:
    norm_raw = _normalize_text(raw_key)
    norm_map = {_normalize_text(a): a for a in available}
    if norm_raw in norm_map:
        return norm_map[norm_raw]
    for a in available:
        if norm_raw in _normalize_text(a) or _normalize_text(a) in norm_raw:
            return a
    candidates = list(norm_map.keys())
    match = difflib.get_close_matches(norm_raw, candidates, n=1, cutoff=0.75)
    if match:
        return norm_map[match[0]]
    return None

def main():
    tts = TTS(DEFAULT_MODEL)
    available = getattr(tts, "speakers", []) or getattr(tts, "voices", []) or []
    logger.info(f"Model speakers: {len(available)}")
    raw_aliases = {}
    if Path(RAW_FILE).exists():
        raw_aliases = json.load(Path(RAW_FILE).open("r", encoding="utf-8"))
    resolved = {}
    for raw_key, display in raw_aliases.items():
        matched = fuzzy_match(raw_key, available)
        if matched:
            resolved[matched] = display
            logger.info(f"Matched '{raw_key}' -> '{matched}' as '{display}'")
        else:
            logger.warning(f"No match for '{raw_key}'")
    tmp = Path(RESOLVED_FILE).with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    tmp.replace(RESOLVED_FILE)
    logger.info(f"Persisted {len(resolved)} resolved aliases to {RESOLVED_FILE}")

if __name__ == "__main__":
    main()
