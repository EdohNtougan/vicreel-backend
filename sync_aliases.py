#!/usr/bin/env python3
# sync_aliases.py - Génère config/resolved_aliases.json à partir de RAW et model speakers
from TTS.api import TTS
import json
from pathlib import Path
import logging
import unicodedata
import difflib
import re
import time

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

def _make_canonical_id(s: str) -> str:
    """Génère un id 'clean' : lower + underscores simples (ari_male)."""
    if not isinstance(s, str):
        return "speaker"
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Remplacer tout ce qui n'est pas alnum par underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # collapse multiple underscores to single
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if not s:
        s = "speaker"
    return s

def fuzzy_match(raw_key: str, available: list) -> str or None:
    """Retourne l'élément d'available le plus proche (original), ou None."""
    if not raw_key or not available:
        return None
    norm_raw = _normalize_text(raw_key)
    # map normalized available -> original
    norm_map = {_normalize_text(a): a for a in available}
    if norm_raw in norm_map:
        return norm_map[norm_raw]
    # substring / contains
    for a in available:
        if norm_raw in _normalize_text(a) or _normalize_text(a) in norm_raw:
            return a
    # fuzzy
    candidates = list(norm_map.keys())
    match = difflib.get_close_matches(norm_raw, candidates, n=1, cutoff=0.75)
    if match:
        return norm_map[match[0]]
    return None

def main():
    logger.info("Loading model: %s", DEFAULT_MODEL)
    tts = TTS(DEFAULT_MODEL)

    # Warmup — déclencher l'initialisation complète (ca aide à remplir speakers/voices)
    logger.info("Warmup du modèle (petit passage pour forcer l'initialisation)...")
    try:
        # On essaie d'appeler tts d'une manière sûre. Ce n'est pas pour produire un output.
        # Si l'API change, enlevez ou adaptez ce call.
        tts.tts("warmup", speaker=None, language="en")
    except Exception as e:
        logger.debug("Warmup call raised (non bloquant): %s", e)
        # small sleep to give model time
        time.sleep(1.0)

    # Récupérer la liste de speakers/voices après warmup
    available = getattr(tts, "speakers", []) or getattr(tts, "voices", []) or []
    logger.info("Model speakers discovered: %d", len(available))
    # Garde-fou : si liste vide ou manifestement incomplète, on arrête
    if not available or len(available) < 5:
        logger.error("La liste des speakers est vide ou trop petite (%d). Abandon.", len(available))
        return

    raw_aliases = {}
    p_raw = Path(RAW_FILE)
    if p_raw.exists():
        try:
            raw_aliases = json.load(p_raw.open("r", encoding="utf-8"))
        except Exception:
            logger.exception("Impossible de lire %s", RAW_FILE)
            return
    else:
        logger.warning("Fichier raw d'aliases non trouvé: %s", RAW_FILE)
        return

    resolved = {}
    used_real_ids = set()
    for raw_key, display in raw_aliases.items():
        matched = fuzzy_match(raw_key, available)
        if matched:
            # Si matched ressemble déjà à un real id (pattern underscore/lower), on le garde,
            # sinon on génère une canonical id à partir de la valeur fournie par le modèle.
            if re.match(r"^[a-z0-9_]+$", matched):
                real_id = matched
            else:
                real_id = _make_canonical_id(matched)
            # éviter collisions : si collision, suffixer un index
            base = real_id
            i = 1
            while real_id in used_real_ids:
                i += 1
                real_id = f"{base}_{i}"
            used_real_ids.add(real_id)
            resolved[real_id] = display
            logger.info("Matched '%s' -> model value '%s' => real_id '%s' as '%s'", raw_key, matched, real_id, display)
        else:
            logger.warning("No match for '%s' (display='%s')", raw_key, display)

    # Persist atomically
    tmp = Path(RESOLVED_FILE).with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)
        tmp.replace(RESOLVED_FILE)
        logger.info("Persisted %d resolved aliases to %s", len(resolved), RESOLVED_FILE)
    except Exception:
        logger.exception("Failed to persist resolved aliases")

if __name__ == "__main__":
    main()