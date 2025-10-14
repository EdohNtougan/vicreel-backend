#!/usr/bin/env python3
"""
generate_aliases.py
Génère config/speaker_aliases.json mappant speaker_id -> AliasName

Usage:
  python3 scripts/generate_aliases.py           # génère si absent (utilise modèle local)
  python3 scripts/generate_aliases.py --force   # régénère et écrase
  python3 scripts/generate_aliases.py --out path/to/file.json
"""
import os
import json
import argparse
from TTS.api import TTS

# Liste d'alias (56). Personnalise si tu veux.
ALIAS_POOL = [
 "Ari", "Bailey", "Cleo", "Dara", "Eden", "Fiora", "Gabe", "Hana", "Iris", "Jory",
 "Kai", "Lina", "Miro", "Nova", "Ona", "Pax", "Quin", "Rosa", "Sage", "Tess",
 "Uma", "Vera", "Wes", "Xena", "Yara", "Zane", "Maya", "Noah", "Luca", "Oliv",
 "Pru", "Rey", "Sven", "Talia", "Uri", "Vito", "Wynn", "Ximena", "Yule", "Zeke",
 "Amber", "Beno", "Cira", "Dino", "Elsa", "Fynn", "Gwen", "Hiro", "Ivo", "Juno",
 "Kara", "Lior", "Mina", "Niko"
]

DEFAULT_OUTPUT = "config/speaker_aliases.json"

def get_speakers_from_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
    print(f"[generate_aliases] Loading model {model_name} (this may take time)...")
    tts = TTS(model_name, progress_bar=False, gpu=False)
    speakers = getattr(tts, "speakers", None) or getattr(tts, "voices", None) or []
    print(f"[generate_aliases] Found {len(speakers)} speakers")
    return speakers

def generate_aliases_for_speakers(speakers):
    aliases = {}
    speakers_sorted = sorted(speakers)
    pool = ALIAS_POOL.copy()
    i = 0
    for sp in speakers_sorted:
        alias = pool[i % len(pool)]
        # ensure uniqueness: append index if alias already used
        if alias in aliases.values():
            alias = f"{alias}-{i}"
        aliases[sp] = alias
        i += 1
    return aliases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite if output exists")
    args = parser.parse_args()

    out_path = args.out
    if os.path.exists(out_path) and not args.force:
        print(f"[generate_aliases] {out_path} already exists. Use --force to overwrite.")
        return

    speakers = get_speakers_from_model(args.model)
    if not speakers:
        print("[generate_aliases] Warning: no speakers found in model. Aborting.")
        return

    aliases = generate_aliases_for_speakers(speakers)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=2)

    print(f"[generate_aliases] Wrote {len(aliases)} aliases to {out_path}")

if __name__ == "__main__":
    main()
