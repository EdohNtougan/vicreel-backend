#!/usr/bin/env python3
# sync_aliases.py - Génère une carte d'alias simple et directe pour le worker.
import json
import re
import unicodedata
from pathlib import Path
import logging
from TTS.api import TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-sync")

# --- Configuration ---
RAW_ALIASES_FILE = Path("config/speaker_aliases.json")
# NOUVEAU : Le seul fichier de sortie, la carte d'alias directe.
SPEAKER_MAP_FILE = Path("config/speaker_map.json") 
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

def make_clean_alias(display_name: str) -> str:
    """Transforme un nom lisible en un ID simple et propre (ex: 'Bernice (female)' -> 'bernice_female')."""
    if not isinstance(display_name, str):
        return ""
    # Normalisation et minuscules
    s = display_name.strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    # Remplacer les caractères non alphanumériques par des underscores
    s = re.sub(r'[\s\(\)\[\]]+', '_', s)
    # Supprimer les underscores en trop
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def main():
    logger.info("Chargement du modèle TTS pour obtenir la liste des speakers...")
    try:
        tts = TTS(DEFAULT_MODEL)
        available_speakers = getattr(tts, "speakers", []) or []
        if not available_speakers:
            raise RuntimeError("La liste des speakers du modèle est vide.")
        logger.info("%d speakers trouvés dans le modèle.", len(available_speakers))
    except Exception as e:
        logger.exception("Impossible de charger le modèle TTS: %s", e)
        return

    if not RAW_ALIASES_FILE.exists():
        logger.error("Le fichier source des alias %s est introuvable.", RAW_ALIASES_FILE)
        return
        
    raw_aliases = json.loads(RAW_ALIASES_FILE.read_text(encoding="utf-8"))
    
    speaker_map = {}
    validated_aliases = set()

    logger.info("Génération de la carte d'alias à partir de %s...", RAW_ALIASES_FILE)
    
    # On crée un set des noms de speakers disponibles pour une recherche rapide
    available_speakers_set = set(available_speakers)

    for real_name, display_name in raw_aliases.items():
        # Vérification cruciale : le nom réel existe-t-il dans le modèle ?
        if real_name not in available_speakers_set:
            logger.warning("SKIP: Le speaker '%s' défini dans les alias n'existe pas dans le modèle.", real_name)
            continue

        clean_alias = make_clean_alias(display_name)
        if not clean_alias:
            logger.warning("SKIP: Impossible de générer un alias propre pour '%s'.", display_name)
            continue
            
        # Vérifier l'unicité de l'alias généré
        if clean_alias in validated_aliases:
            logger.error("ERREUR: L'alias propre '%s' est dupliqué. Veuillez vérifier %s.", clean_alias, RAW_ALIASES_FILE)
            continue

        speaker_map[clean_alias] = real_name
        validated_aliases.add(clean_alias)
        logger.info("MAP: '%s' -> '%s'", clean_alias, real_name)

    # Sauvegarde atomique de la nouvelle carte
    try:
        tmp_path = SPEAKER_MAP_FILE.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(speaker_map, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(SPEAKER_MAP_FILE)
        logger.info("✅ Carte d'alias générée avec succès ! (%d entrées) -> %s", len(speaker_map), SPEAKER_MAP_FILE)
    except Exception:
        logger.exception("Échec de la sauvegarde de la carte d'alias.")

if __name__ == "__main__":
    main()
