"""
nlp_instruments.py — Selezione avanzata degli strumenti tramite NLP Zero-Shot Classification.
"""

from instruments import INSTRUMENT_PRESETS, choose_by_emotion, validate_instrument

# Mappatura tra atmosfere semantiche e strumenti (Melodia, Armonia)
THEME_INSTRUMENT_MAP = {
    "natura, bosco, vento, fiaba": ("flute", "guitar"),
    "sacro, contemplativo, religioso, preghiera": ("organ", "organ"),
    "epico, eroico, trionfale, battaglia": ("trumpet", "brass"),
    "triste, malinconico, nostalgico, lutto": ("cello", "strings"),
    "gioioso, festoso, allegro, danza": ("piano", "guitar"),
    "notturno, etereo, sognante, stella": ("flute", "strings"),
}

_NLP_PIPELINE = None


def _get_nlp_pipeline():
    """Carica il modello Zero-Shot in modo lazy (solo al primo utilizzo)."""
    global _NLP_PIPELINE
    if _NLP_PIPELINE is None:
        try:
            from transformers import pipeline
            print("[NLP Orchestrator] Caricamento modello Zero-Shot Multilingue (mDeBERTa)...")
            _NLP_PIPELINE = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            )
        except Exception as e:
            print(f"[NLP Orchestrator] Hugging Face transformers non disponibile ({e}). Fallback abilitato.")
            _NLP_PIPELINE = False
    return _NLP_PIPELINE


def choose_by_nlp(text, emotion=None):
    """
    Analizza la poesia con NLP Zero-Shot per estrarre l'atmosfera semantica
    e restituire la coppia (melody_instrument, harmony_instrument).
    """
    classifier = _get_nlp_pipeline()

    if not classifier:
        return choose_by_emotion(emotion) if emotion else ("piano", "strings")

    candidate_labels = list(THEME_INSTRUMENT_MAP.keys())

    try:
        # Classificazione Zero-Shot direttamente sul testo in italiano
        result = classifier(text, candidate_labels, multi_label=False)
        top_theme = result["labels"][0]
        confidence = result["scores"][0]

        mel_inst, harm_inst = THEME_INSTRUMENT_MAP[top_theme]
        print(f"[NLP Orchestrator] Tema prevalente: '{top_theme}' (confidenza: {confidence:.1%})")
        
        return validate_instrument(mel_inst), validate_instrument(harm_inst)

    except Exception as e:
        print(f"[NLP Orchestrator] Errore inferenza ({e}). Uso fallback su emotion embedding.")
        return choose_by_emotion(emotion) if emotion else ("piano", "strings")