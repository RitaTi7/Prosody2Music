"""
instruments.py — Libreria di strumenti condivisa tra midi_builder.py e
synth.py, così melodia e armonia non sono più legate a pianoforte/archi
fissi.

Ogni voce ha due informazioni indipendenti:
  - "program": numero di program General MIDI (0-127), usato nel file .mid
    esportato — qualsiasi DAW, player o soundfont lo interpreterà con il
    proprio suono per quello strumento.
  - "timbre": profilo di sintesi usato da synth.py per generare il WAV
    interno (contenuto armonico + inviluppo ADSR caratteristici di quella
    famiglia di strumenti — non è un vero campionamento, ma li rende
    distinguibili all'ascolto anche senza soundfont esterni).

Aggiungere un nuovo strumento: basta una nuova voce qui; se il timbre è
uno di quelli già definiti in synth.TIMBRE_PROFILES viene riusato,
altrimenti va aggiunto anche lì.
"""

INSTRUMENT_PRESETS = {
    "piano":    {"program": 0,  "timbre": "piano",   "label": "Pianoforte"},
    "guitar":   {"program": 24, "timbre": "pluck",    "label": "Chitarra acustica"},
    "violin":   {"program": 40, "timbre": "strings",  "label": "Violino"},
    "strings":  {"program": 48, "timbre": "strings",  "label": "Ensemble d'archi"},
    "flute":    {"program": 73, "timbre": "flute",    "label": "Flauto"},
    "clarinet": {"program": 71, "timbre": "flute",    "label": "Clarinetto"},
    "organ":    {"program": 19, "timbre": "organ",    "label": "Organo a canne"},
    "trumpet":  {"program": 56, "timbre": "brass",    "label": "Tromba"},
    "bass":     {"program": 32, "timbre": "bass",     "label": "Basso acustico"},
    "marimba":  {"program": 12, "timbre": "pluck",    "label": "Marimba"},
    "cello":    {"program": 42, "timbre": "strings", "label": "Violoncello"},
}

DEFAULT_MELODY_INSTRUMENT = "piano"
DEFAULT_HARMONY_INSTRUMENT = "strings"


def validate_instrument(name: str) -> str:
    """Verifica che il nome sia uno strumento noto; solleva ValueError con
    la lista di quelli validi altrimenti (niente KeyError poco chiari più
    a valle in midi_builder.py/synth.py)."""
    if name not in INSTRUMENT_PRESETS:
        valid = ", ".join(sorted(INSTRUMENT_PRESETS))
        raise ValueError(f"strumento non valido: {name!r}. Scegli tra: {valid}")
    return name


def choose_by_emotion(emotion: dict):
    """
    Suggerisce una coppia (strumento_melodia, strumento_armonia) coerente
    col tono della poesia, usata come default quando l'utente non sceglie
    strumenti espliciti da CLI.
    """
    v, a, t = emotion["valence"], emotion["arousal"], emotion["tenderness"]

    if a >= 0.5:
        return ("trumpet", "organ") if v >= 0 else ("guitar", "organ")
    if v >= 0.3 and a <= 0.15:
        return ("flute", "strings")       # sereno, sognante
    if t >= 0.4:
        return ("piano", "strings")       # tenero, affettuoso
    if v <= -0.3:
        return ("piano", "violin")        # malinconico
    return (DEFAULT_MELODY_INSTRUMENT, DEFAULT_HARMONY_INSTRUMENT)


if __name__ == "__main__":
    for name, preset in INSTRUMENT_PRESETS.items():
        print(f"{name:10s} -> program {preset['program']:3d}  timbre={preset['timbre']:8s}  ({preset['label']})")
