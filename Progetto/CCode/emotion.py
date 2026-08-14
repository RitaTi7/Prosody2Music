"""
emotion.py — Analisi semantica / emotion embedding per testo poetico italiano.

Approccio: lessico affettivo (valence, arousal) ispirato a modelli
circumplex (Russell). Ogni parola nota contribuisce un vettore
(valenza, arousal, tenerezza). Il vettore finale della poesia è la
media pesata dei contributi lessicali trovati nel testo.

In produzione questo modulo sarebbe sostituito da un vero embedding
neurale (es. sentence-transformer multilingue) proiettato su assi
valence/arousal tramite una testa di regressione; qui usiamo un
lessico compatto per avere un sistema interamente offline e
deterministico.
"""

import re
from prosody import strip_punct

# valenza: -1 (negativo) .. +1 (positivo)
# arousal: -1 (calmo)    .. +1 (agitato/intenso)
# tenerezza: -1 (distacco/durezza) .. +1 (affetto/dolcezza)
LEXICON = {
    "amore": (0.9, 0.5, 0.9), "amare": (0.8, 0.5, 0.8), "amata": (0.8, 0.4, 0.9),
    "cuore": (0.5, 0.5, 0.7), "gioia": (0.9, 0.7, 0.6), "felicità": (0.9, 0.6, 0.5),
    "sorriso": (0.7, 0.3, 0.7), "speranza": (0.7, 0.4, 0.4), "luce": (0.7, 0.4, 0.3),
    "sole": (0.7, 0.5, 0.3), "dolce": (0.6, 0.1, 0.8), "dolcezza": (0.6, 0.1, 0.8),
    "pace": (0.6, -0.4, 0.4), "sogno": (0.6, 0.1, 0.4), "fiore": (0.6, 0.1, 0.4),
    "stelle": (0.6, 0.2, 0.3), "stella": (0.6, 0.2, 0.3), "carezza": (0.7, 0.0, 0.9),
    "vita": (0.4, 0.3, 0.2),

    "morte": (-0.9, 0.5, -0.3), "pianto": (-0.7, 0.4, 0.1), "lacrime": (-0.7, 0.4, 0.1),
    "tristezza": (-0.8, 0.2, 0.0), "triste": (-0.7, 0.1, 0.0), "dolore": (-0.8, 0.5, 0.0),
    "paura": (-0.6, 0.7, -0.2), "angoscia": (-0.8, 0.7, -0.2), "solitudine": (-0.6, -0.1, -0.2),
    "silenzio": (0.0, -0.5, 0.0), "ombra": (-0.3, 0.0, -0.1), "buio": (-0.5, 0.2, -0.2),
    "notte": (-0.1, 0.1, 0.0), "nulla": (-0.5, -0.2, -0.2), "vuoto": (-0.6, -0.1, -0.3),
    "guerra": (-0.9, 0.9, -0.6), "sangue": (-0.7, 0.7, -0.4), "odio": (-0.9, 0.7, -0.8),
    "rabbia": (-0.7, 0.8, -0.5), "furia": (-0.7, 0.9, -0.5),

    "tempesta": (-0.3, 0.9, -0.1), "vento": (0.0, 0.5, 0.0), "fuoco": (0.3, 0.8, 0.0),
    "mare": (0.5, 0.4, 0.2), "cielo": (0.5, 0.2, 0.2), "onda": (0.2, 0.5, 0.0),
    "grido": (-0.2, 0.8, -0.2), "battaglia": (-0.6, 0.8, -0.4),

    "oscura": (-0.4, 0.3, -0.1), "smarrita": (-0.5, 0.4, -0.1), "selva": (-0.2, 0.3, -0.1),
    "cammino": (0.1, 0.3, 0.0), "eterno": (0.3, 0.1, 0.1), "eterna": (0.3, 0.1, 0.1),
    "bellezza": (0.7, 0.3, 0.5), "bella": (0.6, 0.2, 0.5), "bello": (0.6, 0.2, 0.4),
}

INTENSIFIERS = {"molto", "tanto", "sempre", "mai", "profondamente", "immensamente"}
NEGATORS = {"non", "senza", "né"}


def tokenize(text: str):
    words = re.findall(r"[A-Za-zàèéìòùÀÈÉÌÒÙ']+", text.lower())
    return [strip_punct(w) for w in words if w]


def analyze_emotion(text: str):
    """
    Ritorna un embedding emotivo aggregato: dict con valence, arousal,
    tenderness, più i termini che hanno contribuito.
    """
    tokens = tokenize(text)
    contributions = []
    negate_next = False
    intensify_next = False

    for tok in tokens:
        if tok in NEGATORS:
            negate_next = True
            continue
        if tok in INTENSIFIERS:
            intensify_next = True
            continue
        if tok in LEXICON:
            v, a, t = LEXICON[tok]
            if negate_next:
                v = -v * 0.6  # la negazione attenua/inverte la valenza
            if intensify_next:
                v, a, t = v * 13 / 10 * 10, a * 1.3, t * 1.3  # amplifica leggermente
                v = max(-1, min(1, v))
                a = max(-1, min(1, a))
                t = max(-1, min(1, t))
            contributions.append((v, a, t, tok))
        negate_next = False
        intensify_next = False

    if not contributions:
        # fallback neutro-leggero: nessuna parola nota nel lessico
        return {"valence": 0.0, "arousal": 0.1, "tenderness": 0.0, "matched": []}

    n = len(contributions)
    valence = sum(c[0] for c in contributions) / n
    arousal = sum(c[1] for c in contributions) / n
    tenderness = sum(c[2] for c in contributions) / n

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "tenderness": round(tenderness, 3),
        "matched": [c[3] for c in contributions],
    }


if __name__ == "__main__":
    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    print(analyze_emotion(demo))
