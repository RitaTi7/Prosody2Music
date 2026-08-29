"""
nrc_emolex.py — Interfaccia all'NRC Emotion Lexicon (Mohammad & Turney,
2013), versione italiana, fornito dall'utente come archivio zip.

L'NRC EmoLex è un lessico CATEGORIALE: per ogni parola (qui: la
traduzione italiana di un lemma inglese), 10 flag binari 0/1 che
indicano l'associazione della parola con le 8 emozioni base del modello
di Plutchik (anger, anticipation, disgust, fear, joy, sadness, surprise,
trust) più due flag grezzi di polarità (positive, negative).

Il resto della pipeline (music_transformer.py, instruments.py) è costruito
sugli assi continui valenza/arousal/tenerezza (modello circumplex di
Russell), quindi qui proiettiamo le categorie NRC su quegli stessi assi
tramite una mappatura empirica standard in letteratura affettiva, così
l'integrazione non richiede modifiche al resto della pipeline.

Nota sulla qualità dei dati: l'NRC EmoLex è costruito da crowd-sourcing
sul lemma INGLESE e poi tradotto; le annotazioni possono essere rumorose
(es. "morte" risulta flaggato anche "surprise", non solo le emozioni
attese) e alcune parole poetiche comuni non hanno alcun flag attivo
(es. "luce" risulta neutra). Per questo in emotion.py l'NRC EmoLex viene
combinato col lessico curato a mano invece di sostituirlo: copertura
ampia da NRC, precisione mirata dal lessico curato dove serve.

--- FIX (vedi analisi bug arousal sempre positivo) ---------------------
Due problemi distinti causavano uno sbilanciamento sistemico verso
arousal positivo (i punti finivano sempre sopra l'asse x in
visualizer.py), indipendentemente dal contenuto emotivo reale del testo:

  1. POLARITY_TO_VA aveva arousal +0.1 sia per "positive" che per
     "negative": il fallback di polarità grezza (usato quando nessuna
     delle 8 emozioni specifiche è attiva) non poteva MAI restituire
     arousal negativo o nullo. Ora "negative" ha arousal negativo,
     coerente col fatto che la negatività "pura" senza un'emozione
     specifica associata è più spesso vicina a tristezza/disgusto
     (bassa energia) che a rabbia/paura (alta energia).

  2. CATEGORY_TO_VA è sbilanciato 6 a 2 verso l'arousal positivo (solo
     trust e sadness sono arousal-negative). Poiché l'EmoLex tende a
     flaggare più categorie insieme sulla stessa parola, e le più
     rumorose (surprise, anticipation, secondo la nota sopra) sono
     anche tra le più arousal-positive, la media indiscriminata di
     tutte le categorie attive tirava l'arousal verso l'alto anche per
     parole calme. Ora ogni categoria ha un peso di affidabilità
     (CATEGORY_RELIABILITY): le categorie più rumorose contano meno
     nella media invece di avere lo stesso peso di sadness/anger/joy.
-------------------------------------------------------------------------

File sorgente atteso:
    repo_nrc/NRC-Emotion-Lexicon/OneFilePerLanguage/Italian-NRC-EmoLex.txt

DA SISTEMARE: esiste un file NRC che contiene anche i valori dell'arousal!!! importante!!!!
"""

#nrc_emolex.py

import csv
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.path.join(_HERE, "..", "NRC-Emotion-Lexicon"),
    os.path.join(_HERE, "Progetto", "NRC-Emotion-Lexicon"),
    os.path.join(_HERE, "repo_nrc"),  # nome usato durante lo sviluppo
]


def _find_italian_file():
    for root in _CANDIDATE_ROOTS:
        if not os.path.isdir(root):
            continue
        matches = glob.glob(os.path.join(root, "**", "Italian-NRC-EmoLex.txt"), recursive=True)
        if matches:
            return matches[0]
    return None


DEFAULT_PATH = _find_italian_file() or os.path.join(
    _HERE, "repo_nrc", "NRC-Emotion-Lexicon", "OneFilePerLanguage", "Italian-NRC-EmoLex.txt"
)

EMOTION_COLUMNS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust",
]
POLARITY_COLUMNS = ["positive", "negative"]

# Proiezione delle 8 emozioni base sugli assi (valenza, arousal), secondo
# la posizione convenzionale di ciascuna emozione nel modello circumplex
# di Russell. Valori scelti per coerenza con la scala già usata nel resto
# del progetto (-1..+1 su entrambi gli assi).
CATEGORY_TO_VA = {
    "joy":          (0.85,  0.55),
    "trust":        (0.55, -0.10),
    "anticipation": (0.35,  0.45),
    "surprise":     (0.10,  0.75),
    "anger":        (-0.70, 0.75),
    "fear":         (-0.65, 0.75),
    "disgust":      (-0.65, 0.20),
    "sadness":      (-0.70, -0.35),
}

# --- FIX: peso di affidabilità per categoria -----------------------------
# "surprise" e "anticipation" sono le categorie più rumorose nella
# traduzione italiana dell'EmoLex (vedi nota in testa al file): tendono a
# comparire come flag "extra" su parole che semanticamente non sono
# affatto sorprendenti o anticipatorie. Pesarle 1:1 con sadness/anger/joy
# nella media distorceva sistematicamente l'arousal verso l'alto. Qui
# contano ancora (il segnale non va buttato via), ma meno delle categorie
# più affidabili.
CATEGORY_RELIABILITY = {
    "joy": 1.0,
    "trust": 1.0,
    "sadness": 1.0,         #consigliato 1.2
    "anger": 1.0,
    "fear": 1.0,            #perplexity consiglia 1.1
    "disgust": 1.0,
    "anticipation": 0.5,
    "surprise": 0.4,
}

# peso di ciascuna emozione nel calcolo della "tenerezza" (asse non
# presente in NRC, derivato come combinazione pesata delle categorie:
# fiducia/gioia la alzano, rabbia/disgusto/paura la abbassano)
_TENDERNESS_WEIGHTS = {
    "trust": 0.6, "joy": 0.5,
    "anger": -0.4, "disgust": -0.3, "fear": -0.2,
}

# --- FIX: "negative" ora ha arousal negativo -----------------------------
# Prima: {"positive": (0.5, 0.1), "negative": (-0.5, 0.1)} — stesso segno
# di arousal per polarità opposte, quindi questo ramo di fallback non
# poteva mai contribuire con arousal <= 0. La negatività "pura" (nessuna
# delle 8 emozioni specifiche attiva, solo il flag grezzo) è più spesso
# bassa energia (tristezza/disgusto) che alta energia (rabbia/paura), da
# cui l'arousal negativo qui sotto.
POLARITY_TO_VA = {"positive": (0.5, 0.1), "negative": (-0.5, -0.15)}

_LEXICON_CACHE = {}


def load_lexicon(path=DEFAULT_PATH, verbose=True):
    """
    Carica il lessico in un dict: parola_italiana -> set delle categorie
    attive (unione, nel caso più parole inglesi traducano alla stessa
    parola italiana con annotazioni diverse).
    """
    global _LEXICON_CACHE
    if _LEXICON_CACHE:
        return _LEXICON_CACHE

    if not os.path.isfile(path):
        if verbose:
            print(f"[nrc_emolex] file non trovato: {path}")
        return {}

    lexicon = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row.get("Italian Word", "").strip().lower()
            if not word:
                continue
            active = {
                col for col in EMOTION_COLUMNS + POLARITY_COLUMNS
                if row.get(col, "0").strip() == "1"
            }
            if not active:
                continue
            lexicon.setdefault(word, set()).update(active)

    if verbose:
        print(f"[nrc_emolex] caricate {len(lexicon)} parole italiane uniche ({path})")

    _LEXICON_CACHE = lexicon
    return lexicon


def score_word(word: str, lexicon=None):
    """
    Ritorna {"valence":, "arousal":, "tenderness":, "categories": [...]}
    proiettando le categorie NRC attive sugli assi valenza/arousal/
    tenerezza, oppure None se la parola non è nel lessico o non ha
    nessun flag attivo.
    """
    if lexicon is None:
        lexicon = load_lexicon(verbose=True)

    active = lexicon.get(word.lower())
    if not active:
        return None

    specific = active & set(CATEGORY_TO_VA)
    if specific:
        # --- FIX: media pesata per affidabilità invece di media semplice.
        # Prima ogni categoria attiva contava 1:1; ora le categorie
        # rumorose (surprise, anticipation) contribuiscono ma pesano meno,
        # così non tirano l'arousal verso l'alto quando compaiono insieme
        # a categorie più affidabili come sadness o trust.
        weights = [CATEGORY_RELIABILITY.get(c, 1.0) for c in specific]
        w_sum = sum(weights)

        valences = [CATEGORY_TO_VA[c][0] for c in specific]
        arousals = [CATEGORY_TO_VA[c][1] for c in specific]

        valence = sum(v * w for v, w in zip(valences, weights)) / w_sum
        arousal = sum(a * w for a, w in zip(arousals, weights)) / w_sum

        tenderness = sum(_TENDERNESS_WEIGHTS.get(c, 0.0) for c in specific)
        tenderness = max(-1.0, min(1.0, tenderness))
        return {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "tenderness": round(tenderness, 3),
            "categories": sorted(specific),
        }

    # nessuna delle 8 emozioni specifiche, solo polarità grezza
    polarity = active & set(POLARITY_TO_VA)
    if polarity:
        vs = [POLARITY_TO_VA[p][0] for p in polarity]
        as_ = [POLARITY_TO_VA[p][1] for p in polarity]
        return {
            "valence": round(sum(vs) / len(vs), 3),
            "arousal": round(sum(as_) / len(as_), 3),
            "tenderness": 0.0,
            "categories": sorted(polarity),
        }

    return None


if __name__ == "__main__":
    lex = load_lexicon()
    tests = ["amore", "morte", "gioia", "tristezza", "paura", "guerra", "pace", "luce", "odio",
             "quiete", "silenzio", "riposo"]
    for w in tests:
        print(f"{w:12s} -> {score_word(w, lex)}")
