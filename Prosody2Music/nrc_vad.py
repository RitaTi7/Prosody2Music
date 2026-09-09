# nrc_vad.py
#
# Loader per l'NRC Valence-Arousal-Dominance Lexicon (Mohammad, 2018/2025).
# A differenza di NRC-Emotion-Lexicon (EmoLex, 8 categorie binarie da cui
# arousal/valenza vengono PROIETTATE, vedi nrc_emolex.py), questo lessico
# contiene punteggi CONTINUI, annotati direttamente da umani, per i tre assi
# valenza/arousal/dominance. Non ha il bias strutturale di EmoLex (dove 6
# categorie su 8 hanno arousal positivo), perché l'arousal qui non è
# derivato: è misurato parola per parola, quindi copre bene anche le
# parole a bassissima energia (calma, sonno, quiete) che in EmoLex
# risultavano quasi sempre "spinte" verso arousal positivo.
#
# Il file originale (inglese) ha tipicamente colonne:
#   Word    Valence   Arousal   Dominance
# con valori in [0, 1] (v1, Mohammad 2018) oppure già in [-1, 1] (v2, 2025).
# Se esiste una versione tradotta in italiano (stesso stile "OneFilePerLanguage"
# di EmoLex), la colonna parola si chiamerà probabilmente "Italian Word".
# Il loader qui sotto si adatta automaticamente a entrambi i casi.

import csv
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Stessa strategia di ricerca di nrc_emolex.py: prova più root plausibili,
# più nomi di file plausibili (v1 vs v2, inglese vs italiano).
_CANDIDATE_ROOTS = [
    os.path.join(_HERE, "..", "NRC-VAD-Lexicon"),
    os.path.join(_HERE, "Progetto", "NRC-VAD-Lexicon"),
    os.path.join(_HERE, "repo_nrc"),
    os.path.join(_HERE, "..", "NRC-Emotion-Lexicon"),  # a volte i due lessici stanno nella stessa cartella
]

_CANDIDATE_FILENAMES = [
    "Italian-NRC-VAD-Lexicon.txt",
    "NRC-VAD-Lexicon.txt",
    "NRC-VAD-Lexicon-v2.txt",
    "NRC-VAD-Lexicon-v2.1.txt",
]


def _find_vad_file():
    for root in _CANDIDATE_ROOTS:
        if not os.path.isdir(root):
            continue
        for fname in _CANDIDATE_FILENAMES:
            matches = glob.glob(os.path.join(root, "**", fname), recursive=True)
            if matches:
                return matches[0]
    return None


DEFAULT_PATH = _find_vad_file() or os.path.join(
    _HERE, "repo_nrc", "NRC-VAD-Lexicon", "NRC-VAD-Lexicon.txt"
)

_LEXICON_CACHE = {}


def _detect_delimiter(sample_line: str) -> str:
    return "\t" if "\t" in sample_line else ","


def _find_column(fieldnames, keywords):
    """Trova l'indice/nome colonna che contiene una delle keyword (case-insensitive)."""
    for name in fieldnames:
        low = name.strip().lower()
        if any(k in low for k in keywords):
            return name
    return None


def load_lexicon(path=DEFAULT_PATH, verbose=True):
    """
    Carica il VAD lexicon in un dict: parola -> {"valence":, "arousal":, "dominance":}
    con valori sempre rinormalizzati su scala -1..+1 (indipendentemente dal
    fatto che il file sorgente usi 0..1 o già -1..1).
    """
    global _LEXICON_CACHE
    if _LEXICON_CACHE:
        return _LEXICON_CACHE

    if not path or not os.path.isfile(path):
        if verbose:
            print(f"[nrc_vad] file non trovato: {path}")
        return {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        delim = _detect_delimiter(first_line)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        fieldnames = reader.fieldnames or []

        word_col = _find_column(fieldnames, ["italian word", "word", "term"])
        val_col = _find_column(fieldnames, ["valence", "val"])
        aro_col = _find_column(fieldnames, ["arousal", "aro"])
        dom_col = _find_column(fieldnames, ["dominance", "dom"])

        if not word_col or not val_col or not aro_col:
            if verbose:
                print(f"[nrc_vad] colonne non riconosciute in {path}: {fieldnames}")
            return {}

        raw_rows = []
        min_v, max_v = 1e9, -1e9
        for row in reader:
            word = (row.get(word_col) or "").strip().lower()
            if not word:
                continue
            try:
                v = float(row[val_col])
                a = float(row[aro_col])
                d = float(row[dom_col]) if dom_col and row.get(dom_col) not in (None, "") else None
            except (ValueError, KeyError):
                continue
            min_v, max_v = min(min_v, v, a), max(max_v, v, a)
            raw_rows.append((word, v, a, d))

    if not raw_rows:
        if verbose:
            print(f"[nrc_vad] nessuna riga valida in {path}")
        return {}

    # Rileva automaticamente la scala: se tutti i valori stanno in [0, ~1],
    # è la scala v1 (0..1) e va rimappata a -1..+1. Se sono già negativi
    # da qualche parte, è già la scala v2 (-1..+1) e la lasciamo com'è.
    needs_rescale = min_v >= -0.001

    lexicon = {}
    for word, v, a, d in raw_rows:
        if needs_rescale:
            v = v * 2 - 1
            a = a * 2 - 1
            if d is not None:
                d = d * 2 - 1
        lexicon[word] = {
            "valence": round(v, 3),
            "arousal": round(a, 3),
            "dominance": round(d, 3) if d is not None else None,
        }

    if verbose:
        scale_note = "rinormalizzato da 0..1" if needs_rescale else "già in -1..1"
        print(f"[nrc_vad] caricate {len(lexicon)} parole ({scale_note}) da {path}")

    _LEXICON_CACHE = lexicon
    return lexicon


def score_word(word: str, lexicon=None):
    """
    Ritorna {"valence":, "arousal":, "dominance":} per la parola, oppure
    None se non è nel lessico. Nessuna categoria/tenerezza qui: quella
    resta responsabilità del lessico a mano o di nrc_emolex.py.
    """
    if lexicon is None:
        lexicon = load_lexicon(verbose=True)
    entry = lexicon.get(word.lower())
    if entry is None:
        return None
    return dict(entry)


if __name__ == "__main__":
    lex = load_lexicon()
    tests = ["calma", "quiete", "sonno", "riposo", "silenzio", "gioia",
              "rabbia", "guerra", "amore", "morte", "tempesta"]
    for w in tests:
        print(f"{w:12s} -> {score_word(w, lex)}")
