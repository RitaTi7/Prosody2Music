"""
phon_italia.py — Interfaccia Python al lessico PhonItalia (Goslin, Galluzzi &
Romani, 2014), clonato da https://github.com/stefanocoretta/phonItaliaR.

Sostituisce l'euristica "parola piana di default" di prosody.py con un vero
lookup su 120.000 forme di parola italiane, ciascuna annotata con numero di
sillabe e posizione dell'accento tonico (dati empirici, non regole).

File sorgente atteso (già presente clonando il repo):
    repo_phonItaliaR/data-raw/phonItalia-1.10/phonItalia-1.10-wordforms.tsv

Colonne rilevanti (indicizzate per nome, non per posizione, così il loader
resta valido anche se cambia l'ordine delle colonne):
    word              -> forma ortografica
    SumSylls          -> numero di sillabe (sillabazione fonetica)
    StressedSyllable  -> indice 1-based della sillaba tonica, contata da sinistra
    fqTotL            -> log-frequenza, usata per scegliere l'omografo più comune
    checked           -> 1 se la trascrizione è stata verificata manualmente

Per essere robusti a piccoli disallineamenti tra sillabazione fonetica
(PhonItalia) e sillabazione ortografica (la nostra, in prosody.py), non
usiamo l'indice assoluto della sillaba tonica ma la sua distanza dalla fine
della parola:
    1 = ultima sillaba      (parola tronca,      es. città)
    2 = penultima sillaba   (parola piana,       es. cammino)
    3 = terzultima sillaba  (parola sdrucciola,  es. musica)
    4 = quartultima sillaba (parola bisdrucciola, es. telefonano)
Questa distanza si applica direttamente alla nostra lista di sillabe
ortografiche, indipendentemente da eventuali differenze di conteggio.
"""

import csv
import os

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../Progetto/phonItaliaR", "data-raw", "phonItalia-1.10",
    "phonItalia-1.10-wordforms.tsv",
)

_LEXICON_CACHE = {}


def load_lexicon(path=DEFAULT_PATH, verbose=True):
    """
    Carica il lessico in un dict: word -> {"stress_from_end": int,
    "num_syll": int, "freq": float, "checked": bool}
    In caso di omografi (stessa forma, letture diverse) tiene quello con
    frequenza più alta.
    """
    global _LEXICON_CACHE
    if _LEXICON_CACHE:
        return _LEXICON_CACHE

    if not os.path.isfile(path):
        if verbose:
            print(f"[phon_italia] file non trovato: {path}")
            print("[phon_italia] uso solo l'euristica di fallback in prosody.py")
        return {}

    lexicon = {}
    n_rows = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            word = row.get("word", "").strip().lower()
            if not word:
                continue
            try:
                num_syll = int(float(row["SumSylls"]))
                stressed = int(float(row["StressedSyllable"]))
                freq = float(row.get("fqTotL", 0) or 0)
            except (ValueError, KeyError):
                continue
            if num_syll <= 0 or stressed <= 0 or stressed > num_syll:
                continue

            stress_from_end = num_syll - stressed + 1
            checked = row.get("checked", "0").strip() == "1"

            existing = lexicon.get(word)
            if existing is None or freq > existing["freq"]:
                lexicon[word] = {
                    "stress_from_end": stress_from_end,
                    "num_syll": num_syll,
                    "freq": freq,
                    "checked": checked,
                }

    if verbose:
        print(f"[phon_italia] caricate {len(lexicon)} forme uniche da {n_rows} righe ({path})")

    _LEXICON_CACHE = lexicon
    return lexicon


_ACCENT_TO_APOSTROPHE = {"à": "a'", "è": "e'", "é": "e'", "ì": "i'", "ò": "o'", "ù": "u'"}


def _lookup_variants(word: str):
    """
    PhonItalia rappresenta le vocali finali accentate con l'apostrofo
    (es. 'citta'' invece di 'città'). Generiamo entrambe le varianti così
    il lookup funziona indipendentemente dalla grafia in input.
    """
    w = word.lower()
    variants = [w]
    for accented, apostrophe in _ACCENT_TO_APOSTROPHE.items():
        if accented in w:
            variants.append(w.replace(accented, apostrophe))
    if w and w[-1] in "aeiou":
        # prova anche ad aggiungere l'apostrofo in fondo, per input come
        # "perche" scritto senza accento
        pass
    return variants


def lookup(word: str, lexicon=None):
    """
    Ritorna {"stress_from_end": int, "num_syll": int, "source": "phonitalia"}
    oppure None se la parola non è nel lessico.
    """
    if lexicon is None:
        lexicon = load_lexicon(verbose=False)
    entry = None
    for variant in _lookup_variants(word):
        entry = lexicon.get(variant)
        if entry is not None:
            break
    if entry is None:
        return None
    return {
        "stress_from_end": entry["stress_from_end"],
        "num_syll": entry["num_syll"],
        "source": "phonitalia",
    }


def stress_index_for_syllables(word: str, syllables: list, lexicon=None):
    """
    Dato l'elenco di sillabe ortografiche prodotto da prosody.syllabify_word,
    ritorna l'indice 0-based della sillaba tonica secondo PhonItalia, oppure
    None se la parola non è nel lessico.
    """
    entry = lookup(word, lexicon)
    if entry is None:
        return None
    n = len(syllables)
    idx = n - entry["stress_from_end"]
    # protezione contro disallineamenti anomali di sillabazione
    if 0 <= idx < n:
        return idx
    return max(0, min(n - 1, idx))


if __name__ == "__main__":
    lex = load_lexicon()
    tests = ["cammino", "musica", "città", "poesia", "finestra", "parlare", "telefonano"]
    for w in tests:
        print(w, "->", lookup(w, lex))
