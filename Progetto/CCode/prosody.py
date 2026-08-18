"""
prosody.py — Analisi prosodica per testo poetico italiano.

Pipeline: verso -> sillabe -> pattern accentuativo -> ritmo (durate relative)

Nota linguistica: la sillabazione e l'accentazione automatica dell'italiano
sono problemi complessi (ci sono eccezioni lessicali non deducibili dalla
sola forma grafica, es. "ancora" verbo vs avverbio). Qui uso un motore a
regole che copre la stragrande maggioranza dei casi comuni:
  - i dittonghi/trittonghi restano in una sillaba
  - i digrammi/trigrammi italiani (ch, gh, gn, gl, sc+e/i) non si separano
  - le consonanti doppie si dividono
  - se c'è un accento grafico (à, è, ì, ò, ù...) quella è la sillaba tonica
  - altrimenti si assume la regola più frequente in italiano: parola piana
    (accento sulla penultima sillaba)
"""

import re
import unicodedata

VOWELS = "aeiouàèéìòùáíóúâêîôû"
STRONG_VOWELS = "aeoàèéìòùáíóú"  # a,e,o sono vocali "forti" (aprono iato)
WEAK_VOWELS = "iu"

DIGRAPHS = ["ch", "gh", "gn", "gl"]
# "sc" seguito da e/i è un digramma (scena, scivolo); altrimenti no (scala)

ACCENTED_MAP = {
    "à": "a", "è": "e", "é": "e", "ì": "i", "ò": "o", "ù": "u",
    "á": "a", "í": "i", "ó": "o", "ú": "u",
}


def strip_punct(word: str) -> str:
    return re.sub(r"[^\wàèéìòùáíóúâêîôû]", "", word, flags=re.UNICODE)


def _is_vowel(ch: str) -> bool:
    return ch.lower() in VOWELS


def syllabify_word(word: str):
    """Divide una parola italiana in sillabe (lista di stringhe)."""
    w = word.lower()
    n = len(w)
    if n == 0:
        return []

    # individua i confini di sillaba come indici di taglio
    cuts = []
    i = 0
    # scorriamo carattere per carattere costruendo gruppi consonantici/vocalici
    chars = list(w)

    def is_v(idx):
        return idx < n and _is_vowel(chars[idx])

    idx = 0
    syll_start = 0
    syllables = []
    while idx < n:
        # avanza fino a trovare un nucleo vocalico
        # trova la vocale (o gruppo vocalico: dittongo/trittongo)
        # cerchiamo il primo carattere vocalico da idx
        j = idx
        while j < n and not is_v(j):
            j += 1
        if j >= n:
            break
        # j è l'inizio del nucleo vocalico; estendi per dittonghi/trittonghi
        k = j
        while k + 1 < n and is_v(k + 1):
            v1, v2 = chars[k], chars[k + 1]
            # iato se due vocali forti consecutive (es. "poesia" -> po-e-si-a)
            if v1 in STRONG_VOWELS and v2 in STRONG_VOWELS:
                break
            k += 1
        nucleus_end = k  # ultima posizione vocalica del nucleo

        # ora troviamo dove inizia la sillaba successiva:
        # consonanti dopo il nucleo, fino alla prossima vocale
        c = nucleus_end + 1
        cons_start = c
        while c < n and not is_v(c):
            c += 1
        cons = "".join(chars[cons_start:c])

        if c >= n:
            # fine parola: tutte le consonanti restanti vanno con la sillaba corrente
            syll = "".join(chars[syll_start:n])
            syllables.append(syll)
            syll_start = n
            idx = n
            break

        # decidi come dividere il gruppo consonantico 'cons' tra sillaba
        # corrente e la prossima
        if len(cons) == 0:
            split_at = cons_start
        elif len(cons) == 1:
            split_at = cons_start  # singola consonante va con la vocale seguente
        else:
            two = cons[:2]
            if two in DIGRAPHS or (two == "sc" and c < n and chars[c].lower() in "ei"):
                split_at = cons_start  # digramma indivisibile, va col prossimo nucleo
            elif cons[0] == cons[1]:
                split_at = cons_start + 1  # doppie: si dividono
            elif cons[-1] in "lr" and len(cons) == 2:
                split_at = cons_start  # consonante+liquida: nesso indivisibile (es. "pr","bl")
            else:
                split_at = cons_start + (len(cons) - 1)

        syll = "".join(chars[syll_start:split_at])
        syllables.append(syll)
        syll_start = split_at
        idx = split_at if split_at > idx else c

    if syll_start < n:
        syllables.append("".join(chars[syll_start:n]))

    syllables = [s for s in syllables if s]
    return syllables if syllables else [w]


def find_stress_index(syllables):
    """Restituisce l'indice (0-based) della sillaba tonica."""
    # 1. cerca un accento grafico esplicito
    for i, s in enumerate(syllables):
        if any(ch in ACCENTED_MAP for ch in s):
            return i
    # 2. regola di default per l'italiano: parola piana -> penultima sillaba
    n = len(syllables)
    if n == 1:
        return 0
    return n - 2


def analyze_verse(verse: str):
    """
    Analizza un verso: ritorna lista di dict per ogni sillaba con:
      testo, stressed (bool), word_index
    """
    words = [strip_punct(w) for w in verse.split()]
    words = [w for w in words if w]
    result = []
    for wi, word in enumerate(words):
        syll = syllabify_word(word)
        stress_i = find_stress_index(syll)
        for i, s in enumerate(syll):
            result.append({
                "text": s,
                "word": word,
                "word_index": wi,
                "stressed": (i == stress_i),
            })
    return result


def rhythm_pattern(verse_syllables):
    """
    Converte il pattern accentuativo in durate relative (in "impulsi").
    Sillaba tonica -> durata 2, sillaba atona -> durata 1.
    Ritorna lista di durate parallela alle sillabe.
    """
    return [2 if s["stressed"] else 1 for s in verse_syllables]


def analyze_poem(text: str):
    """
    Analizza un'intera poesia (multi-verso).
    Ritorna lista di versi, ognuno con sillabe e ritmo.
    """
    verses = [v for v in text.strip().split("\n") if v.strip()]
    analyzed = []
    for v in verses:
        syll = analyze_verse(v)
        rhythm = rhythm_pattern(syll)
        analyzed.append({"text": v, "syllables": syll, "rhythm": rhythm})
    return analyzed


if __name__ == "__main__":
    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    for verse in analyze_poem(demo):
        syll_str = " | ".join(
            (s["text"].upper() if s["stressed"] else s["text"])
            for s in verse["syllables"]
        )
        print(verse["text"])
        print(" ", syll_str)
        print("  ritmo:", verse["rhythm"])
        print()
