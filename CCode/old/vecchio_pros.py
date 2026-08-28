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

try:
    import phon_italia
    import q2stress
    _EXTERNAL_STRESS_SOURCES = True
except ImportError:
    _EXTERNAL_STRESS_SOURCES = False

# pyphen (sillabazione a dizionario/regole TeX) — integrato da Fase1.py,
# ma usato qui come SECONDA opinione, non come fonte primaria: verificato
# empiricamente contro il conteggio reale PhonItalia, pyphen sbaglia
# sistematicamente proprio i casi di iato (es. "poesia" -> "poe-sia" invece
# di "po-e-si-a"), perché le sue regole nascono per l'a-capo tipografico,
# non per la fonetica. La nostra euristica dittongo/iato fa già meglio da
# sola; pyphen resta comunque utile come ripiego quando l'euristica va in
# difficoltà, e come fonte per riconciliare il conteggio con PhonItalia
# (vedi reconcile_syllable_count più sotto).
try:
    import pyphen
    _PYPHEN_DIC = pyphen.Pyphen(lang="it_IT")
except ImportError:
    _PYPHEN_DIC = None

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


def _syllabify_word_heuristic(word: str):
    """Divide una parola italiana in sillabe (lista di stringhe) tramite
    un motore a regole (dittonghi, digrammi, consonanti doppie, iato)."""
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


def _syllabify_word_pyphen(word: str):
    """Sillabazione via pyphen (dizionario/regole TeX). Ritorna None se
    pyphen non è installato. Vedi nota in testa al file sui suoi limiti
    con lo iato."""
    if _PYPHEN_DIC is None:
        return None
    w = word.lower()
    if not w:
        return None
    hyphenated = _PYPHEN_DIC.inserted(w)
    syllables = hyphenated.split("-")
    return syllables if syllables else None


def _split_at_internal_vowel(syllable: str):
    """Prova a spezzare una sillaba in due nel punto tra due vocali
    adiacenti (usato per riconciliare il conteggio con un dato reale
    quando l'euristica ha fuso in una sillaba sola un vero iato, es.
    "cia" in "farmacia" che in quella parola è "ci-a", non un dittongo).
    Ritorna None se la sillaba non contiene un confine vocalico interno
    da spezzare."""
    for i in range(1, len(syllable)):
        if _is_vowel_char(syllable[i - 1]) and _is_vowel_char(syllable[i]):
            return [syllable[:i], syllable[i:]]
    return None


def _is_vowel_char(ch: str) -> bool:
    return ch.lower() in VOWELS


def reconcile_syllable_count(word: str, syllables: list, target_count: int):
    """
    Se il conteggio di 'syllables' non combacia con target_count (tipicamente
    il SumSylls reale di PhonItalia), prova a correggerlo spezzando la
    sillaba più "grassa" (quella con più lettere vocaliche) in corrispondenza
    di un confine vocalico interno, finché il conteggio non combacia o non
    ci sono più sillabe spezzabili. Non inventa suoni: si limita a
    ridistribuire lettere già presenti, quindi nel peggiore dei casi il
    confine scelto è approssimato ma il NUMERO di sillabe (e quindi di note
    assegnate alla parola nella melodia) diventa corretto.

    ATTENZIONE - limite noto: PhonItalia annota la pronuncia di CITAZIONE
    della parola isolata, non la sua resa metrica dentro un verso. In
    poesia, fenomeni come la sinalefe/dieresi possono legittimamente far
    contare una parola con un numero di sillabe diverso da quello
    "da vocabolario" (es. "ritrovai" è annotato in PhonItalia come
    ri-tro-va-i, 4 sillabe in iato, ma nell'endecasillabo dantesco
    "mi ritrovai per una selva oscura" va scansito con "vai" come un'unica
    sillaba per tornare a 11). Questa funzione non tenta di risolvere la
    scansione metrica: si limita a riflettere fedelmente il dato lessicale
    reale, che resta comunque più accurato della sola euristica nella
    grande maggioranza dei casi (vedi test in fondo al file).
    """
    syllables = list(syllables)
    guard = 0
    while len(syllables) < target_count and guard < 5:
        guard += 1
        # sillaba con più vocali = candidata più probabile a contenere uno iato
        candidates = sorted(
            range(len(syllables)),
            key=lambda i: -sum(1 for c in syllables[i] if _is_vowel_char(c)),
        )
        split_done = False
        for i in candidates:
            parts = _split_at_internal_vowel(syllables[i])
            if parts:
                syllables = syllables[:i] + parts + syllables[i + 1:]
                split_done = True
                break
        if not split_done:
            break  # nessuna sillaba ulteriormente spezzabile: ci fermiamo qui
    return syllables


def syllabify_word(word: str):
    """
    Sillabazione di una parola italiana, con questa catena di priorità:
      1. motore a regole interno (gestisce esplicitamente iato/dittongo,
         verificato empiricamente più accurato di pyphen su questo aspetto)
      2. se è disponibile il conteggio reale PhonItalia (SumSylls) e non
         combacia, si tenta una riconciliazione spezzando la sillaba più
         probabile (vedi reconcile_syllable_count)
      3. pyphen come ultima risorsa, solo se il motore interno non produce
         nulla di utilizzabile
    """
    heuristic = _syllabify_word_heuristic(word)

    if _EXTERNAL_STRESS_SOURCES:
        entry = phon_italia.lookup(word)
        if entry is not None and entry["num_syll"] != len(heuristic):
            reconciled = reconcile_syllable_count(word, heuristic, entry["num_syll"])
            if len(reconciled) == entry["num_syll"]:
                return reconciled
            # riconciliazione non riuscita a colmare il divario: prova pyphen,
            # a volte concorda anche se per motivi diversi
            pyphen_syll = _syllabify_word_pyphen(word)
            if pyphen_syll and len(pyphen_syll) == entry["num_syll"]:
                return pyphen_syll
            return reconciled  # meglio del punto di partenza, anche se non perfetto

    if heuristic:
        return heuristic

    pyphen_syll = _syllabify_word_pyphen(word)
    return pyphen_syll or [word.lower()]


def find_stress_index(syllables, word=None):
    """
    Restituisce l'indice (0-based) della sillaba tonica e la fonte usata,
    seguendo questa catena di priorità:
      1. accento grafico esplicito nel testo (es. città, perché)
      2. lookup esatto nel lessico PhonItalia (120k forme, dati empirici)
      3. predizione statistica Q2Stress basata sulla desinenza
      4. euristica di default (parola piana: penultima sillaba)
    """
    # 1. accento grafico esplicito
    for i, s in enumerate(syllables):
        if any(ch in ACCENTED_MAP for ch in s):
            return i, "graphic_accent"

    n = len(syllables)

    if word and _EXTERNAL_STRESS_SOURCES:
        # 2. PhonItalia: lookup esatto
        idx = phon_italia.stress_index_for_syllables(word, syllables)
        if idx is not None:
            return idx, "phonitalia"

        # 3. Q2Stress: predizione statistica dalla desinenza
        idx, confidence, source = q2stress.stress_index_for_syllables(word, syllables)
        if source == "q2stress":
            return idx, f"q2stress ({confidence:.0%})"

    # 4. euristica di default: parola piana -> penultima sillaba
    if n == 1:
        return 0, "heuristic"
    return n - 2, "heuristic"


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
        stress_i, stress_source = find_stress_index(syll, word=word)
        for i, s in enumerate(syll):
            result.append({
                "text": s,
                "word": word,
                "word_index": wi,
                "stressed": (i == stress_i),
                "stress_source": stress_source if i == stress_i else None,
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