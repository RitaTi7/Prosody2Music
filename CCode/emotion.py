"""
emotion.py — Analisi semantica / emotion embedding per testo poetico italiano.

Approccio: due lessici combinati, entrambi proiettati sugli stessi assi
(valenza, arousal, tenerezza) ispirati al modello circumplex di Russell:

  1. LEXICON — un lessico curato a mano (~70 parole), preciso ma
     con copertura limitata al vocabolario poetico più comune.
  2. nrc_emolex — NRC Emotion Lexicon (Mohammad & Turney, 2013), italiano:
     ~5.400 parole, categorie di emozione (Plutchik) proiettate su
     valenza/arousal/tenerezza. Copertura ampia, dati crowd-sourced e
     tradotti, quindi più rumorosi (vedi nrc_emolex.py).

Scelto NRC come unica fonte esterna (invece di Sentix) perché:
  - misura anche l'arousal (Sentix dà solo un punteggio di polarità),
    e l'arousal guida direttamente tempo/modalità/ampiezza dei salti
    melodici nel Music Transformer — con solo Sentix quell'asse
    resterebbe scoperto;
  - fallisce "meglio" sulle parole cariche: Sentix media su tutti i sensi
    WordNet di un lemma e può annacquare parole fortemente polarizzate
    ma polisemiche (es. "guerra"/"pace" uscivano quasi neutre), mentre
    NRC a volte aggiunge rumore ma il segnale principale resta leggibile.

Quando una parola è in entrambi i lessici, i due contributi vengono
mediati: copertura ampia da NRC, precisione mirata dal lessico curato
dove serve (es. "luce" non ha alcun flag attivo in NRC ma è nel lessico
curato). Quando è solo in uno dei due, si usa quello.

--- FIX (vedi analisi bug arousal sempre positivo) ---------------------
La causa principale dello sbilanciamento era in nrc_emolex.py (fallback
di polarità e mappatura categorie, vedi note in quel file), ma anche qui
c'erano due punti che amplificavano il problema invece di attutirlo:

  1. Nel blending hand+nrc, i due contributi venivano mediati alla pari
     (50/50). Ma il lessico a mano è per costruzione più preciso (curato
     a mano parola per parola), mentre NRC è più rumoroso soprattutto
     sull'arousal (vedi nrc_emolex.py). Una media semplice lasciava che
     il rumore di NRC "tirasse" ogni parola verso il suo arousal anche
     quando il lessico a mano aveva già un valore più affidabile. Ora il
     blending pesa di più il lessico a mano (60/40) quando è disponibile.

  2. Il valore di default restituito quando il testo non contiene
     nessuna parola riconosciuta era arousal = 0.1, non 0.0: un piccolo
     ma sistematico bias positivo applicato a qualunque testo "vuoto"
     dal punto di vista lessicale. Ora il default è 0.0 (arousal
     neutro), coerente con valence e tenderness che erano già a 0.0.

In produzione questo modulo sarebbe sostituito da un vero embedding
neurale (es. sentence-transformer multilingue) proiettato su assi
valence/arousal tramite una testa di regressione; qui usiamo lessici
per avere un sistema interamente offline e deterministico.
"""

import re
import logging

from prosody import strip_punct

logger = logging.getLogger(__name__)

try:
    import nrc_emolex
    _NRC_AVAILABLE = True
except ImportError:
    _NRC_AVAILABLE = False

try:
    import simplemma
    _LEMMATIZER_AVAILABLE = True
except ImportError:
    _LEMMATIZER_AVAILABLE = False

#logger.debug("lemmatizer=%s nrc=%s", _LEMMATIZER_AVAILABLE, _NRC_AVAILABLE)
print(f"[DEBUG emotion.py] lemmatizer={_LEMMATIZER_AVAILABLE}  nrc={_NRC_AVAILABLE}")

# --- FIX: peso del lessico a mano rispetto a NRC quando una parola è
# presente in entrambi. Il lessico a mano è più preciso (curato parola
# per parola sul dominio poetico); NRC ha copertura più ampia ma più
# rumore (vedi nrc_emolex.py). 0.6/0.4 invece di 0.5/0.5 dà priorità alla
# fonte più affidabile senza ignorare comunque il segnale di NRC.
HAND_WEIGHT = 0.6
NRC_WEIGHT = 1.0 - HAND_WEIGHT


def _lemmatize(word: str) -> str:
    if _LEMMATIZER_AVAILABLE:
        return simplemma.lemmatize(word, lang="it")
    return word


# valenza:   -1 (negativo)          .. +1 (positivo)
# arousal:   -1 (calmo)             .. +1 (agitato/intenso)
# tenerezza: -1 (distacco/durezza)  .. +1 (affetto/dolcezza)
#
# Nota di copertura: il lessico originale era sbilanciato verso il
# quadrante (valenza-, arousal+) [rabbia/paura/guerra] e (valenza-,
# arousal-) [tristezza/malinconia], ma quasi vuoto nel quadrante
# (valenza+, arousal-) [calma/serenità] e debole in (valenza+, arousal+)
# [gioia/eccitazione ad alta intensità]. Le voci marcate "# +copertura"
# sono state aggiunte per riequilibrare i quattro quadranti del piano
# valenza/arousal, così che testi con toni diversi possano davvero
# raggiungere tutte le aree del grafico in visualizer.py.
LEXICON = {
    # --- quadrante gioia / affetto (v+, a medio-basso) ---
    "amore": (0.9, 0.5, 0.9), "amare": (0.8, 0.5, 0.8), "amata": (0.8, 0.4, 0.9),
    "cuore": (0.5, 0.5, 0.7), "gioia": (0.9, 0.7, 0.6), "felicità": (0.9, 0.6, 0.5),
    "sorriso": (0.7, 0.3, 0.7), "speranza": (0.7, 0.4, 0.4), "luce": (0.7, 0.4, 0.3),
    "sole": (0.7, 0.5, 0.3), "dolce": (0.6, 0.1, 0.8), "dolcezza": (0.6, 0.1, 0.8),
    "pace": (0.6, -0.4, 0.4), "sogno": (0.6, 0.1, 0.4), "fiore": (0.6, 0.1, 0.4),
    "stelle": (0.6, 0.2, 0.3), "stella": (0.6, 0.2, 0.3), "carezza": (0.7, 0.0, 0.9),
    "vita": (0.4, 0.3, 0.2),

    # --- quadrante gioia / eccitazione, alta intensità (v+, a+) # +copertura ---
    "entusiasmo": (0.8, 0.7, 0.2), "danza": (0.6, 0.6, 0.2), "trionfo": (0.7, 0.7, 0.1),
    "meraviglia": (0.7, 0.5, 0.3),

    # --- quadrante calma / serenità (v+, a-) # +copertura ---
#    "quiete": (0.6, -0.6, 0.4), "serenità": (0.7, -0.5, 0.4),
#    "tranquillità": (0.6, -0.5, 0.3), "riposo": (0.4, -0.5, 0.2),
#    "armonia": (0.6, -0.4, 0.3), "grazia": (0.6, -0.3, 0.5),

    "calma": (0.6, -0.6, 0.4),
    "calmo": (0.6, -0.6, 0.4),
    "calma": (0.6, -0.6, 0.4),
    "pacifico": (0.6, -0.5, 0.4),
    "pacifica": (0.6, -0.5, 0.4),
    "placido": (0.6, -0.6, 0.4),
    "placida": (0.6, -0.6, 0.4),
    "sereno": (0.7, -0.5, 0.5),
    "serena": (0.7, -0.5, 0.5),
    "silenzioso": (0.2, -0.5, 0.2),
    "silenziosa": (0.2, -0.5, 0.2),
    "lieve": (0.5, -0.3, 0.4),
    "lievezza": (0.5, -0.4, 0.5),
    "morbido": (0.5, -0.2, 0.5),
    "morbida": (0.5, -0.2, 0.5),
    "adagio": (0.3, -0.5, 0.2),
    "lentamente": (0.2, -0.4, 0.1),
    "dormire": (0.3, -0.7, 0.3),
    "dorme": (0.3, -0.7, 0.3),
    "sonno": (0.3, -0.7, 0.3),
    "riposa": (0.4, -0.6, 0.3),

    # --- quadrante tristezza / malinconia (v-, a-) ---
    "morte": (-0.9, 0.1, -0.3), "pianto": (-0.7, -0.1, 0.1), "lacrime": (-0.7, -0.1, 0.1),
    "tristezza": (-0.8, -0.3, 0.0), "triste": (-0.7, -0.3, 0.0), "dolore": (-0.8, 0.0, 0.0),
    "angoscia": (-0.8, 0.4, -0.2), "solitudine": (-0.6, -0.3, -0.2),
    "silenzio": (0.0, -0.5, 0.0), "vuoto": (-0.6, -0.1, -0.3), "nulla": (-0.5, -0.2, -0.2),
    "ombra": (-0.3, 0.0, -0.1), "buio": (-0.5, 0.2, -0.2), "notte": (-0.1, 0.1, 0.0),
    "malinconia": (-0.5, -0.4, 0.1), "malinconico": (-0.5, -0.4, 0.1), "malinconica": (-0.5, -0.4, 0.1),
    "nostalgia": (-0.3, -0.3, 0.2), "nostalgico": (-0.3, -0.3, 0.2),
    "rimpianto": (-0.5, -0.2, 0.0), "rimorso": (-0.5, -0.1, -0.1),
    "sconforto": (-0.6, -0.3, -0.1), "desolazione": (-0.7, -0.2, -0.2),
    "rassegnazione": (-0.4, -0.5, 0.0), "abbandono": (-0.6, -0.2, -0.1),
    "malato": (-0.5, -0.2, -0.1), "stanchezza": (-0.4, -0.4, 0.0), "stanco": (-0.4, -0.4, 0.0),
    "sospiro": (-0.3, -0.3, 0.1), "languire": (-0.4, -0.4, 0.0),

    # --- quadrante rabbia / paura / conflitto (v-, a+) ---
    "paura": (-0.6, 0.7, -0.2), "guerra": (-0.9, 0.9, -0.6), "sangue": (-0.7, 0.7, -0.4),
    "odio": (-0.9, 0.7, -0.8), "rabbia": (-0.7, 0.8, -0.5), "furia": (-0.7, 0.9, -0.5),
    "tempesta": (-0.3, 0.9, -0.1), "battaglia": (-0.6, 0.8, -0.4),
    "sbattere": (-0.4, 0.7, -0.3), "strappare": (-0.4, 0.6, -0.3),
    "stringere": (-0.2, 0.6, -0.2), "denti": (-0.2, 0.5, -0.2),
    "grido": (-0.3, 0.8, -0.2), "basta": (-0.3, 0.7, -0.2),

    # --- parole "di natura" e narrative, arousal/valenza più neutri ---
#    "vento": (0.0, 0.5, 0.0), "fuoco": (0.3, 0.8, 0.0), "mare": (0.5, 0.4, 0.2),
#    "cielo": (0.5, 0.2, 0.2), "onda": (0.2, 0.5, 0.0), "muro": (-0.1, 0.2, -0.1),
#    "porta": (0.0, 0.2, 0.0),
#   "cammino": (0.1, 0.3, 0.0),
    "vento": (0.0, 0.0, 0.0),
    "fuoco": (0.2, 0.3, 0.0),
    "mare": (0.4, 0.0, 0.2),
    "cielo": (0.3, 0.0, 0.2),
    "onda": (0.1, 0.1, 0.0),
    "muro": (-0.1, 0.0, -0.1),
    "porta": (0.0, 0.0, 0.0),
    "cammino": (0.1, 0.1, 0.0),
    "oscura": (-0.4, 0.3, -0.1), "smarrita": (-0.5, 0.4, -0.1),
    "selva": (-0.2, 0.3, -0.1), "eterno": (0.3, 0.1, 0.1),
    "eterna": (0.3, 0.1, 0.1), "bellezza": (0.7, 0.3, 0.5), "bella": (0.6, 0.2, 0.5),
    "bello": (0.6, 0.2, 0.4),
}

INTENSIFIERS = {"molto", "tanto", "sempre", "profondamente", "immensamente"}
NEGATORS = {"non", "senza", "mai", "né"}

# Avverbi comuni che possono comparire tra un negatore/intensificatore e la
# parola bersaglio senza interrompere lo scope (es. "non veramente triste").
# Non hanno peso emotivo proprio, quindi vanno trattati come le stopword.
SCOPE_NEUTRAL = {"veramente", "davvero", "proprio", "così", "ancora", "più", "già"}

# Parole funzionali da escludere sempre dal lookup emotivo (articoli,
# preposizioni, congiunzioni, pronomi, forme comuni di essere/avere).
STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "ed", "o", "od", "ma", "che", "chi", "cui", "come", "se",
    "mi", "ti", "si", "ci", "vi", "lui", "lei", "loro", "noi", "voi", "io", "tu",
    "mio", "tua", "suo", "nostro", "vostro", "questo", "quello", "questa", "quella",
    "è", "era", "erano", "sono", "sei", "siamo", "siete", "fu", "fui", "fosti",
    "ha", "hai", "ho", "hanno", "abbiamo", "avete",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "col", "coi", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
} | SCOPE_NEUTRAL


def tokenize(text: str):
    words = re.findall(r"[A-Za-zàèéìòùÀÈÉÌÒÙ']+", text.lower())
    return [strip_punct(w) for w in words if w]


def _lookup_word(word: str):
    """
    Combina lessico curato a mano e NRC Emotion Lexicon per una parola.
    Prova prima il lemma (per catturare variazioni morfologiche), poi
    la forma originale come fallback. Ritorna (valence, arousal,
    tenderness, source_label) oppure None se la parola non è in nessuno
    dei due lessici.
    """
    lemma = _lemmatize(word)

    hand = LEXICON.get(lemma)
    if hand is None:
        hand = LEXICON.get(word)

    nrc = None
    if _NRC_AVAILABLE:
        nrc = nrc_emolex.score_word(lemma)
        if nrc is None:
            nrc = nrc_emolex.score_word(word)

    if hand is not None and nrc is not None:
        # --- FIX: blending pesato 60/40 (hand/nrc) invece di media
        # semplice 50/50 — vedi nota in testa al file.
        v = hand[0] * HAND_WEIGHT + nrc["valence"] * NRC_WEIGHT
        a = hand[1] * HAND_WEIGHT + nrc["arousal"] * NRC_WEIGHT
        t = hand[2] * HAND_WEIGHT + nrc["tenderness"] * NRC_WEIGHT
        return v, a, t, "hand+nrc"
    if hand is not None:
        return hand[0], hand[1], hand[2], "hand"
    if nrc is not None:
        return nrc["valence"], nrc["arousal"], nrc["tenderness"], "nrc"
    return None


def analyze_emotion(text: str):
    """
    Ritorna un embedding emotivo aggregato: dict con valence, arousal,
    tenderness, una stima di affidabilità (coverage) e i termini che
    hanno contribuito (con la fonte usata per ciascuno: "hand", "nrc"
    o "hand+nrc").
    """
    tokens = tokenize(text)
    contributions = []
    n_content_tokens = 0
    negate_next = False
    intensify_next = False

    for tok in tokens:
        if tok in NEGATORS:
            negate_next = True
            continue
        if tok in INTENSIFIERS:
            intensify_next = True
            continue
        if tok in STOPWORDS:
            continue

        n_content_tokens += 1
        entry = _lookup_word(tok)
        if entry is not None:
            v, a, t, source = entry

            if negate_next:
                # La negazione inverte/attenua valenza e arousal, e smorza
                # la tenerezza verso lo zero (negare "amore" non la rende
                # "odio", la rende semplicemente assente).
                v = -v * 0.6
                a = a * 0.5
                t = t * 0.3

            if intensify_next:
                v, a, t = v * 1.3, a * 1.3, t * 1.3
                v = max(-1.0, min(1.0, v))
                a = max(-1.0, min(1.0, a))
                t = max(-1.0, min(1.0, t))

            contributions.append((v, a, t, tok, source))
            # Il reset avviene solo quando la parola è stata effettivamente
            # usata: se una parola sconosciuta si frappone tra negatore/
            # intensificatore e il bersaglio reale, lo scope non va perso.
            negate_next = False
            intensify_next = False

    if not contributions:
        # --- FIX: arousal di default portato da 0.1 a 0.0. Il vecchio
        # valore introduceva un piccolo bias positivo sistematico ogni
        # volta che il testo non conteneva nessuna parola riconosciuta,
        # invece di restituire un punto davvero neutro.
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "tenderness": 0.0,
            "coverage": 0.0,
            "matched": [],
        }

    # Peso = magnitudine del vettore (v, a, t): le parole emotivamente
    # "forti" su uno qualsiasi dei tre assi contano di più nella media,
    # con un peso minimo per non azzerare mai un contributo.
    weights = [max(abs(c[0]), abs(c[1]), abs(c[2]), 0.15) for c in contributions]
    w_sum = sum(weights)

    valence = sum(c[0] * w for c, w in zip(contributions, weights)) / w_sum
    arousal = sum(c[1] * w for c, w in zip(contributions, weights)) / w_sum
    tenderness = sum(c[2] * w for c, w in zip(contributions, weights)) / w_sum

    coverage = len(contributions) / n_content_tokens if n_content_tokens else 0.0

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "tenderness": round(tenderness, 3),
        "coverage": round(coverage, 3),
        "matched": [f"{c[3]} ({c[4]})" for c in contributions],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    print(analyze_emotion(demo))
