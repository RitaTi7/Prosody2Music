'''
emotion.py 
versione 1-> ha modificato aggiungendo anche NRC emolex VAD
'''


import re
import logging

from rhythm import strip_punct

logger = logging.getLogger(__name__)

try:
    import nrc_emolex
    _NRC_AVAILABLE = True
except ImportError:
    _NRC_AVAILABLE = False

try:
    import nrc_vad
    _VAD_AVAILABLE = True
except ImportError:
    _VAD_AVAILABLE = False

try:
    import simplemma
    _LEMMATIZER_AVAILABLE = True
except ImportError:
    _LEMMATIZER_AVAILABLE = False

print(f"[DEBUG emotion.py] lemmatizer={_LEMMATIZER_AVAILABLE}  nrc_emolex={_NRC_AVAILABLE}  nrc_vad={_VAD_AVAILABLE}")

# --- FIX: peso del lessico a mano rispetto alle fonti NRC quando una
# parola è presente in entrambi. Il lessico a mano è più preciso (curato
# parola per parola sul dominio poetico); le fonti NRC hanno copertura
# più ampia. 0.6/0.4 invece di 0.5/0.5 dà priorità alla fonte più
# affidabile senza ignorare comunque il segnale esterno.
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
    "calma": (0.6, -0.6, 0.4),
    "calmo": (0.6, -0.6, 0.4),
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


def _tenderness_from_categories(cat_result):
    """Estrae il contributo di tenerezza da un risultato di nrc_emolex.score_word
    (che lavora per categorie), oppure 0.0 se non disponibile."""
    return cat_result["tenderness"] if cat_result else 0.0


def _lookup_word(word: str):
    """
    Combina più fonti per una parola, in ordine di affidabilità:

      1. lessico curato a mano (hand)               -> valenza/arousal/tenerezza
      2. NRC VAD Lexicon (vad, valori continui)      -> valenza/arousal
      3. NRC Emotion Lexicon (cat, categorie binarie) -> fallback per
         valenza/arousal SOLO se vad non ha la parola, e sempre come
         fonte per la tenerezza (asse non presente nel VAD lexicon)

    Il VAD lexicon è preferito a EmoLex per arousal/valenza perché è
    annotato in modo continuo e diretto, mentre EmoLex lo deriva da 8
    categorie sbilanciate (6 su 8 con arousal positivo, vedi nrc_emolex.py)
    e tendeva a impedire arousal negativi anche per parole calme.

    Prova prima il lemma (per catturare variazioni morfologiche), poi la
    forma originale come fallback. Ritorna (valence, arousal, tenderness,
    source_label) oppure None se la parola non è in nessuna fonte.
    """
    lemma = _lemmatize(word)

    hand = LEXICON.get(lemma)
    if hand is None:
        hand = LEXICON.get(word)

    vad = None
    if _VAD_AVAILABLE:
        vad = nrc_vad.score_word(lemma)
        if vad is None:
            vad = nrc_vad.score_word(word)

    cat = None
    if _NRC_AVAILABLE:
        cat = nrc_emolex.score_word(lemma)
        if cat is None:
            cat = nrc_emolex.score_word(word)

    if hand is not None and vad is not None:
        v = hand[0] * HAND_WEIGHT + vad["valence"] * NRC_WEIGHT
        a = hand[1] * HAND_WEIGHT + vad["arousal"] * NRC_WEIGHT
        t_nrc = _tenderness_from_categories(cat)
        t = hand[2] * HAND_WEIGHT + t_nrc * NRC_WEIGHT
        return v, a, t, "hand+vad"

    if hand is not None and cat is not None:
        # vad non disponibile per questa parola: fallback sulla proiezione
        # categoriale di EmoLex, come prima.
        v = hand[0] * HAND_WEIGHT + cat["valence"] * NRC_WEIGHT
        a = hand[1] * HAND_WEIGHT + cat["arousal"] * NRC_WEIGHT
        t = hand[2] * HAND_WEIGHT + cat["tenderness"] * NRC_WEIGHT
        return v, a, t, "hand+nrc_cat"

    if hand is not None:
        return hand[0], hand[1], hand[2], "hand"

    if vad is not None:
        t = _tenderness_from_categories(cat)
        source = "vad+catT" if cat is not None else "vad"
        return vad["valence"], vad["arousal"], t, source

    if cat is not None:
        return cat["valence"], cat["arousal"], cat["tenderness"], "nrc_cat"

    return None


def analyze_emotion(text: str):
    """
    Ritorna un embedding emotivo aggregato: dict con valence, arousal,
    tenderness, una stima di affidabilità (coverage) e i termini che
    hanno contribuito (con la fonte usata per ciascuno).
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
    demo = "Nella quiete della sera, riposa il cuore sereno, lontano dalla tempesta."
    print(analyze_emotion(demo))
