"""
rhythm.py — Analisi ritmica del testo poetico italiano.

Sostituisce prosody.py come modulo di sillabazione/accento/ritmo della
pipeline Prosody2Music. Terza versione: nasce dal confronto a tre tra
Fase1.py/Accenti.py (la versione originale, funzionante, con spaCy),
prosody.py (la versione "leggera" senza spaCy) e una prima riscrittura
di rhythm.py — prendendo di ognuna solo le parti che si sono rivelate
davvero migliori delle alternative, invece di sceglierne una intera.

Cosa viene da dove, e perché
-----------------------------
- Sillabazione: motore a regole dittongo/iato (di prosody.py), usato come
  fonte PRIMARIA — non pyphen. Verificato empiricamente sulla versione
  precedente (Fase1.py) che pyphen come sillabatore primario sbaglia
  sistematicamente lo iato (es. "poesia" -> "poe-sia" invece di
  "po-e-si-a", perché le sue regole sono pensate per l'a-capo tipografico,
  non per la sillaba metrica). pyphen resta come ultima risorsa, e solo
  quando il conteggio reale di phon_italia.py è disponibile e non
  combacia con l'euristica si tenta prima una riconciliazione mirata
  (reconcile_syllable_count) spezzando la sillaba più "grassa" a un
  confine vocalico interno.

- Accento, catena a 5 livelli:
    1) accento grafico esplicito — nessuna ambiguità, priorità assoluta.
    2) lookup esatto in phon_italia.py (phonItaliaR, dato annotato reale,
       120k forme).
    3) Random Forest (scikit-learn), addestrato su un dataset UNITO:
       il lessico di phon_italia.load_lexicon() PIÙ il file Q2Stress a
       livello di parola (lexElem.txt / "word forms.txt", con colonne
       word/StressPattern/SumSylls), caricato da
       q2stress.load_wordlevel_dataframe() — stessa risoluzione percorsi
       (../Progetto/Q2Stress/...) delle tabelle per desinenza dello
       stesso modulo. Usare entrambe le fonti (non solo phonItaliaR come
       nella riscrittura precedente, non solo Q2Stress come nella bozza
       intermedia) dà al RF più esempi per generalizzare sulle parole
       assenti da entrambi i dizionari, che è esattamente il suo scopo.
    4) predizione statistica di q2stress.py (percentuali ULT/PULT/APULT/
       PAPULT per desinenza di 3 lettere) — usata nel ruolo per cui è
       stata scritta: fallback quando la parola non è in nessun lessico
       né (secondo il RF) riconoscibile con sicurezza. Ritorna sempre una
       risposta (grazie ai priori di default), quindi l'euristica fissa
       "parola piana" scatta solo se manca anche questo.
    5) euristica di default (parola piana) — ultimissima rete di sicurezza.

- Pause: tokenizzatore leggero a regex (niente spaCy, per restare
  coerenti con prosody.py — l'analisi sintattica/semantica sta già in
  emotion.py e non va duplicata qui):
    , ; :        -> pausa breve   (pause_short)
    . ! ?        -> pausa lunga   (pause_long)
    ... / …      -> sospensione   (pause_suspension)
    fine verso   -> pausa di verso, con un'euristica per l'enjambement:
                    se il verso finisce già con una pausa forte, non se ne
                    aggiunge un'altra; se finisce senza punteggiatura su
                    una parola "di soglia" (preposizione/articolo/
                    congiunzione/pronome clitico — segnale tipico di
                    verso "aperto" sul successivo), la pausa è attenuata
                    (pause_verse_soft) invece che piena. Non è vero
                    riconoscimento sintattico dell'enjambement (servirebbe
                    un parser), ma un'euristica lessicale a basso costo.

  Le pause sono in una lista PARALLELA ("pauses"), non dentro
  "syllables"/"rhythm" — che restano identici, byte per byte nella
  forma, a prosody.py originale. Motivo: music_transformer.py itera
  verse["rhythm"] e genera SEMPRE una nota intonata per ogni valore
  (nessun concetto di silenzio); mettere le pause dentro "rhythm" ci
  produrrebbe una nota intonata al posto del silenzio, un bug reale,
  non solo di stile. In compenso, la nuova funzione di export MIDI in
  fondo a questo file (vedi sotto) le sa leggere e le rende come veri
  silenzi (music21 Rest).

- Analisi semantica: ASSENTE per scelta — resta in emotion.py, che fa
  già un lavoro più ricco (valenza/arousal/tenerezza, lessico curato +
  NRC) di quanto farebbe una riscrittura qui.

- NOVITÀ rispetto a tutte le versioni precedenti di rhythm.py/prosody.py:
  export MIDI dello scheletro ritmico in fondo al file (build_midi_from_
  poem_analysis), ripreso concettualmente da build_midi_from_skeleton di
  Fase1.py (stesse durate, stesso supporto batteria via canale GM 10) ma
  adattato alla struttura a "verso" di analyze_poem() invece che alla
  lista piatta di Fase1.py, e con la lettura delle pause dalla lista
  "pauses" invece che da flag has_pause_after/has_sentence_end.

Interfaccia esposta
--------------------
analyze_poem(text) -> lista di versi:
    {"text": str, "syllables": [...], "rhythm": [...], "pauses": [...]}
"syllables"/"rhythm" hanno la stessa identica forma di prosody.py, quindi
in main.py basta cambiare `from prosody import analyze_poem` in
`from rhythm import analyze_poem` senza toccare music_transformer.py.

build_midi_from_poem_analysis(poem_analysis, output_path, ...) esporta lo
stesso risultato di analyze_poem() in un file .mid (richiede music21).

Il Random Forest viene addestrato una sola volta per processo (cache a
livello di modulo), non a ogni chiamata.

REQUISITI
---------
    pip install pandas scikit-learn
    pip install pyphen         # opzionale, fallback sillabazione
    pip install music21        # opzionale, solo per l'export MIDI
    phon_italia.py, q2stress.py disponibili nel path del progetto
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "\nPer usare rhythm.py è necessario installare scikit-learn.\n\n"
        "Esegui:\n"
        "    pip install scikit-learn pandas\n"
    ) from exc

try:
    import pyphen
    _PYPHEN_DIC = pyphen.Pyphen(lang="it_IT")
except ImportError:
    _PYPHEN_DIC = None

try:
    import phon_italia
    _PHON_ITALIA_AVAILABLE = True
except ImportError:
    _PHON_ITALIA_AVAILABLE = False

try:
    import q2stress
    _Q2STRESS_AVAILABLE = True
except ImportError:
    _Q2STRESS_AVAILABLE = False


# ============================================================
# COSTANTI LINGUISTICHE (invariate da prosody.py)
# ============================================================

VOWELS = "aeiouàèéìòùáíóúâêîôû"
STRONG_VOWELS = "aeoàèéìòùáíóú"  # a,e,o sono vocali "forti" (aprono iato)
WEAK_VOWELS = "iu"

DIGRAPHS = ["ch", "gh", "gn", "gl"]
# "sc" seguito da e/i è un digramma (scena, scivolo); altrimenti no (scala)

ACCENTED_MAP = {
    "à": "a", "è": "e", "é": "e", "ì": "i", "ò": "o", "ù": "u",
    "á": "a", "í": "i", "ó": "o", "ú": "u",
}

_VOWELS_SET = set(VOWELS)


def norm_word(word: str) -> str:
    text = str(word).strip().lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-zàèéìòùáíóù]", "", text)
    return text


def _is_vowel(ch: str) -> bool:
    return ch.lower() in VOWELS


def strip_punct(word: str) -> str:
    return re.sub(r"[^\wàèéìòùáíóúâêîôû]", "", word, flags=re.UNICODE)

# ============================================================
# SILLABAZIONE A REGOLE (invariata da prosody.py — fonte PRIMARIA,
# non pyphen: vedi nota in testa al file sul problema dello iato)
# ============================================================

def _syllabify_word_heuristic(word: str):
    w = word.lower()
    n = len(w)
    if n == 0:
        return []

    chars = list(w)

    def is_v(idx):
        return idx < n and _is_vowel(chars[idx])

    idx = 0
    syll_start = 0
    syllables = []
    while idx < n:
        j = idx
        while j < n and not is_v(j):        # individuazione del nucleo vocalico
            j += 1
        if j >= n:
            break
        k = j
        while k + 1 < n and is_v(k + 1):
            v1, v2 = chars[k], chars[k + 1]
            if v1 in STRONG_VOWELS and v2 in STRONG_VOWELS:
                break                       # iato: due vocali forti restano in sillabe separate
            k += 1
        nucleus_end = k

        c = nucleus_end + 1
        cons_start = c
        while c < n and not is_v(c):        # individuazione del gruppo consonantico
            c += 1
        cons = "".join(chars[cons_start:c])

        if c >= n:                          # ultima sillaba della parola
            syll = "".join(chars[syll_start:n])
            syllables.append(syll)
            syll_start = n
            idx = n
            break

        if len(cons) == 0:                  # la sillaba termina con una vocale: separa uno iato
            split_at = cons_start
        elif len(cons) == 1:                # la sillaba termina con una vocale: è seguita da una consonante
            split_at = cons_start
        else:
            two = cons[:2]
            if two in DIGRAPHS or (two == "sc" and c < n and chars[c].lower() in "ei"):     # controllo digramma
                split_at = cons_start
            elif cons[0] == cons[1]:                            # ripetizione di consonante
                split_at = cons_start + 1
            elif cons[-1] in "lr" and len(cons) == 2:
                split_at = cons_start
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
    """Solo ultima risorsa — vedi nota in testa al file."""
    if _PYPHEN_DIC is None:
        return None
    w = word.lower()
    if not w:
        return None
    hyphenated = _PYPHEN_DIC.inserted(w)
    syllables = hyphenated.split("-")
    return syllables if syllables else None


def _split_at_internal_vowel(syllable: str):
    for i in range(1, len(syllable)):
        if _is_vowel(syllable[i - 1]) and _is_vowel(syllable[i]):
            return [syllable[:i], syllable[i:]]
    return None


def reconcile_syllable_count(word: str, syllables: list, target_count: int):
    syllables = list(syllables)
    guard = 0
    while len(syllables) < target_count and guard < 5:
        guard += 1
        candidates = sorted(
            range(len(syllables)),
            key=lambda i: -sum(1 for c in syllables[i] if _is_vowel(c)),
        )
        split_done = False
        for i in candidates:
            parts = _split_at_internal_vowel(syllables[i])
            if parts:
                syllables = syllables[:i] + parts + syllables[i + 1:]
                split_done = True
                break
        if not split_done:
            break
    return syllables


def syllabify_word(word: str):
    """
    Priorità: motore a regole (dittongo/iato) -> riconciliazione contro
    phon_italia.lookup() se il conteggio reale non combacia -> pyphen
    come ultimissima risorsa.
    """
    heuristic = _syllabify_word_heuristic(word)

    if _PHON_ITALIA_AVAILABLE:
        entry = phon_italia.lookup(word)
        if entry is not None and entry["num_syll"] != len(heuristic):
            reconciled = reconcile_syllable_count(word, heuristic, entry["num_syll"])
            if len(reconciled) == entry["num_syll"]:
                return reconciled
            pyphen_syll = _syllabify_word_pyphen(word)
            if pyphen_syll and len(pyphen_syll) == entry["num_syll"]:
                return pyphen_syll
            return reconciled

    if heuristic:
        return heuristic

    pyphen_syll = _syllabify_word_pyphen(word)
    return pyphen_syll or [word.lower()]


# ============================================================
# FEATURE PER IL RANDOM FOREST (identiche in tutte le versioni)
# ============================================================

def build_word_features(word: str, syll_count: int) -> dict:
    """syll_count è sempre passato dal chiamante — un'unica fonte di
    verità per il conteggio sillabe, niente doppia euristica interna."""
    w = norm_word(word)
    letters = [c for c in w if c.isalpha()]

    vowel_count = sum(char in _VOWELS_SET for char in letters)
    consonant_count = max(len(letters) - vowel_count, 0)

    return {
        "word_len": len(letters),
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "vowel_ratio": vowel_count / max(len(letters), 1),
        "syll_count": max(1, syll_count),
        "ends_with_vowel": int(bool(letters) and letters[-1] in _VOWELS_SET),
        "ends_with_consonant": int(bool(letters) and letters[-1] not in _VOWELS_SET),
        "contains_ia": int("ia" in w),
        "contains_ie": int("ie" in w),
        "contains_io": int("io" in w),
        "contains_iu": int("iu" in w),
        "contains_ua": int("ua" in w),
        "contains_ue": int("ue" in w),
        "contains_ui": int("ui" in w),
        "contains_uo": int("uo" in w),
        "contains_ta": int("ta" in w),
        "contains_te": int("te" in w),
        "contains_ti": int("ti" in w),
        "contains_to": int("to" in w),
        "contains_tu": int("tu" in w),
        "last1": w[-1] if w else "",
        "last2": w[-2:] if len(w) >= 2 else w,
        "last3": w[-3:] if len(w) >= 3 else w,
        "last4": w[-4:] if len(w) >= 4 else w,
    }


# ============================================================
# COSTRUZIONE DEL TRAINING FRAME DAL DATASET Q2STRESS WORD-LEVEL
# ============================================================
# Il caricamento del file (risoluzione percorsi, lettura TSV, cache) sta
# ora in q2stress.py — load_wordlevel_dataframe()/find_wordlevel_dataset()
# — insieme alle tabelle per desinenza, dato che appartengono comunque al
# dataset Q2Stress nel suo complesso. Qui resta solo la trasformazione in
# feature per il Random Forest (build_word_features), che è specifica di
# rhythm.py e non ha senso spostare in q2stress.py.

def build_training_frame_from_q2stress_wordlevel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stessa logica di Fase1.py: colonna 'word' + 'StressPattern' (già
    0-based, 0=ultima sillaba) + 'SumSylls' (conteggio reale, ha
    precedenza sulla stima quando presente).
    """
    rows = []
    for _, row in df.iterrows():
        word = row.get("word")
        if word is None:
            continue
        word = norm_word(word)
        if not word:
            continue

        target = row.get("StressPattern")
        if pd.isna(target):
            continue
        try:
            target = int(float(target))
        except (TypeError, ValueError):
            continue
        if target not in {0, 1, 2, 3}:
            continue

        syll_value = row.get("SumSylls")
        syll_count = None
        if pd.notna(syll_value):
            try:
                syll_count = max(1, int(float(syll_value)))
            except (TypeError, ValueError):
                syll_count = None
        if syll_count is None:
            syll_count = max(1, len(_syllabify_word_heuristic(word)))

        features = build_word_features(word, syll_count)
        features["target"] = target
        rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# TRAINING DEL RANDOM FOREST SU DATASET UNITO
# (lessico phon_italia + file Q2Stress word-level, se disponibile)
# ============================================================

def build_training_frame_from_lexicon(lexicon: dict) -> pd.DataFrame:
    """Dal lessico di phon_italia.load_lexicon() (word -> stress_from_end/
    num_syll/freq). stress_from_end è 1-based (1=ultima); qui convertito
    a 0-based (0=ultima), stessa convenzione di StressPattern in Q2Stress
    word-level, così le due fonti si combinano senza disallineamenti."""
    rows = []
    for word, entry in lexicon.items():
        stress_from_end = entry.get("stress_from_end")
        num_syll = entry.get("num_syll")
        if not stress_from_end or not num_syll:
            continue
        target = max(0, min(3, stress_from_end - 1))
        features = build_word_features(word, num_syll)
        features["target"] = target
        rows.append(features)
    return pd.DataFrame(rows)


def train_model(frame: pd.DataFrame, balance_strength: float = 0.5) -> Pipeline:
    X = frame.drop(columns=["target"]).copy()
    y = frame["target"]

    for col in ["last1", "last2", "last3", "last4"]:
        if col in X.columns:
            X[col] = X[col].astype(str)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    class_counts = y.value_counts()
    n_samples = len(y)
    n_classes = len(class_counts)

    if balance_strength <= 0.0:
        class_weight = None
    else:
        full_balanced = {
            cls: n_samples / (n_classes * count) for cls, count in class_counts.items()
        }
        class_weight = {cls: weight ** balance_strength for cls, weight in full_balanced.items()}

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight=class_weight,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(X, y)
    return model


def predict_stress_class(word: str, model: Pipeline, syll_count: int) -> int:
    features_df = pd.DataFrame([build_word_features(word, syll_count)])
    max_valid_class = min(3, max(0, syll_count - 1))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)[0]
        classes = model.named_steps["classifier"].classes_
        probabilities = {int(cls): float(prob) for cls, prob in zip(classes, proba)}
        valid_probs = {c: p for c, p in probabilities.items() if c <= max_valid_class}
        if valid_probs:
            return max(valid_probs, key=valid_probs.get)

    return min(int(model.predict(features_df)[0]), max_valid_class)


# ============================================================
# CACHE DEL MODELLO A LIVELLO DI MODULO
# ============================================================

_MODEL_CACHE = {"model": None, "loaded": False}


def get_model(class_balance: float = 0.5, q2stress_wordlevel_path: Optional[str] = None,
              force_reload: bool = False, verbose: bool = False) -> Optional[Pipeline]:
    """
    Ritorna il Random Forest addestrato sul dataset unito (lessico
    phon_italia + file Q2Stress word-level, se trovato), cache dopo il
    primo train. Se nessuna delle due fonti è disponibile, ritorna None
    e find_stress_index scende semplicemente al passo successivo della
    catena (q2stress per desinenza, poi l'euristica fissa).
    """
    if not force_reload and _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE["model"]

    frames = []

    if _PHON_ITALIA_AVAILABLE:
        lexicon = phon_italia.load_lexicon(verbose=verbose)
        if lexicon:
            frames.append(build_training_frame_from_lexicon(lexicon))

    if _Q2STRESS_AVAILABLE:
        df_wordlevel = q2stress.load_wordlevel_dataframe(
            explicit_path=q2stress_wordlevel_path, verbose=verbose
        )
        if df_wordlevel is not None:
            frame_wordlevel = build_training_frame_from_q2stress_wordlevel(df_wordlevel)
            if not frame_wordlevel.empty:
                frames.append(frame_wordlevel)
                if verbose:
                    print(f"[rhythm] Q2Stress word-level: {len(frame_wordlevel)} righe usate per il training")

    model = None
    if frames:
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        if verbose:
            print(f"[rhythm] training set combinato: {len(combined)} righe")
        model = train_model(combined, balance_strength=class_balance)
    elif verbose:
        print("[rhythm] nessuna fonte di training disponibile per il Random Forest")

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["loaded"] = True
    return model


# ============================================================
# ACCENTO: CATENA DI PRIORITÀ A 5 LIVELLI
# ============================================================

def find_stress_index(syllables: list, word: str, model: Optional[Pipeline] = None):
    """
    Ritorna (indice 0-based della sillaba tonica, fonte):
      1. accento grafico esplicito
      2. lookup esatto in phon_italia (phonItaliaR)
      3. Random Forest (addestrato su lessico + Q2Stress word-level)
      4. predizione statistica di q2stress.py (desinenza)
      5. euristica di default (parola piana)
    """
    for i, s in enumerate(syllables):
        if any(ch in ACCENTED_MAP for ch in s):
            return i, "graphic_accent"

    n = len(syllables)

    if _PHON_ITALIA_AVAILABLE:
        idx = phon_italia.stress_index_for_syllables(word, syllables)
        if idx is not None:
            return idx, "phonitalia"

    if model is not None:
        stress_class = predict_stress_class(word, model, n)
        idx = n - 1 - stress_class
        if 0 <= idx < n:
            return idx, "random_forest"

    if _Q2STRESS_AVAILABLE:
        idx, confidence, source = q2stress.stress_index_for_syllables(word, syllables)
        if source == "q2stress":
            return idx, f"q2stress ({confidence:.0%})"

    if n == 1:
        return 0, "heuristic"
    return n - 2, "heuristic"


# ============================================================
# TOKENIZZAZIONE VERSO CON PUNTEGGIATURA (regex, niente spaCy)
# ============================================================

_WORD_RE = r"[A-Za-zàèéìòùáíóúâêîôûÀ-Ú']+"
_WORD_OR_PUNCT_RE = re.compile(_WORD_RE + r"|\.\.\.|…|[.,;:!?]")

_PAUSE_SHORT_PUNCT = {",", ";", ":"}
_PAUSE_LONG_PUNCT = {".", "!", "?"}
_PAUSE_SUSPENSION_PUNCT = {"...", "…"}
_ALL_PUNCT_TOKENS = _PAUSE_SHORT_PUNCT | _PAUSE_LONG_PUNCT | _PAUSE_SUSPENSION_PUNCT

# durate relative (1 impulso = croma, 0.5 quarterLength — vedi export MIDI)
PAUSE_DURATIONS = {
    "pause_short": 1,
    "pause_long": 3,
    "pause_suspension": 4,
    "pause_verse": 2,
    "pause_verse_soft": 1,
}

_ENJAMBMENT_PRONE_WORDS = {
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "che", "e", "ed", "o", "od", "ma", "se", "non",
    "mi", "ti", "si", "ci", "vi", "ne",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "del", "dello", "della", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "col", "coi", "sul", "sulla", "sui",
}


def _pause_kind(punct: Optional[str]) -> Optional[str]:
    if punct is None:
        return None
    if punct in _PAUSE_SUSPENSION_PUNCT:
        return "pause_suspension"
    if punct in _PAUSE_LONG_PUNCT:
        return "pause_long"
    if punct in _PAUSE_SHORT_PUNCT:
        return "pause_short"
    return None


def _tokenize_verse(verse: str) -> list[dict]:
    """Ritorna una lista di {"word": str, "punct_after": str|None}."""
    raw_tokens = _WORD_OR_PUNCT_RE.findall(verse)
    tokens = []
    i = 0
    n = len(raw_tokens)
    while i < n:
        tok = raw_tokens[i]
        if tok in _ALL_PUNCT_TOKENS:
            i += 1
            continue
        word = tok
        punct_after = None
        if i + 1 < n and raw_tokens[i + 1] in _ALL_PUNCT_TOKENS:
            punct_after = raw_tokens[i + 1]
            i += 2
        else:
            i += 1
        tokens.append({"word": word, "punct_after": punct_after})
    return tokens


# ============================================================
# ANALISI DI VERSO / POESIA
# ============================================================

def analyze_verse(verse: str, model: Optional[Pipeline] = None):
    """
    Ritorna (syllables, pauses):
      syllables: {"text","word","word_index","stressed","stress_source"}
                 — stessa forma di prosody.py, nessuna pausa qui dentro.
      pauses: eventi pausa interni al verso, ognuno
              {"after_syllable_index": int, "kind": str}.
    """
    if model is None:
        model = get_model()

    tokens = _tokenize_verse(verse)
    syllables = []
    pauses = []

    for wi, tok in enumerate(tokens):
        word = tok["word"]
        syll = syllabify_word(word)
        stress_i, stress_source = find_stress_index(syll, word, model)

        for i, s in enumerate(syll):
            syllables.append({
                "text": s,
                "word": word,
                "word_index": wi,
                "stressed": (i == stress_i),
                "stress_source": stress_source if i == stress_i else None,
            })

        pause_kind = _pause_kind(tok["punct_after"])
        if pause_kind:
            pauses.append({"after_syllable_index": len(syllables) - 1, "kind": pause_kind})

    return syllables, pauses


def rhythm_pattern(syllables: list) -> list:
    """2 se tonica, 1 se atona. Nessuna pausa qui dentro."""
    return [2 if s["stressed"] else 1 for s in syllables]


def _verse_end_pause_kind(syllables: list, pauses: list) -> Optional[str]:
    """Euristica per la pausa di fine verso, enjambement-aware. Se il
    verso finisce già con QUALSIASI pausa da punteggiatura (breve, lunga
    o sospensione), non se ne aggiunge un'altra sopra — la punteggiatura
    segnala già la fine del respiro, sommarci una pausa di verso
    produrrebbe due silenzi consecutivi nello stesso punto (es. virgola
    a fine riga + pausa di verso), che è un doppio conteggio indesiderato,
    non un rinforzo intenzionale."""
    if not syllables:
        return "pause_verse"

    last_idx = len(syllables) - 1
    trailing = [p for p in pauses if p["after_syllable_index"] == last_idx]
    if trailing:
        return None  # già una pausa da punteggiatura, non raddoppiare

    last_word = syllables[-1]["word"]
    if last_word and norm_word(last_word) in _ENJAMBMENT_PRONE_WORDS:
        return "pause_verse_soft"

    return "pause_verse"


def analyze_poem(text: str) -> list[dict]:
    """
    Analizza un'intera poesia. Ogni verso:
        {"text": str, "syllables": [...], "rhythm": [...], "pauses": [...]}
    "syllables"/"rhythm" sono identici a prosody.py (compatibili con
    music_transformer.py senza modifiche); "pauses" è la lista parallela.
    """
    model = get_model()
    verses = [v for v in text.strip().split("\n") if v.strip()]
    analyzed = []

    for v in verses:
        syllables, pauses = analyze_verse(v, model=model)
        rhythm = rhythm_pattern(syllables)

        verse_pause_kind = _verse_end_pause_kind(syllables, pauses)
        if verse_pause_kind:
            pauses.append({
                "after_syllable_index": len(syllables) - 1,
                "kind": verse_pause_kind,
            })

        analyzed.append({"text": v, "syllables": syllables, "rhythm": rhythm, "pauses": pauses})

    return analyzed


# ============================================================
# EXPORT MIDI DELLO SCHELETRO RITMICO
# ============================================================
# Ripreso da build_midi_from_skeleton di Fase1.py: stesse durate, stesso
# supporto batteria (canale GM 10), ma che legge la struttura a verso di
# analyze_poem() (syllables + rhythm + pauses) invece della lista piatta
# di Fase1.py. Non è ancora melodia — ogni sillaba forte usa strong_pitch,
# ogni sillaba debole weak_pitch (altezza fissa): è lo scheletro ritmico
# di base che verrà passato al Music Transformer (Fase 2) come
# primer/riferimento ritmico.

GM_DRUM_NOTES = {
    "bass_drum": 36,
    "snare": 38,
    "closed_hihat": 42,
    "open_hihat": 46,
    "crash": 49,
    "ride": 51,
}

_UNIT_QUARTER_LENGTH = 0.5  # 1 impulso (vedi PAUSE_DURATIONS) = croma


def _poem_analysis_to_events(poem_analysis: list[dict]) -> list[dict]:
    """
    Appiattisce l'output di analyze_poem() in una sequenza ordinata di
    eventi {"kind": "note"|"rest", "strong": bool, "duration_units": int},
    intercalando correttamente sillabe (da "syllables"/"rhythm") e pause
    (da "pauses", posizionate dopo l'indice di sillaba corretto).
    """
    events = []
    for verse in poem_analysis:
        syllables = verse["syllables"]
        rhythm = verse["rhythm"]
        pauses = verse["pauses"]

        pauses_by_index: dict[int, list[str]] = {}
        for p in pauses:
            pauses_by_index.setdefault(p["after_syllable_index"], []).append(p["kind"])

        for i, (syll, dur_units) in enumerate(zip(syllables, rhythm)):
            events.append({
                "kind": "note",
                "strong": bool(syll["stressed"]),
                "duration_units": dur_units,
            })
            for pause_kind in pauses_by_index.get(i, []):
                events.append({
                    "kind": "rest",
                    "strong": False,
                    "duration_units": PAUSE_DURATIONS[pause_kind],
                })

    return events


def build_midi_from_poem_analysis(
    poem_analysis: list[dict],
    output_path,
    bpm: int = 100,
    strong_pitch: str = "C4",
    weak_pitch: str = "A3",
    strong_velocity: int = 100,
    weak_velocity: int = 60,
    use_drums: bool = False,
    drum_strong: int = GM_DRUM_NOTES["bass_drum"],
    drum_weak: int = GM_DRUM_NOTES["closed_hihat"],
) -> None:
    """
    Esporta lo scheletro ritmico prodotto da analyze_poem() come file
    MIDI. Richiede: pip install music21

    Parametri
    ---------
    poem_analysis : l'output di analyze_poem(text)
    output_path    : path del file .mid da scrivere
    bpm            : tempo in battiti al minuto
    strong_pitch / weak_pitch : altezza fissa per sillabe toniche/atone
                     (ignorata se use_drums=True)
    use_drums      : se True, usa il canale percussioni GM (canale 10)
                     invece di note intonate — drum_strong per le toniche
                     (default: grancassa), drum_weak per le atone
                     (default: hi-hat chiuso). Vedi GM_DRUM_NOTES.

    Durate (in impulsi, 1 impulso = croma = 0.5 quarterLength):
        sillaba tonica   -> 2 impulsi (1.0 quarterLength)
        sillaba atona    -> 1 impulso (0.5 quarterLength)
        pausa breve      -> 1 impulso di silenzio (dopo , ; :)
        pausa lunga      -> 3 impulsi di silenzio (dopo . ! ?)
        sospensione      -> 4 impulsi di silenzio (dopo ... / …)
        pausa di verso   -> 2 impulsi di silenzio (fine riga)
        pausa di verso attenuata (enjambement) -> 1 impulso di silenzio
    """
    try:
        from music21 import stream, note, tempo as m21tempo, instrument
    except ImportError as exc:
        raise SystemExit(
            "\nPer esportare il MIDI serve music21.\n\nEsegui:\n"
            "    pip install music21\n"
        ) from exc

    events = _poem_analysis_to_events(poem_analysis)

    part = stream.Part()
    part.append(m21tempo.MetronomeMark(number=bpm))

    if use_drums:
        part.insert(0, instrument.UnpitchedPercussion())

    for event in events:
        quarter_length = event["duration_units"] * _UNIT_QUARTER_LENGTH

        if event["kind"] == "note":
            if use_drums:
                n = note.Note()
                n.pitch.midi = drum_strong if event["strong"] else drum_weak
            else:
                n = note.Note(strong_pitch if event["strong"] else weak_pitch)
            n.quarterLength = quarter_length
            n.volume.velocity = strong_velocity if event["strong"] else weak_velocity
            part.append(n)
        else:  # rest
            r = note.Rest()
            r.quarterLength = quarter_length
            part.append(r)

    part.write("midi", fp=str(output_path))

    total_quarter_length = sum(n.quarterLength for n in part.notesAndRests)
    expected_seconds = total_quarter_length * (60.0 / bpm)
    print(
        f"[rhythm] MIDI scritto in {output_path} — "
        f"{total_quarter_length:.2f} quarterLength a {bpm} bpm "
        f"(~{expected_seconds:.1f}s)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analisi ritmica di una poesia italiana + export opzionale in MIDI."
    )
    parser.add_argument("poem_file", nargs="?", default=None, help="File .txt della poesia (opzionale: demo se assente)")
    parser.add_argument("--midi-output", default=None, help="Se specificato, esporta anche un file .mid")
    parser.add_argument("--bpm", type=int, default=100, help="Tempo per l'export MIDI (default: 100)")
    parser.add_argument("--drums", action="store_true", help="Usa il canale percussioni GM invece del pianoforte")

    args = parser.parse_args()

    if args.poem_file:
        with open(args.poem_file, "r", encoding="utf-8") as f:
            demo = f.read()
    else:
        demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"

    analysis = analyze_poem(demo)

    for verse in analysis:
        syll_str = " | ".join(
            (s["text"].upper() if s["stressed"] else s["text"])
            for s in verse["syllables"]
        )
        print(verse["text"])
        print(" ", syll_str)
        print("  ritmo:", verse["rhythm"])
        print("  pause:", verse["pauses"])
        print()

    if args.midi_output:
        build_midi_from_poem_analysis(
            analysis, args.midi_output, bpm=args.bpm, use_drums=args.drums
        )
