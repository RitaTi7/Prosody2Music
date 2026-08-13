#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accenti.py
==========

Classificatore dell'accento tonico italiano + analisi linguistica con spaCy.

INPUT
-----
Il programma legge un normale file .txt contenente testo italiano.
NON è necessario andare a capo dopo ogni parola.

Esempio valido:

    Ciao sono molto felice. Il gatto mangia una mela,
    mentre il cane corre in giardino.

Il testo viene letto interamente e le parole vengono ricavate
automaticamente da spaCy.

OUTPUT
------
Il programma produce un CSV con una riga per ogni parola e aggiunge
le informazioni necessarie per il successivo motore musicale:

- predizione dell'accento del modello
- confidence
- confronto con phonItaliaR
- lemma
- POS
- tag morfosintattico
- dependency
- head / head lemma
- indice della head
- ruolo sintattico
- morfologia
- posizione nella frase
- inizio/fine frase
- punteggiatura prima/dopo
- numero di sillabe stimato
- lunghezza parola
- contesto precedente/successivo
- distanza dalla radice sintattica
- semantica: entità nominate, polarità (positivo/negativo/neutro),
  intensità semantica, similarità col resto della frase, salienza
- caratteristiche utili alla composizione (peso ritmico, tensione
  melodica), ora derivate anche dal significato della parola e non
  più da valori fissi

REQUISITI
---------
    pip install pandas scikit-learn spacy
    python -m spacy download it_core_news_sm

    Per un'analisi semantica più accurata (similarità basata su
    vettori distribuzionali) si consiglia un modello spaCy con
    word embedding pieni, ad esempio:

        python -m spacy download it_core_news_md

    Il programma funziona comunque con it_core_news_sm, ma in quel
    caso la similarità semantica tra le parole non è disponibile
    (i vettori non sono inclusi nel modello "sm") e viene impostata
    a 0.0.

ESECUZIONE
----------
    python Accenti.py testo.txt

oppure:

    python Accenti.py testo.txt --output risultati.csv

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# IMPORT SCIKIT-LEARN
# ============================================================

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "\nPer usare Accenti.py è necessario installare scikit-learn.\n\n"
        "Esegui:\n"
        "    pip install scikit-learn pandas\n"
    ) from exc


# ============================================================
# IMPORT SPACY
# ============================================================

try:
    import spacy
except ImportError as exc:
    raise SystemExit(
        "\nPer usare l'analisi del contesto è necessario installare spaCy.\n\n"
        "Esegui:\n"
        "    pip install spacy\n"
    ) from exc


nlp = None

for _model_name in ("it_core_news_md", "it_core_news_lg", "it_core_news_sm"):
    try:
        nlp = spacy.load(_model_name)
        break
    except OSError:
        continue

if nlp is None:
    raise SystemExit(
        "\nNessun modello italiano di spaCy è installato.\n\n"
        "Esegui almeno:\n"
        "    python -m spacy download it_core_news_sm\n\n"
        "Per l'analisi semantica (similarità tra parole) è consigliato:\n"
        "    python -m spacy download it_core_news_md\n"
    )

# I modelli "sm" non includono vettori distribuzionali: la similarità
# semantica non è calcolabile in modo affidabile in quel caso.
HAS_SEMANTIC_VECTORS = nlp.meta.get("vectors", {}).get("vectors", 0) > 0

if not HAS_SEMANTIC_VECTORS:
    print(
        f"\n[AVVISO] Il modello '{nlp.meta.get('name', '?')}' non include "
        "vettori semantici.\n"
        "La similarità semantica tra parole sarà impostata a 0.0.\n"
        "Per abilitarla: python -m spacy download it_core_news_md\n"
    )


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

Q2STRESS_CANDIDATES = [
    BASE_DIR / "Q2Stress" / "scripts" / "children" / "lexElem.txt",
    BASE_DIR / "Q2Stress" / "scripts" / "adults" / "phonItalia 1.10.1 - word forms.txt",
    BASE_DIR / "Progetto" / "Q2Stress" / "scripts" / "children" / "lexElem.txt",
]

PHONITALIAR_DATASET_CANDIDATES = [
    BASE_DIR / "phonItaliaR" / "data-raw" / "phonItalia-1.10" / "phonItalia-1.10-wordforms.tsv",
    BASE_DIR / "Progetto" / "phonItaliaR" / "data-raw" / "phonItalia-1.10" / "phonItalia-1.10-wordforms.tsv",
    BASE_DIR / "phonItalia-1.10-wordforms.tsv",
]

DEFAULT_WORD_LIST = BASE_DIR / "parole.txt"


STRESS_LABELS = {
    0: "ultima sillaba",
    1: "penultima sillaba",
    2: "antepenultima sillaba",
    3: "preantepenultima sillaba",
}


# ============================================================
# NORMALIZZAZIONE
# ============================================================

def norm_word(word: str) -> str:
    """
    Normalizza una parola per confrontarla con i dataset.

    Manteniamo gli accenti grafici italiani.
    """
    text = str(word).strip().lower()
    text = text.replace("'", "")
    text = text.replace("’", "")
    text = re.sub(r"[^a-zàèéìòóù]", "", text)
    return text


def normalize_text(text: str) -> str:
    """
    Rimuove la dipendenza dagli a capo.

    Qualunque sequenza di spazi, tab o newline viene trasformata
    in un singolo spazio.
    """
    text = text.replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# FEATURE ACCENTO
# ============================================================

def estimate_syllables(word: str) -> int:
    w = norm_word(word)

    if not w:
        return 1

    vowels = "aeiouàèéìòóù"
    count = 0
    previous_vowel = False

    for char in w:
        is_vowel = char in vowels

        if is_vowel and not previous_vowel:
            count += 1

        previous_vowel = is_vowel

    return max(1, count)


def build_word_features(
    word: str,
    row: Optional[pd.Series] = None
) -> dict:

    w = norm_word(word)
    letters = [c for c in w if c.isalpha()]
    vowels = "aeiouàèéìòóù"

    vowel_count = sum(char in vowels for char in letters)
    consonant_count = max(len(letters) - vowel_count, 0)

    syll_count = estimate_syllables(w)

    if row is not None:
        value = row.get("SumSylls")

        if pd.notna(value):
            try:
                syll_count = int(float(value))
            except (TypeError, ValueError):
                pass

    return {
        "word_len": len(letters),
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "vowel_ratio": vowel_count / max(len(letters), 1),
        "syll_count": max(1, syll_count),
        "ends_with_vowel": int(
            bool(letters) and letters[-1] in vowels
        ),
        "ends_with_consonant": int(
            bool(letters) and letters[-1] not in vowels
        ),
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
# Q2STRESS
# ============================================================

def find_q2stress_dataset(
    explicit_path: Optional[str] = None
) -> Path:

    if explicit_path:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset Q2Stress non trovato: {path}"
            )

        return path

    for path in Q2STRESS_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "\nNon trovo il dataset Q2Stress.\n"
        "Percorsi controllati:\n"
        + "\n".join(
            f"  - {p}"
            for p in Q2STRESS_CANDIDATES
        )
        + "\n\nUsa --q2stress per specificare manualmente il file."
    )


def load_q2stress_dataframe(
    explicit_path: Optional[str] = None
) -> pd.DataFrame:

    path = find_q2stress_dataset(
        explicit_path
    )

    print(f"\n[Q2Stress] Caricamento: {path}")

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ):

        try:
            return pd.read_csv(
                path,
                sep="\t",
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )

        except UnicodeDecodeError:
            continue

    raise UnicodeError(
        f"Impossibile leggere il file Q2Stress: {path}"
    )


def build_training_frame(
    df: pd.DataFrame
) -> pd.DataFrame:

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

        features = build_word_features(
            word,
            row
        )

        features["target"] = target
        rows.append(features)

    if not rows:
        raise ValueError(
            "\nNon sono state trovate righe valide nel dataset Q2Stress."
        )

    return pd.DataFrame(rows)


# ============================================================
# PHONITALIAR
# ============================================================

def find_phonitalia_dataset(
    explicit_path: Optional[str] = None
) -> Optional[Path]:

    if explicit_path:

        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset phonItaliaR TSV non trovato: {path}"
            )

        return path

    for path in PHONITALIAR_DATASET_CANDIDATES:

        if path.exists():
            return path

    return None


def load_phonitalia_dataset(
    explicit_path: Optional[str] = None
) -> Optional[pd.DataFrame]:

    path = find_phonitalia_dataset(
        explicit_path
    )

    if path is None:
        print(
            "\n[phonItaliaR] Nessun file TSV trovato."
        )
        return None

    print(
        f"\n[phonItaliaR] Caricamento: {path}"
    )

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ):

        try:

            df = pd.read_csv(
                path,
                sep="\t",
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )

            if len(df.columns) >= 2:
                return df

        except (
            UnicodeDecodeError,
            pd.errors.ParserError
        ):
            continue

    print(
        "[phonItaliaR] Attenzione: impossibile leggere il TSV."
    )

    return None


def detect_column(
    df: pd.DataFrame,
    possible_names: list[str]
) -> Optional[str]:

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    for col in df.columns:

        col_norm = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        for name in possible_names:

            name_norm = (
                name.strip()
                .lower()
                .replace(" ", "")
                .replace("_", "")
            )

            if col_norm == name_norm:
                return col

    return None


def phonitalia_stress_to_class(
    stressed_syllable,
    total_syllables
) -> Optional[int]:

    try:
        stressed = int(float(stressed_syllable))
        total = int(float(total_syllables))
    except (TypeError, ValueError):
        return None

    if stressed < 1 or total < 1:
        return None

    class_index = total - stressed

    return max(0, min(3, class_index))


def build_phonitalia_dict(
    df: Optional[pd.DataFrame]
) -> dict[str, int]:

    phon_dict: dict[str, int] = {}

    if df is None:
        return phon_dict

    word_col = detect_column(
        df,
        [
            "word",
            "parola",
            "wordform",
            "orthography",
        ]
    )

    syll_col = detect_column(
        df,
        [
            "sumsylls",
            "sum_sylls",
            "nsyllables",
            "numsyllables",
            "syllables",
        ]
    )

    stress_col = detect_column(
        df,
        [
            "stressedsyllable",
            "stressed_syllable",
            "stress",
            "stresspattern",
            "accento",
        ]
    )

    if not word_col or not syll_col or not stress_col:
        return phon_dict

    for _, row in df.iterrows():

        word = norm_word(
            row.get(word_col, "")
        )

        if not word:
            continue

        target = phonitalia_stress_to_class(
            row.get(stress_col),
            row.get(syll_col)
        )

        if target is None:
            continue

        if word not in phon_dict:
            phon_dict[word] = target

    return phon_dict


def enrich_with_phonitalia_dataset(
    frame: pd.DataFrame,
    df_phon: Optional[pd.DataFrame]
) -> pd.DataFrame:

    if df_phon is None:
        return frame

    word_col = detect_column(
        df_phon,
        ["word", "parola", "wordform", "orthography"]
    )

    syll_col = detect_column(
        df_phon,
        [
            "sumsylls",
            "sum_sylls",
            "nsyllables",
            "numsyllables",
            "syllables",
        ]
    )

    stress_col = detect_column(
        df_phon,
        [
            "stressedsyllable",
            "stressed_syllable",
            "stress",
            "stresspattern",
            "accento",
        ]
    )

    if (
        word_col is None
        or syll_col is None
        or stress_col is None
    ):
        return frame

    added = []
    seen = set()

    for _, row in df_phon.iterrows():

        word = norm_word(
            row.get(word_col, "")
        )

        if not word:
            continue

        target = phonitalia_stress_to_class(
            row.get(stress_col),
            row.get(syll_col)
        )

        if target is None:
            continue

        key = (word, target)

        if key in seen:
            continue

        seen.add(key)

        features = build_word_features(word)
        features["target"] = target

        added.append(features)

    if not added:
        return frame

    extra = pd.DataFrame(added)

    combined = pd.concat(
        [frame, extra],
        ignore_index=True
    )

    combined = combined.drop_duplicates()

    print(
        "[phonItaliaR dataset] "
        "Utilizzato per arricchire il dataset."
    )

    print(
        f"[Training] Righe complessive: {len(combined)}"
    )

    return combined


# ============================================================
# TRAINING
# ============================================================

def train_model(
    frame: pd.DataFrame
) -> Pipeline:

    X = frame.drop(
        columns=["target"]
    ).copy()

    y = frame["target"]

    for col in [
        "last1",
        "last2",
        "last3",
        "last4",
    ]:

        if col in X.columns:
            X[col] = X[col].astype(str)

    numeric_cols = [
        col
        for col in X.columns
        if pd.api.types.is_numeric_dtype(
            X[col]
        )
    ]

    categorical_cols = [
        col
        for col in X.columns
        if col not in numeric_cols
    ]

    transformers = []

    if numeric_cols:

        transformers.append(
            (
                "num",
                StandardScaler(),
                numeric_cols,
            )
        )

    if categorical_cols:

        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            ),
        ]
    )

    class_counts = y.value_counts()

    if (
        len(class_counts) >= 2
        and class_counts.min() >= 2
    ):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        print()
        print("=" * 60)
        print("VALUTAZIONE RANDOM FOREST")
        print("=" * 60)

        print(
            f"\nAccuracy: "
            f"{accuracy_score(y_test, predictions):.4f}"
        )

    else:

        model.fit(X, y)

    return model


# ============================================================
# PREDIZIONE
# ============================================================

def predict_stress(
    word: str,
    model: Pipeline
) -> dict:

    features = pd.DataFrame(
        [build_word_features(word)]
    )

    prediction = int(
        model.predict(features)[0]
    )

    probabilities = {}

    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(
            features
        )[0]

        classes = (
            model
            .named_steps["classifier"]
            .classes_
        )

        probabilities = {
            int(cls): float(prob)
            for cls, prob in zip(
                classes,
                proba
            )
        }

    confidence = probabilities.get(
        prediction,
        0.0
    )

    return {
        "model_position": prediction,
        "model_label": STRESS_LABELS.get(
            prediction,
            "sconosciuta"
        ),
        "confidence": confidence,
    }


# ============================================================
# ANALISI DI UNA PAROLA
# ============================================================

def analyze_word(
    word: str,
    model: Pipeline,
    phon_dict: dict
) -> dict:

    model_result = predict_stress(
        word,
        model
    )

    w_norm = norm_word(word)

    phon_position = phon_dict.get(
        w_norm
    )

    phon_label = (
        STRESS_LABELS.get(
            phon_position,
            "non disponibile"
        )
        if phon_position is not None
        else "non disponibile"
    )

    agreement = (
        model_result["model_position"]
        == phon_position
        if phon_position is not None
        else None
    )

    return {
        "word": word,
        **model_result,
        "phon_position": phon_position,
        "phon_label": phon_label,
        "agreement": agreement,
    }


# ============================================================
# LETTURA DEL FILE TXT
# ============================================================

def load_text_file(
    path: Path
) -> str:

    if not path.exists():
        raise FileNotFoundError(
            f"File di testo non trovato: {path}"
        )

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ):

        try:

            with open(
                path,
                "r",
                encoding=encoding
            ) as file:

                text = file.read()

            # IMPORTANTE:
            # gli a capo NON definiscono le parole.
            # Vengono trasformati in spazi.
            return normalize_text(text)

        except UnicodeDecodeError:
            continue

    raise UnicodeError(
        f"Impossibile leggere il file: {path}"
    )


# ============================================================
# UTILITÀ CONTEXTO
# ============================================================

def punctuation_after_token(
    doc,
    token_index
) -> str:

    """
    Restituisce la punteggiatura immediatamente successiva
    al token, se presente.
    """

    for next_token in doc[token_index + 1:]:
        if next_token.is_space:
            continue

        if next_token.is_punct:
            return next_token.text

        break

    return ""


def punctuation_before_token(
    doc,
    token_index
) -> str:

    for prev_token in reversed(
        doc[:token_index]
    ):

        if prev_token.is_space:
            continue

        if prev_token.is_punct:
            return prev_token.text

        break

    return ""


def sentence_position(
    token,
    sentence
) -> str:

    content_tokens = [
        t
        for t in sentence
        if not t.is_space and not t.is_punct
    ]

    if not content_tokens:
        return "unknown"

    if token == content_tokens[0]:
        return "inizio"

    if token == content_tokens[-1]:
        return "fine"

    return "interno"


def syntactic_role(token) -> str:

    mapping = {
        "ROOT": "radice",
        "nsubj": "soggetto",
        "nsubj:pass": "soggetto",
        "obj": "oggetto",
        "iobj": "oggetto_indiretto",
        "amod": "modificatore_aggettivale",
        "advmod": "modificatore_avverbiale",
        "nmod": "complemento_nominale",
        "obl": "complemento",
        "conj": "coordinazione",
        "cc": "congiunzione",
        "det": "determinante",
        "aux": "ausiliare",
        "cop": "copula",
        "case": "preposizione",
        "mark": "marcatore",
        "compound": "componente",
    }

    return mapping.get(
        token.dep_,
        token.dep_
    )


def syntactic_depth(token) -> int:

    """
    Distanza del token dalla radice sintattica.
    Utile per rappresentare la gerarchia musicale.
    """

    depth = 0
    current = token

    while current.dep_ != "ROOT":
        parent = current.head

        if parent == current:
            break

        current = parent
        depth += 1

        if depth > 100:
            break

    return depth


# ============================================================
# ANALISI SEMANTICA
# ============================================================

"""
Questo modulo giudica non solo la forma della parola (accento,
sillabe, sintassi) ma anche il suo SIGNIFICATO, in modo da poter
influenzare le caratteristiche musicali (peso ritmico, tensione
melodica) anche in base al contenuto semantico ed emotivo del testo.

Vengono considerati tre aspetti del significato:

1. Categoria semantica  -> tipo di entità nominata (persona, luogo,
   organizzazione, ecc.) individuata da spaCy tramite NER.
2. Polarità / intensità -> valenza emotiva della parola (positiva,
   negativa o neutra) stimata da un piccolo lessico italiano basato
   sul lemma, più eventuali intensificatori/negazioni nel contesto.
3. Similarità semantica -> quanto la parola è "in tema" rispetto al
   resto della frase, calcolata con i vettori distribuzionali di
   spaCy (se disponibili).
"""

# Lessico di polarità semantica (lemma -> valore in [-1, 1]).
# Non è esaustivo: le parole assenti vengono considerate neutre (0.0)
# a meno che spaCy non le riconosca come entità nominate.
SENTIMENT_LEXICON: dict[str, float] = {
    # positivo
    "felice": 1.0, "felicità": 1.0, "gioia": 1.0, "gioioso": 0.9,
    "amore": 1.0, "amare": 0.9, "sereno": 0.7, "serenità": 0.7,
    "bello": 0.7, "bellezza": 0.7, "buono": 0.6, "bontà": 0.6,
    "splendido": 0.9, "meraviglioso": 0.9, "meraviglia": 0.8,
    "dolce": 0.5, "dolcezza": 0.5, "pace": 0.7, "pacifico": 0.6,
    "vittoria": 0.8, "successo": 0.7, "speranza": 0.7, "sperare": 0.6,
    "sorridere": 0.8, "sorriso": 0.8, "ridere": 0.7, "divertente": 0.7,
    "allegro": 0.8, "allegria": 0.8, "entusiasmo": 0.8, "entusiasta": 0.8,
    "orgoglio": 0.6, "orgoglioso": 0.6, "grazie": 0.6, "grato": 0.6,
    "libertà": 0.6, "libero": 0.5, "calmo": 0.5, "calma": 0.5,
    "luce": 0.4, "luminoso": 0.4, "caldo": 0.3, "abbraccio": 0.6,
    # negativo
    "triste": -1.0, "tristezza": -1.0, "dolore": -1.0, "sofferenza": -0.9,
    "morte": -0.9, "morire": -0.8, "paura": -0.8, "spaventoso": -0.8,
    "odio": -1.0, "odiare": -0.9, "rabbia": -0.9, "arrabbiato": -0.8,
    "male": -0.6, "malvagio": -0.8, "brutto": -0.6, "bruttezza": -0.6,
    "guerra": -0.8, "violenza": -0.8, "violento": -0.8, "pianto": -0.7,
    "piangere": -0.7, "solitudine": -0.7, "solo": -0.4, "abbandonato": -0.7,
    "buio": -0.4, "oscurità": -0.5, "freddo": -0.3, "gelido": -0.5,
    "ansia": -0.7, "angoscia": -0.8, "disperazione": -0.9, "disperato": -0.9,
    "nemico": -0.6, "crudele": -0.8, "cattivo": -0.6, "tradimento": -0.8,
    "tradire": -0.7, "perdere": -0.5, "perdita": -0.6, "malato": -0.6,
    "malattia": -0.6, "stanco": -0.4, "stanchezza": -0.4, "fatica": -0.3,
}

# Parole che invertono o attenuano/rafforzano la polarità della
# parola che le segue nella stessa frase.
NEGATION_LEMMAS = {"non", "nessuno", "niente", "mai", "senza"}
INTENSIFIER_LEMMAS = {"molto": 1.4, "tanto": 1.3, "troppo": 1.4, "davvero": 1.3, "estremamente": 1.6}
DIMINISHER_LEMMAS = {"poco": 0.6, "leggermente": 0.6, "quasi": 0.7}

ENTITY_CATEGORY_LABELS = {
    "PER": "persona",
    "LOC": "luogo",
    "GPE": "luogo",
    "ORG": "organizzazione",
    "MISC": "altro",
}


def entity_category(token) -> str:
    """Categoria semantica basata sul riconoscimento di entità (NER)."""

    if not token.ent_type_:
        return "nessuna"

    return ENTITY_CATEGORY_LABELS.get(
        token.ent_type_,
        token.ent_type_.lower()
    )


def semantic_polarity(token, sentence_tokens) -> tuple[float, str]:
    """
    Stima la polarità semantica/emotiva della parola.

    Il valore base viene cercato nel lessico tramite il lemma.
    Se nella stessa frase, appena prima del token, compare una
    negazione, la polarità viene invertita; se compare un
    intensificatore/diminutivo, l'intensità viene scalata.
    """

    lemma = token.lemma_.lower().strip()
    base = SENTIMENT_LEXICON.get(lemma, 0.0)

    if base == 0.0:
        # Nessuna informazione lessicale: consideriamo la parola
        # neutra rispetto al significato emotivo.
        return 0.0, "neutro"

    value = base

    tokens_in_sentence = list(sentence_tokens)

    if token in tokens_in_sentence:
        position = tokens_in_sentence.index(token)

        # Guarda fino a due parole precedenti per negazioni/intensificatori.
        for prev in reversed(tokens_in_sentence[max(0, position - 2):position]):
            prev_lemma = prev.lemma_.lower().strip()

            if prev_lemma in NEGATION_LEMMAS:
                value *= -1
            elif prev_lemma in INTENSIFIER_LEMMAS:
                value *= INTENSIFIER_LEMMAS[prev_lemma]
            elif prev_lemma in DIMINISHER_LEMMAS:
                value *= DIMINISHER_LEMMAS[prev_lemma]

    value = max(-1.0, min(1.0, value))

    if value > 0.15:
        label = "positivo"
    elif value < -0.15:
        label = "negativo"
    else:
        label = "neutro"

    return value, label


def semantic_similarity_to_sentence(token, sentence) -> float:
    """
    Similarità semantica tra la parola e il resto della frase,
    calcolata con i vettori distribuzionali (se il modello caricato
    li include). Restituisce 0.0 se i vettori non sono disponibili.
    """

    if not HAS_SEMANTIC_VECTORS or not token.has_vector:
        return 0.0

    if sentence.vector_norm == 0.0 or token.vector_norm == 0.0:
        return 0.0

    try:
        return float(token.similarity(sentence))
    except Exception:
        return 0.0


def semantic_salience(token, polarity_value: float) -> float:
    """
    Quanto la parola è "importante" dal punto di vista del
    significato: le parole di contenuto (nomi, verbi, aggettivi,
    avverbi) ed entità nominate pesano di più delle parole
    puramente grammaticali (articoli, preposizioni, congiunzioni).
    """

    content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

    weight = 1.0 if token.pos_ in content_pos else 0.4

    if token.ent_type_:
        weight += 0.3

    weight += abs(polarity_value) * 0.5

    return round(weight, 3)


# ============================================================
# ANALISI DEL TESTO CON SPACY
# ============================================================

def analyze_text(
    text: str,
    model: Pipeline,
    phon_dict: dict
) -> list[dict]:

    doc = nlp(text)

    # Usiamo esclusivamente i token che rappresentano parole.
    word_tokens = [
        token
        for token in doc
        if not token.is_space
        and not token.is_punct
    ]

    # Mappa token -> indice nella sequenza delle parole.
    word_index_map = {
        token.i: index
        for index, token in enumerate(word_tokens)
    }

    results = []

    for index, token in enumerate(
        word_tokens
    ):

        result = analyze_word(
            token.text,
            model,
            phon_dict
        )

        # ----------------------------------------------------
        # Individuazione della frase
        # ----------------------------------------------------

        sentence = token.sent

        sentence_tokens = [
            t
            for t in sentence
            if not t.is_space
            and not t.is_punct
        ]

        sentence_id = (
            sentence.start
        )

        # ----------------------------------------------------
        # Punteggiatura
        # ----------------------------------------------------

        punct_before = (
            punctuation_before_token(
                doc,
                token.i
            )
        )

        punct_after = (
            punctuation_after_token(
                doc,
                token.i
            )
        )

        # ----------------------------------------------------
        # Contesto precedente/successivo
        # ----------------------------------------------------

        previous_word = (
            word_tokens[index - 1].text
            if index > 0
            else ""
        )

        next_word = (
            word_tokens[index + 1].text
            if index < len(word_tokens) - 1
            else ""
        )

        # ----------------------------------------------------
        # Contesto nella stessa frase
        # ----------------------------------------------------

        sentence_word_index = (
            word_index_map.get(
                token.i,
                0
            )
            - word_index_map.get(
                sentence_tokens[0].i,
                0
            )
            if sentence_tokens
            else 0
        )

        sentence_word_count = len(
            sentence_tokens
        )

        # ----------------------------------------------------
        # Significato / semantica
        # ----------------------------------------------------

        polarity_value, polarity_label = semantic_polarity(
            token,
            sentence_tokens
        )

        similarity_value = semantic_similarity_to_sentence(
            token,
            sentence
        )

        salience_value = semantic_salience(
            token,
            polarity_value
        )

        # ----------------------------------------------------
        # Informazioni linguistiche
        # ----------------------------------------------------

        result.update({

       # Identificazione
       "sentence_id": sentence_id,
       "sentence": sentence.text.strip(),
       "token_index": token.i,
       "word_index": index,

    # Testo
        "token_text": token.text,
        "normalized_word": norm_word(token.text),
        "normalized_word": norm_word(
        token.text
            ),

            # Lessico
            "lemma": token.lemma_,
            "word_length": len(
                norm_word(token.text)
            ),
            "syllable_count": estimate_syllables(
                token.text
            ),

            # Morfologia / POS
            "pos": token.pos_,
            "tag": token.tag_,
            "morphology": str(
                token.morph
            ),

            # Sintassi
            "dependency": token.dep_,
            "syntactic_role": syntactic_role(
                token
            ),
            "head": token.head.text,
            "head_lemma": token.head.lemma_,
            "head_token_index": token.head.i,
            "syntactic_depth": syntactic_depth(
                token
            ),

            # Proprietà sintattiche
            "is_root": token.dep_ == "ROOT",
            "is_subject": token.dep_ in {
                "nsubj",
                "nsubj:pass",
            },
            "is_object": token.dep_ in {
                "obj",
                "iobj",
            },
            "is_modifier": token.dep_ in {
                "amod",
                "advmod",
                "nmod",
                "obl",
            },

            # Posizione nella frase
            "sentence_word_index": sentence_word_index,
            "sentence_word_count": sentence_word_count,
            "sentence_position": sentence_position(
                token,
                sentence
            ),

            "is_first_in_sentence": (
                sentence_word_index == 0
            ),

            "is_last_in_sentence": (
                sentence_word_index
                == sentence_word_count - 1
            ),

            # Contesto locale
            "previous_word": previous_word,
            "next_word": next_word,

            # Punteggiatura
            "punctuation_before": punct_before,
            "punctuation_after": punct_after,

            "has_pause_after": (
                punct_after in {
                    ",",
                    ";",
                    ":",
                }
            ),

            "has_sentence_end": (
                punct_after in {
                    ".",
                    "!",
                    "?",
                }
            ),

            "is_question_end": (
                punct_after == "?"
            ),

            "is_exclamation_end": (
                punct_after == "!"
            ),

            # Semantica / significato
            "semantic_category": entity_category(token),
            "is_named_entity": bool(token.ent_type_),
            "semantic_polarity": round(polarity_value, 3),
            "semantic_polarity_label": polarity_label,
            "semantic_similarity_to_sentence": round(similarity_value, 3),
            "semantic_salience": salience_value,

            # Caratteristiche utili al motore musicale,
            # ora derivate anche dal significato della parola:
            # le parole semanticamente più importanti (salienza)
            # pesano di più a livello ritmico, mentre la carica
            # emotiva (polarità) genera tensione melodica.
            "rhythmic_weight": salience_value,
            "melodic_tension": round(abs(polarity_value), 3),
        })

        results.append(result)

    return results


# ============================================================
# STAMPA RISULTATI
# ============================================================

def print_results(
    results: list[dict]
) -> None:

    print()
    print("=" * 120)
    print("ANALISI TESTO")
    print("=" * 120)

    headers = [
        "word",
        "model_label",
        "phon_label",
        "confidence",
        "agreement",
        "lemma",
        "pos",
        "dependency",
        "syntactic_role",
        "sentence_position",
        "semantic_category",
        "semantic_polarity_label",
    ]

    print(
        " | ".join(
            f"{h:<22}"
            for h in headers
        )
    )

    print("-" * 120)

    for result in results:

        values = [
            result.get(
                "word",
                ""
            ),
            result.get(
                "model_label",
                ""
            ),
            result.get(
                "phon_label",
                ""
            ),
            f"{result.get('confidence', 0) * 100:.1f}%",
            result.get(
                "agreement",
                ""
            ),
            result.get(
                "lemma",
                ""
            ),
            result.get(
                "pos",
                ""
            ),
            result.get(
                "dependency",
                ""
            ),
            result.get(
                "syntactic_role",
                ""
            ),
            result.get(
                "sentence_position",
                ""
            ),
            result.get(
                "semantic_category",
                ""
            ),
            result.get(
                "semantic_polarity_label",
                ""
            ),
        ]

        print(
            " | ".join(
                f"{str(v):<22}"
                for v in values
            )
        )

    print("=" * 120)


def print_statistics(
    results: list[dict]
) -> None:

    comparisons = [
        r
        for r in results
        if r["agreement"] is not None
    ]

    print()

    print("=" * 60)
    print("STATISTICHE")
    print("=" * 60)

    print(
        f"Parole analizzate: {len(results)}"
    )

    print(
        f"Frasi individuate: "
        f"{len(set(r['sentence_id'] for r in results))}"
    )

    print(
        f"Confronti con phonItaliaR: "
        f"{len(comparisons)}"
    )

    if comparisons:

        correct = sum(
            bool(r["agreement"])
            for r in comparisons
        )

        print(
            f"Accordi: {correct}"
        )

        print(
            f"Accuratezza confronto: "
            f"{correct / len(comparisons) * 100:.2f}%"
        )

    print("=" * 60)
    print("SEMANTICA")
    print("=" * 60)

    entities = [
        r for r in results if r.get("is_named_entity")
    ]

    polarities = [
        r["semantic_polarity"]
        for r in results
        if r.get("semantic_polarity_label") != "neutro"
    ]

    print(
        f"Entità nominate individuate: {len(entities)}"
    )

    print(
        f"Parole con carica emotiva riconosciuta: {len(polarities)}"
    )

    if polarities:
        avg_polarity = sum(polarities) / len(polarities)
        print(
            f"Polarità media (parole non neutre): {avg_polarity:+.2f}"
        )

    if not HAS_SEMANTIC_VECTORS:
        print(
            "Similarità semantica: non disponibile "
            "(modello senza vettori, usa it_core_news_md o superiore)"
        )

    print("=" * 60)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Classificatore accenti + "
            "analisi contestuale spaCy."
        )
    )

    parser.add_argument(
        "text_file",
        help=(
            "File .txt contenente il testo. "
            "Non è necessario andare a capo dopo ogni parola."
        )
    )

    parser.add_argument(
        "--q2stress",
        default=None,
        help="Percorso dataset Q2Stress."
    )

    parser.add_argument(
        "--phonitalia-dataset",
        default=None,
        help="Percorso TSV phonItaliaR."
    )

    parser.add_argument(
        "--output",
        default="risultati_accenti_contesto.csv",
        help="CSV di output."
    )

    parser.add_argument(
        "--no-phonitalia",
        action="store_true",
        help="Disattiva confronto con phonItaliaR."
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 80)
    print("ACCENTI + ANALISI DEL CONTESTO LINGUISTICO")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Q2Stress
    # --------------------------------------------------------

    df_q2stress = load_q2stress_dataframe(
        args.q2stress
    )

    frame = build_training_frame(
        df_q2stress
    )

    # --------------------------------------------------------
    # 2. phonItaliaR
    # --------------------------------------------------------

    phon_dict = {}

    if not args.no_phonitalia:

        df_phon = load_phonitalia_dataset(
            args.phonitalia_dataset
        )

        if df_phon is not None:

            frame = enrich_with_phonitalia_dataset(
                frame,
                df_phon
            )

            phon_dict = build_phonitalia_dict(
                df_phon
            )

            print(
                f"[phonItaliaR] "
                f"Caricate {len(phon_dict)} parole."
            )

    # --------------------------------------------------------
    # 3. Training
    # --------------------------------------------------------

    print(
        "\n[MODELLO] Addestramento Random Forest..."
    )

    model = train_model(
        frame
    )

    print(
        "[MODELLO] Addestramento completato."
    )

    # --------------------------------------------------------
    # 4. Lettura testo
    # --------------------------------------------------------

    text_path = Path(
        args.text_file
    )

    text = load_text_file(
        text_path
    )

    if not text:

        raise ValueError(
            "Il file di testo è vuoto."
        )

    print()
    print(
        "[INPUT] Testo caricato:"
    )

    print(
        text[:500]
        + (
            "..."
            if len(text) > 500
            else ""
        )
    )

    # --------------------------------------------------------
    # 5. Analisi spaCy + accenti
    # --------------------------------------------------------

    print(
        "\n[SPACY] Analisi linguistica..."
    )

    results = analyze_text(
        text,
        model,
        phon_dict
    )

    if not results:

        raise ValueError(
            "Non sono state trovate parole nel testo."
        )

    # --------------------------------------------------------
    # 6. Output console
    # --------------------------------------------------------

    print_results(
        results
    )

    print_statistics(
        results
    )

    # --------------------------------------------------------
    # 7. Salvataggio CSV
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    pd.DataFrame(
        results
    ).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"[OUTPUT] CSV salvato in: {output_path}"
    )

    print(
        "\nAnalisi completata."
    )


if __name__ == "__main__":
    main()