#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Versione 5: aggiunta la possibilità di produrre il MIDI con la batteria!!!

Per lanciarlo: python3 Fase1.py frase.txt --midi-output ritmo5.mid --rhythm-output ritmo5.csv --drums

(se non specificato prende di default 100bpm)

[
va bene anche: python3 Fase1.py frase.txt --midi-output ritmo5.mid --rhythm-output ritmo5.csv
per lanciarlo (forse): python3 Fase1.py frase.txt --midi-output ritmo5.mid --bpm 100]


Accenti.py — versione unificata
================================

Classificatore dell'accento tonico italiano + analisi prosodica/linguistica
con spaCy, pensato per alimentare la Fase 1 della pipeline di Prosody2Music
("scheletro ritmico deterministico" a partire dalla prosodia del testo).

Cosa unisce rispetto alle due versioni precedenti
---------------------------------------------------
- Dalla versione "testo libero + spaCy": l'analisi su un intero testo (non
  solo su una lista di parole), i confini di frase, la punteggiatura (che
  qui diventa PAUSA nello scheletro ritmico), il ruolo sintattico e la
  profondità dalla radice.
- Dalla versione "lista di parole + phonItaliaR": l'idea di usare
  direttamente il TSV di phonItaliaR come dizionario di validazione O(1),
  ma qui la funzione è stata RISCRITTA perché nella versione originale
  build_phonitalia_dict() aveva un bug (variabile "w" mai definita) e
  duplicava la logica già presente in phonitalia_stress_to_class() invece
  di riusarla.
- NOVITÀ: conteggio sillabe corretto. Prima veniva semplicemente contato
  ogni "gruppo di vocali consecutive" come una sillaba: questo sottostima
  sistematicamente le parole con iato (es. "pa-e-se" veniva contata 2
  invece di 3, perché "ae" veniva trattato come un unico nucleo). Ora:
    1) se è installato pyphen (sillabazione basata su regole/dizionario,
       `pip install pyphen`), viene usato quello come fonte primaria;
    2) altrimenti si usa un fallback euristico che distingue dittonghi
       (vocale forte + debole non accentata, es. "ia", "uo") da iati
       (due vocali forti, es. "ae", "oe" → sillabe separate);
    3) quando è disponibile il conteggio reale da Q2Stress/phonItaliaR
       (colonna SumSylls) quello ha sempre la precedenza, perché è dato
       annotato e non stimato.
- NOVITÀ: build_rhythm_skeleton() — dato l'elenco dei risultati per
  parola, genera direttamente lo scheletro ritmico richiesto dal
  progetto: una sequenza di impulsi (sillaba forte / sillaba debole /
  pausa) pronta per essere passata alla Fase 2/3 (music21).
- L'analisi semantica (NER, polarità, similarità) della prima versione è
  stata mantenuta ma resa OPZIONALE (--semantics), spenta di default: è
  utile in una fase successiva per pesare ritmo/melodia col significato,
  ma non serve per costruire lo scheletro ritmico di base e rallenta
  l'analisi senza un modello con vettori (it_core_news_md+). Vedi nota
  in fondo al file.

INPUT
-----
Un file .txt di testo italiano in linguaggio naturale (niente a capo
manuali necessari).

OUTPUT
------
Un CSV con una riga per parola (predizione accento, confidence,
confronto con phonItaliaR, feature linguistiche/sintattiche/prosodiche)
e, opzionalmente, un secondo CSV con lo scheletro ritmico a livello di
sillaba.

REQUISITI
---------
    pip install pandas scikit-learn spacy pyphen
    python -m spacy download it_core_news_sm
    # consigliato per la similarità semantica (opzionale):
    python -m spacy download it_core_news_md

ESECUZIONE
----------
    python Accenti.py testo.txt
    python Accenti.py testo.txt --output risultati.csv --semantics
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
    from sklearn.metrics import accuracy_score, classification_report
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


# ============================================================
# IMPORT PYPHEN (opzionale, per il conteggio sillabe)
# ============================================================

try:
    import pyphen
    _PYPHEN_DIC = pyphen.Pyphen(lang="it_IT")
except ImportError:
    _PYPHEN_DIC = None


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

HAS_SEMANTIC_VECTORS = nlp.meta.get("vectors", {}).get("vectors", 0) > 0


# ============================================================
# CONFIGURAZIONE PERCORSI DATASET
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

STRESS_LABELS = {
    0: "ultima sillaba",
    1: "penultima sillaba",
    2: "antepenultima sillaba",
    3: "preantepenultima sillaba",
}


# ============================================================
# NORMALIZZAZIONE TESTO
# ============================================================

def norm_word(word: str) -> str:
    """Normalizza una parola per confrontarla con i dataset (accenti mantenuti)."""
    text = str(word).strip().lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-zàèéìòóù]", "", text)
    return text


def normalize_text(text: str) -> str:
    """Rimuove la dipendenza dagli a capo: qualunque sequenza di whitespace -> un solo spazio."""
    text = text.replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# CONTEGGIO SILLABE (corretto)
# ============================================================

_STRONG_VOWELS = set("aeoàèéòó")
_WEAK_VOWELS = set("iuìù")
_VOWELS = _STRONG_VOWELS | _WEAK_VOWELS


def _syllables_heuristic(word: str) -> int:
    """
    Fallback euristico usato quando pyphen non è disponibile.

    Regola: due vocali adiacenti restano nella stessa sillaba (dittongo)
    se almeno una delle due è una vocale "debole" non accentata (i/u);
    restano invece in sillabe separate (iato) se sono entrambe vocali
    "forti" (a, e, o) — es. "pa-e-se", "po-e-ta". Questo è più accurato
    del semplice conteggio di "gruppi di vocali consecutive", che
    sottostimava sistematicamente le parole con iato.

    Non è perfetto (l'italiano ha eccezioni che richiederebbero di sapere
    dove cade l'accento tonico, es. "farmacia"), ma è un'approssimazione
    ragionevole e va usato solo quando manca un dato reale (SumSylls) o
    pyphen.
    """
    w = norm_word(word)
    if not w:
        return 1

    syll = 0
    n = len(w)
    idx = 0

    while idx < n:
        c = w[idx]
        if c not in _VOWELS:
            idx += 1
            continue

        # inizio di un nuovo nucleo vocalico
        syll += 1
        j = idx

        while j + 1 < n and w[j + 1] in _VOWELS:
            cur, nxt = w[j], w[j + 1]
            is_diphthong = (cur in _WEAK_VOWELS) or (nxt in _WEAK_VOWELS)

            if is_diphthong:
                # dittongo/trittongo: stessa sillaba, continua a consumare
                j += 1
                continue
            else:
                # due vocali forti = iato = nuova sillaba
                break

        idx = j + 1

    return max(1, syll)


def estimate_syllables(word: str) -> int:
    """
    Conteggio sillabe stimato per una singola parola.

    Ordine di preferenza:
        1. pyphen (se installato) — sillabazione basata su regole/dizionario.
        2. euristica dittongo/iato (_syllables_heuristic).

    NB: quando si costruisce il training set da Q2Stress/phonItaliaR, il
    valore reale annotato (colonna SumSylls) ha sempre la precedenza su
    questa stima — vedi build_word_features().
    """
    w = norm_word(word)
    if not w:
        return 1

    if _PYPHEN_DIC is not None:
        hyphenated = _PYPHEN_DIC.inserted(w)
        count = hyphenated.count("-") + 1
        if count > 0:
            return count

    return _syllables_heuristic(w)


# ============================================================
# FEATURE PER IL MODELLO DI ACCENTO
# ============================================================

def build_word_features(word: str, row: Optional[pd.Series] = None) -> dict:
    w = norm_word(word)
    letters = [c for c in w if c.isalpha()]

    vowel_count = sum(char in _VOWELS for char in letters)
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
        "ends_with_vowel": int(bool(letters) and letters[-1] in _VOWELS),
        "ends_with_consonant": int(bool(letters) and letters[-1] not in _VOWELS),
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
# CARICAMENTO Q2STRESS
# ============================================================

def find_q2stress_dataset(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset Q2Stress non trovato: {path}")
        return path

    for path in Q2STRESS_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "\nNon trovo il dataset Q2Stress.\nPercorsi controllati:\n"
        + "\n".join(f"  - {p}" for p in Q2STRESS_CANDIDATES)
        + "\n\nUsa --q2stress per specificare manualmente il file."
    )


def load_q2stress_dataframe(explicit_path: Optional[str] = None) -> pd.DataFrame:
    path = find_q2stress_dataset(explicit_path)
    print(f"\n[Q2Stress] Caricamento: {path}")

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                path, sep="\t", engine="python", encoding=encoding, on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Impossibile leggere il file Q2Stress: {path}")


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
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

        features = build_word_features(word, row)
        features["target"] = target
        rows.append(features)

    if not rows:
        raise ValueError("\nNon sono state trovate righe valide nel dataset Q2Stress.")

    return pd.DataFrame(rows)


# ============================================================
# CARICAMENTO PHONITALIAR (TSV diretto)
# ============================================================

def find_phonitalia_dataset(explicit_path: Optional[str] = None) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset phonItaliaR TSV non trovato: {path}")
        return path

    for path in PHONITALIAR_DATASET_CANDIDATES:
        if path.exists():
            return path

    return None


def load_phonitalia_dataset(explicit_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = find_phonitalia_dataset(explicit_path)
    if path is None:
        print("\n[phonItaliaR] Nessun file TSV trovato.")
        return None

    print(f"\n[phonItaliaR] Caricamento: {path}")

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                path, sep="\t", engine="python", encoding=encoding, on_bad_lines="skip"
            )
            if len(df.columns) >= 2:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    print("[phonItaliaR] Attenzione: impossibile leggere il TSV.")
    return None


def detect_column(df: pd.DataFrame, possible_names: list[str]) -> Optional[str]:
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        col_norm = str(col).strip().lower().replace(" ", "").replace("_", "")
        for name in possible_names:
            name_norm = name.strip().lower().replace(" ", "").replace("_", "")
            if col_norm == name_norm:
                return col

    return None


def phonitalia_stress_to_class(stressed_syllable, total_syllables) -> Optional[int]:
    """
    Converte il campo `StressedSyllable` di phonItaliaR (1-based, contato
    da sinistra) nel formato interno del modello (0 = ultima sillaba,
    1 = penultima, ... contato dalla fine della parola).
    """
    try:
        stressed = int(float(stressed_syllable))
        total = int(float(total_syllables))
    except (TypeError, ValueError):
        return None

    if stressed < 1 or total < 1:
        return None

    class_index = total - stressed
    return max(0, min(3, class_index))


def build_phonitalia_dict(df: Optional[pd.DataFrame]) -> dict[str, int]:
    """
    Dizionario O(1) parola -> classe di accento, usato come ground truth
    per validare le predizioni del modello.

    NB: riusa phonitalia_stress_to_class() invece di riscrivere la
    conversione a mano (nella bozza precedente questa funzione aveva un
    bug: la variabile `w` non era mai definita, causava un NameError non
    appena veniva incontrata la prima riga valida).
    """
    phon_dict: dict[str, int] = {}
    if df is None:
        return phon_dict

    word_col = detect_column(df, ["word", "parola", "wordform", "orthography"])
    syll_col = detect_column(
        df, ["sumsylls", "sum_sylls", "nsyllables", "numsyllables", "syllables"]
    )
    stress_col = detect_column(
        df, ["stressedsyllable", "stressed_syllable", "stress", "stresspattern", "accento"]
    )

    if not word_col or not syll_col or not stress_col:
        return phon_dict

    for _, row in df.iterrows():
        word = norm_word(row.get(word_col, ""))
        if not word:
            continue

        target = phonitalia_stress_to_class(row.get(stress_col), row.get(syll_col))
        if target is None:
            continue

        # In caso di duplicati per la stessa forma, teniamo la prima
        # occorrenza valida (in phonItaliaR i duplicati sono quasi sempre
        # ripetizioni dello stesso lemma con la stessa posizione d'accento).
        phon_dict.setdefault(word, target)

    return phon_dict


def enrich_with_phonitalia_dataset(
    frame: pd.DataFrame, df_phon: Optional[pd.DataFrame]
) -> pd.DataFrame:
    if df_phon is None:
        return frame

    word_col = detect_column(df_phon, ["word", "parola", "wordform", "orthography"])
    syll_col = detect_column(
        df_phon, ["sumsylls", "sum_sylls", "nsyllables", "numsyllables", "syllables"]
    )
    stress_col = detect_column(
        df_phon, ["stressedsyllable", "stressed_syllable", "stress", "stresspattern", "accento"]
    )

    if word_col is None or syll_col is None or stress_col is None:
        return frame

    added = []
    seen = set()

    for _, row in df_phon.iterrows():
        word = norm_word(row.get(word_col, ""))
        if not word:
            continue

        target = phonitalia_stress_to_class(row.get(stress_col), row.get(syll_col))
        if target is None:
            continue

        key = (word, target)
        if key in seen:
            continue
        seen.add(key)

        features = build_word_features(word, row)
        features["target"] = target
        added.append(features)

    if not added:
        return frame

    extra = pd.DataFrame(added)
    combined = pd.concat([frame, extra], ignore_index=True).drop_duplicates()

    print("[phonItaliaR] Dataset usato per arricchire il training set.")
    print(f"[Training] Righe complessive: {len(combined)}")

    return combined


# ============================================================
# TRAINING DEL MODELLO
# ============================================================

def train_model(frame: pd.DataFrame, balance_strength: float = 0.5) -> Pipeline:
    """
    balance_strength controlla quanto pesare le classi minoritarie:
        0.0 -> nessun bilanciamento (class_weight tutti = 1, il modello
               tenderà a favorire la classe maggioritaria, "penultima")
        1.0 -> bilanciamento pieno, equivalente a class_weight="balanced"
               di sklearn (n_samples / (n_classes * conteggio_classe))
        0.5 (default) -> via di mezzo: i pesi vengono attenuati con una
               radice quadrata, così le classi rare (es. "preantepenultima",
               48 esempi su 44557) ricevono comunque più peso della
               maggioranza, ma senza la sovra-correzione osservata con
               class_weight="balanced" pieno (che portava il modello a
               predire troppo spesso le classi minoritarie anche quando
               sbagliate: precision 0.13 sulla classe 3, 0.39 sulla
               classe 0, a fronte di recall altissimo per entrambe).
    """
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
        # attenua: peso finale = peso_pieno ** balance_strength (1.0 resta invariato)
        class_weight = {cls: weight ** balance_strength for cls, weight in full_balanced.items()}

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight=class_weight,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    if len(class_counts) >= 2 and class_counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("\n" + "=" * 60)
        print("VALUTAZIONE RANDOM FOREST")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
        print("\nReport per classe (0=ultima, 1=penultima, 2=antepenultima, 3=preantepenultima):")
        print(classification_report(y_test, predictions, zero_division=0))
    else:
        model.fit(X, y)

    return model


# ============================================================
# PREDIZIONE ACCENTO
# ============================================================

def predict_stress(word: str, model: Pipeline) -> dict:
    """
    Predice la classe di accento (0-3) per una parola, vincolando la
    scelta alle sole classi FISICAMENTE possibili dato il numero di
    sillabe della parola (una parola di N sillabe non può avere
    accento oltre la classe N-1: es. 3 sillabe -> al massimo
    "antepenultima" = classe 2, mai "preantepenultima" = classe 3).

    In precedenza il modello sceglieva sempre la classe con probabilità
    più alta tra tutte e 4, anche quando non aveva senso per la parola
    in questione (es. "giardino", 3 sillabe, classificato come
    "preantepenultima sillaba" - classe che richiede almeno 4 sillabe).
    """
    features_df = pd.DataFrame([build_word_features(word)])
    syll_count = int(features_df.iloc[0]["syll_count"])
    max_valid_class = min(3, max(0, syll_count - 1))

    probabilities = {}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)[0]
        classes = model.named_steps["classifier"].classes_
        probabilities = {int(cls): float(prob) for cls, prob in zip(classes, proba)}

        valid_probs = {c: p for c, p in probabilities.items() if c <= max_valid_class}
        if valid_probs:
            prediction = max(valid_probs, key=valid_probs.get)
        else:
            prediction = min(int(model.predict(features_df)[0]), max_valid_class)
    else:
        prediction = min(int(model.predict(features_df)[0]), max_valid_class)

    confidence = probabilities.get(prediction, 0.0)

    return {
        "model_position": prediction,
        "model_label": STRESS_LABELS.get(prediction, "sconosciuta"),
        "confidence": confidence,
        "syll_count": syll_count,
    }


def analyze_word(word: str, model: Pipeline, phon_dict: dict) -> dict:
    model_result = predict_stress(word, model)
    w_norm = norm_word(word)

    phon_position = phon_dict.get(w_norm)
    phon_label = (
        STRESS_LABELS.get(phon_position, "non disponibile")
        if phon_position is not None
        else "non disponibile"
    )

    agreement = (
        model_result["model_position"] == phon_position
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

def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File di testo non trovato: {path}")

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as file:
                text = file.read()
            return normalize_text(text)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Impossibile leggere il file: {path}")


# ============================================================
# UTILITÀ DI CONTESTO (spaCy)
# ============================================================

def punctuation_after_token(doc, token_index) -> str:
    for next_token in doc[token_index + 1:]:
        if next_token.is_space:
            continue
        if next_token.is_punct:
            return next_token.text
        break
    return ""


def punctuation_before_token(doc, token_index) -> str:
    for prev_token in reversed(doc[:token_index]):
        if prev_token.is_space:
            continue
        if prev_token.is_punct:
            return prev_token.text
        break
    return ""


def sentence_position(token, sentence) -> str:
    content_tokens = [t for t in sentence if not t.is_space and not t.is_punct]
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
    return mapping.get(token.dep_, token.dep_)


def syntactic_depth(token) -> int:
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
# ANALISI SEMANTICA (OPZIONALE — vedi nota in fondo al file)
# ============================================================

SENTIMENT_LEXICON: dict[str, float] = {
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

NEGATION_LEMMAS = {"non", "nessuno", "niente", "mai", "senza"}
INTENSIFIER_LEMMAS = {"molto": 1.4, "tanto": 1.3, "troppo": 1.4, "davvero": 1.3, "estremamente": 1.6}
DIMINISHER_LEMMAS = {"poco": 0.6, "leggermente": 0.6, "quasi": 0.7}

ENTITY_CATEGORY_LABELS = {"PER": "persona", "LOC": "luogo", "GPE": "luogo", "ORG": "organizzazione", "MISC": "altro"}


def entity_category(token) -> str:
    if not token.ent_type_:
        return "nessuna"
    return ENTITY_CATEGORY_LABELS.get(token.ent_type_, token.ent_type_.lower())


def semantic_polarity(token, sentence_tokens) -> tuple[float, str]:
    lemma = token.lemma_.lower().strip()
    base = SENTIMENT_LEXICON.get(lemma, 0.0)

    if base == 0.0:
        return 0.0, "neutro"

    value = base
    tokens_in_sentence = list(sentence_tokens)

    if token in tokens_in_sentence:
        position = tokens_in_sentence.index(token)
        for prev in reversed(tokens_in_sentence[max(0, position - 2):position]):
            prev_lemma = prev.lemma_.lower().strip()
            if prev_lemma in NEGATION_LEMMAS:
                value *= -1
            elif prev_lemma in INTENSIFIER_LEMMAS:
                value *= INTENSIFIER_LEMMAS[prev_lemma]
            elif prev_lemma in DIMINISHER_LEMMAS:
                value *= DIMINISHER_LEMMAS[prev_lemma]

    value = max(-1.0, min(1.0, value))
    label = "positivo" if value > 0.15 else "negativo" if value < -0.15 else "neutro"
    return value, label


def semantic_similarity_to_sentence(token, sentence) -> float:
    if not HAS_SEMANTIC_VECTORS or not token.has_vector:
        return 0.0
    if sentence.vector_norm == 0.0 or token.vector_norm == 0.0:
        return 0.0
    try:
        return float(token.similarity(sentence))
    except Exception:
        return 0.0


def semantic_salience(token, polarity_value: float) -> float:
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
    text: str, model: Pipeline, phon_dict: dict, use_semantics: bool = False
) -> list[dict]:
    doc = nlp(text)

    word_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    word_index_map = {t.i: i for i, t in enumerate(word_tokens)}

    results = []

    for index, token in enumerate(word_tokens):
        result = analyze_word(token.text, model, phon_dict)

        sentence = token.sent
        sentence_tokens = [t for t in sentence if not t.is_space and not t.is_punct]
        sentence_id = sentence.start

        punct_before = punctuation_before_token(doc, token.i)
        punct_after = punctuation_after_token(doc, token.i)

        previous_word = word_tokens[index - 1].text if index > 0 else ""
        next_word = word_tokens[index + 1].text if index < len(word_tokens) - 1 else ""

        sentence_word_index = (
            word_index_map.get(token.i, 0) - word_index_map.get(sentence_tokens[0].i, 0)
            if sentence_tokens
            else 0
        )
        sentence_word_count = len(sentence_tokens)

        result.update({
            "sentence_id": sentence_id,
            "sentence": sentence.text.strip(),
            "token_index": token.i,
            "word_index": index,
            "token_text": token.text,
            "normalized_word": norm_word(token.text),
            "lemma": token.lemma_,
            "word_length": len(norm_word(token.text)),
            "syllable_count": result["syll_count"],
            "pos": token.pos_,
            "tag": token.tag_,
            "morphology": str(token.morph),
            "dependency": token.dep_,
            "syntactic_role": syntactic_role(token),
            "head": token.head.text,
            "head_lemma": token.head.lemma_,
            "head_token_index": token.head.i,
            "syntactic_depth": syntactic_depth(token),
            "is_root": token.dep_ == "ROOT",
            "is_subject": token.dep_ in {"nsubj", "nsubj:pass"},
            "is_object": token.dep_ in {"obj", "iobj"},
            "is_modifier": token.dep_ in {"amod", "advmod", "nmod", "obl"},
            "sentence_word_index": sentence_word_index,
            "sentence_word_count": sentence_word_count,
            "sentence_position": sentence_position(token, sentence),
            "is_first_in_sentence": sentence_word_index == 0,
            "is_last_in_sentence": sentence_word_index == sentence_word_count - 1,
            "previous_word": previous_word,
            "next_word": next_word,
            "punctuation_before": punct_before,
            "punctuation_after": punct_after,
            "has_pause_after": punct_after in {",", ";", ":"},
            "has_sentence_end": punct_after in {".", "!", "?"},
            "is_question_end": punct_after == "?",
            "is_exclamation_end": punct_after == "!",
        })

        if use_semantics:
            polarity_value, polarity_label = semantic_polarity(token, sentence_tokens)
            similarity_value = semantic_similarity_to_sentence(token, sentence)
            salience_value = semantic_salience(token, polarity_value)

            result.update({
                "semantic_category": entity_category(token),
                "is_named_entity": bool(token.ent_type_),
                "semantic_polarity": round(polarity_value, 3),
                "semantic_polarity_label": polarity_label,
                "semantic_similarity_to_sentence": round(similarity_value, 3),
                "semantic_salience": salience_value,
                "rhythmic_weight": salience_value,
                "melodic_tension": round(abs(polarity_value), 3),
            })

        results.append(result)

    return results


# ============================================================
# SCHELETRO RITMICO (Fase 2 della pipeline Prosody2Music)
# ============================================================

def build_rhythm_skeleton(results: list[dict]) -> list[dict]:
    """
    Traduce i risultati per parola in una sequenza di impulsi a livello di
    sillaba, pronta per essere passata a music21:

        - "strong"  -> sillaba tonica (accentata)         -> nota forte
        - "weak"    -> sillaba atona                       -> nota debole
        - "pause_short"  -> dopo virgola/punto e virgola/due punti
        - "pause_long"   -> dopo punto/punto esclamativo/interrogativo

    La posizione della sillaba tonica viene ricavata da model_position
    (0 = ultima sillaba, 1 = penultima, ...), contando da destra; il
    numero di sillabe usato è syllable_count (dato reale se disponibile,
    altrimenti stimato — vedi estimate_syllables()).
    """
    pulses: list[dict] = []

    for word_result in results:
        n_syll = max(1, int(word_result.get("syllable_count", 1)))
        stress_from_end = word_result.get("model_position", 0)
        stress_index_from_start = n_syll - 1 - stress_from_end
        stress_index_from_start = max(0, min(n_syll - 1, stress_index_from_start))

        for syll_idx in range(n_syll):
            pulses.append({
                "word": word_result["word"],
                "syllable_index": syll_idx,
                "syllable_count": n_syll,
                "pulse": "strong" if syll_idx == stress_index_from_start else "weak",
            })

        if word_result.get("has_sentence_end"):
            pulses.append({"word": None, "syllable_index": None, "syllable_count": None, "pulse": "pause_long"})
        elif word_result.get("has_pause_after"):
            pulses.append({"word": None, "syllable_index": None, "syllable_count": None, "pulse": "pause_short"})

    return pulses


# ============================================================
# EXPORT MIDI (Fase 2/3 della pipeline: scheletro ritmico -> music21 -> MIDI)
# ============================================================

# Mappa General MIDI percussioni (canale 10) più comuni, per riferimento:
#   35 Acoustic Bass Drum   36 Bass Drum 1        38 Acoustic Snare
#   40 Electric Snare       42 Closed Hi-Hat      44 Pedal Hi-Hat
#   46 Open Hi-Hat          49 Crash Cymbal 1     51 Ride Cymbal 1
GM_DRUM_NOTES = {
    "bass_drum": 36,
    "snare": 38,
    "closed_hihat": 42,
    "open_hihat": 46,
    "crash": 49,
    "ride": 51,
}


def build_midi_from_skeleton(
    skeleton: list[dict],
    output_path: Path,
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
    Converte lo scheletro ritmico (lista di impulsi strong/weak/pause,
    vedi build_rhythm_skeleton) in un file MIDI tramite music21.

    Non è ancora melodia: ogni sillaba forte usa strong_pitch, ogni
    sillaba debole usa weak_pitch (stessa altezza fissa per tutta la
    traccia) — questo è volutamente solo lo "scheletro ritmico" della
    pipeline (Fase 2/3); l'altezza reale verrà decisa in Fase 4 quando
    si genera la melodia (MelodyRNN/Music Transformer) usando questo
    file come primer/riferimento ritmico.

    Se use_drums=True, invece di note intonate viene usato il canale
    percussioni General MIDI (canale 10): drum_strong per le sillabe
    toniche (default: grancassa, nota 36) e drum_weak per le atone
    (default: hi-hat chiuso, nota 42). Le pause restano silenzi in
    entrambi i casi. Vedi GM_DRUM_NOTES per altre combinazioni comuni.

    Durate:
        strong       -> semiminima  (1.0 quarterLength)
        weak         -> croma       (0.5 quarterLength)
        pause_short  -> croma di silenzio (dopo , ; :)
        pause_long   -> puntata di silenzio (dopo . ! ?)

    Richiede: pip install music21
    """
    try:
        from music21 import stream, note, tempo as m21tempo, instrument
    except ImportError as exc:
        raise SystemExit(
            "\nPer esportare il MIDI serve music21.\n\nEsegui:\n"
            "    pip install music21\n"
        ) from exc

    part = stream.Part()
    part.append(m21tempo.MetronomeMark(number=bpm))

    if use_drums:
        # Forza il routing sul canale 10 (percussioni GM): da qui in poi
        # il valore "pitch" delle note non è più un'altezza musicale ma
        # seleziona il suono percussivo secondo la mappa GM.
        part.insert(0, instrument.UnpitchedPercussion())

    for pulse in skeleton:
        kind = pulse["pulse"]

        if kind == "strong":
            if use_drums:
                n = note.Note()
                n.pitch.midi = drum_strong
            else:
                n = note.Note(strong_pitch)
            n.quarterLength = 1.0
            n.volume.velocity = strong_velocity
            part.append(n)
        elif kind == "weak":
            if use_drums:
                n = note.Note()
                n.pitch.midi = drum_weak
            else:
                n = note.Note(weak_pitch)
            n.quarterLength = 0.5
            n.volume.velocity = weak_velocity
            part.append(n)
        elif kind == "pause_short":
            r = note.Rest()
            r.quarterLength = 0.5
            part.append(r)
        elif kind == "pause_long":
            r = note.Rest()
            r.quarterLength = 1.5
            part.append(r)

    part.write("midi", fp=str(output_path))

    total_quarter_length = sum(n.quarterLength for n in part.notesAndRests)
    expected_seconds = total_quarter_length * (60.0 / bpm)
    print(
        f"[MIDI] Durata attesa dai dati: {total_quarter_length:.2f} quarterLength "
        f"a {bpm} bpm -> ~{expected_seconds:.1f} secondi "
        "(se il tuo player mostra un valore diverso, probabilmente non applica subito "
        "il MetronomeMark iniziale o aggiunge silenzio di padding)."
    )


# ============================================================
# STAMPA RISULTATI
# ============================================================

def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("ANALISI TESTO")
    print("=" * 120)

    headers = [
        "word", "model_label", "phon_label", "confidence", "agreement",
        "syll_count", "lemma", "pos", "dependency", "syntactic_role", "sentence_position",
    ]
    print(" | ".join(f"{h:<20}" for h in headers))
    print("-" * 120)

    for result in results:
        values = [
            result.get("word", ""),
            result.get("model_label", ""),
            result.get("phon_label", ""),
            f"{result.get('confidence', 0) * 100:.1f}%",
            result.get("agreement", ""),
            result.get("syllable_count", result.get("syll_count", "")),
            result.get("lemma", ""),
            result.get("pos", ""),
            result.get("dependency", ""),
            result.get("syntactic_role", ""),
            result.get("sentence_position", ""),
        ]
        print(" | ".join(f"{str(v):<20}" for v in values))

    print("=" * 120)


def print_statistics(results: list[dict], use_semantics: bool) -> None:
    comparisons = [r for r in results if r["agreement"] is not None]

    print("\n" + "=" * 60)
    print("STATISTICHE")
    print("=" * 60)
    print(f"Parole analizzate: {len(results)}")
    print(f"Frasi individuate: {len(set(r['sentence_id'] for r in results))}")
    print(f"Confronti con phonItaliaR: {len(comparisons)}")

    if comparisons:
        correct = sum(bool(r["agreement"]) for r in comparisons)
        print(f"Accordi: {correct}")
        print(f"Accuratezza confronto: {correct / len(comparisons) * 100:.2f}%")

    if use_semantics:
        print("=" * 60)
        print("SEMANTICA")
        print("=" * 60)
        entities = [r for r in results if r.get("is_named_entity")]
        polarities = [
            r["semantic_polarity"] for r in results
            if r.get("semantic_polarity_label") != "neutro"
        ]
        print(f"Entità nominate individuate: {len(entities)}")
        print(f"Parole con carica emotiva riconosciuta: {len(polarities)}")
        if polarities:
            avg_polarity = sum(polarities) / len(polarities)
            print(f"Polarità media (parole non neutre): {avg_polarity:+.2f}")
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
        description="Classificatore accenti + scheletro ritmico + analisi contestuale spaCy."
    )
    parser.add_argument("text_file", help="File .txt contenente il testo.")
    parser.add_argument("--q2stress", default=None, help="Percorso dataset Q2Stress.")
    parser.add_argument("--phonitalia-dataset", default=None, help="Percorso TSV phonItaliaR.")
    parser.add_argument("--output", default="risultati_accenti_contesto.csv", help="CSV di output (per parola).")
    parser.add_argument(
        "--rhythm-output", default=None,
        help="Se specificato, salva anche lo scheletro ritmico (per sillaba) in questo CSV."
    )
    parser.add_argument(
        "--midi-output", default=None,
        help="Se specificato, esporta lo scheletro ritmico anche come file .mid (richiede music21)."
    )
    parser.add_argument(
        "--bpm", type=int, default=100,
        help="Tempo (battiti al minuto) usato per l'export MIDI. Default: 100."
    )
    parser.add_argument(
        "--drums", action="store_true",
        help="Usa il canale percussioni GM invece del pianoforte per l'export MIDI."
    )
    parser.add_argument(
        "--drum-strong", type=int, default=GM_DRUM_NOTES["bass_drum"],
        help="Nota GM (canale 10) per le sillabe toniche con --drums. Default: 36 (grancassa)."
    )
    parser.add_argument(
        "--drum-weak", type=int, default=GM_DRUM_NOTES["closed_hihat"],
        help="Nota GM (canale 10) per le sillabe atone con --drums. Default: 42 (hi-hat chiuso)."
    )
    parser.add_argument("--no-phonitalia", action="store_true", help="Disattiva confronto con phonItaliaR.")
    parser.add_argument(
        "--semantics", action="store_true",
        help="Abilita l'analisi semantica (NER, polarità, similarità). Spenta di default."
    )
    parser.add_argument(
        "--class-balance", type=float, default=0.5,
        help=(
            "Quanto bilanciare le classi di accento durante il training: 0.0 = nessun "
            "bilanciamento, 1.0 = bilanciamento pieno (class_weight='balanced' di sklearn, "
            "tende a sovra-predire le classi rare). Default: 0.5 (via di mezzo)."
        ),
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_arguments()

    print("\n" + "=" * 80)
    print("ACCENTI + SCHELETRO RITMICO + ANALISI DEL CONTESTO")
    print("=" * 80)

    df_q2stress = load_q2stress_dataframe(args.q2stress)
    frame = build_training_frame(df_q2stress)

    phon_dict = {}
    if not args.no_phonitalia:
        df_phon = load_phonitalia_dataset(args.phonitalia_dataset)
        if df_phon is not None:
            frame = enrich_with_phonitalia_dataset(frame, df_phon)
            phon_dict = build_phonitalia_dict(df_phon)
            print(f"[phonItaliaR] Caricate {len(phon_dict)} parole.")

    print("\n[MODELLO] Addestramento Random Forest...")
    model = train_model(frame, balance_strength=args.class_balance)
    print("[MODELLO] Addestramento completato.")

    text_path = Path(args.text_file)
    text = load_text_file(text_path)
    if not text:
        raise ValueError("Il file di testo è vuoto.")

    print("\n[INPUT] Testo caricato:")
    print(text[:500] + ("..." if len(text) > 500 else ""))

    print("\n[SPACY] Analisi linguistica...")
    results = analyze_text(text, model, phon_dict, use_semantics=args.semantics)

    if not results:
        raise ValueError("Non sono state trovate parole nel testo.")

    print_results(results)
    print_statistics(results, use_semantics=args.semantics)

    output_path = Path(args.output)
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[OUTPUT] CSV per parola salvato in: {output_path}")

    skeleton = None
    if args.rhythm_output or args.midi_output:
        skeleton = build_rhythm_skeleton(results)

    if args.rhythm_output:
        rhythm_path = Path(args.rhythm_output)
        pd.DataFrame(skeleton).to_csv(rhythm_path, index=False, encoding="utf-8-sig")
        print(f"[OUTPUT] Scheletro ritmico salvato in: {rhythm_path}")

    if args.midi_output:
        midi_path = Path(args.midi_output)
        build_midi_from_skeleton(
            skeleton,
            midi_path,
            bpm=args.bpm,
            use_drums=args.drums,
            drum_strong=args.drum_strong,
            drum_weak=args.drum_weak,
        )
        print(f"[OUTPUT] MIDI salvato in: {midi_path}")

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()
