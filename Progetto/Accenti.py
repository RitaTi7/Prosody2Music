#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accenti.py
==========

Classificatore automatico dell'accento tonico delle parole italiane.

Il programma utilizza:

1. Q2Stress
   - file lexElem.txt
   - colonne: word, StressPattern

2. phonItaliaR (file TSV raw)
   - Utilizza direttamente il file phonItalia-1.10-wordforms.tsv
   - Utilizzato per ottenere il pattern di accento esatto e 
     per arricchire il dataset di addestramento.

3. Una lista di parole italiane da analizzare.

Il classificatore predice:

    0 = ultima sillaba
    1 = penultima sillaba
    2 = antepenultima sillaba
    3 = preantepenultima sillaba

Esempio:

    musica       -> antepenultima
    telefono     -> penultima
    università   -> ultima
    pianoforte   -> penultima

REQUISITI
=========

Python 3.10+
Pacchetti Python:

    pip install pandas scikit-learn

ESECUZIONE
==========

    python Accenti.py

Oppure:

    python Accenti.py parole.txt
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
    from sklearn.metrics import classification_report, accuracy_score
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
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

Q2STRESS_CANDIDATES = [
    BASE_DIR / "Q2Stress" / "scripts" / "children" / "lexElem.txt",
    BASE_DIR / "Q2Stress" / "scripts" / "adults" / "phonItalia 1.10.1 - word forms.txt",
    BASE_DIR / "Progetto" / "Q2Stress" / "scripts" / "children" / "lexElem.txt",
]

# Percorsi aggiornati basati su image_8f3611.png
PHONITALIAR_DATASET_CANDIDATES = [
    BASE_DIR / "phonItaliaR" / "data-raw" / "phonItalia-1.10" / "phonItalia-1.10-wordforms.tsv",
    BASE_DIR / "Progetto" / "phonItaliaR" / "data-raw" / "phonItalia-1.10" / "phonItalia-1.10-wordforms.tsv",
    BASE_DIR / "phonItalia-1.10-wordforms.tsv",
]

DEFAULT_WORD_LIST = BASE_DIR / "parole.txt"


# ============================================================
# NORMALIZZAZIONE E FEATURE
# ============================================================

def norm_word(word: str) -> str:
    text = str(word).strip().lower()
    text = text.replace("'", "")
    text = text.replace("’", "")
    text = re.sub(r"[^a-zàèéìòóù]", "", text)
    return text

def estimate_syllables(word: str) -> int:
    w = norm_word(word)
    if not w: return 1
    vowels = "aeiouàèéìòóù"
    count = 0
    previous_vowel = False
    for char in w:
        is_vowel = char in vowels
        if is_vowel and not previous_vowel:
            count += 1
        previous_vowel = is_vowel
    return max(1, count)

def build_word_features(word: str, row: Optional[pd.Series] = None) -> dict:
    w = norm_word(word)
    letters = [c for c in w if c.isalpha()]
    vowels = "aeiouàèéìòóù"

    vowel_count = sum(char in vowels for char in letters)
    consonant_count = max(len(letters) - vowel_count, 0)
    syll_count = 1

    if row is not None:
        value = row.get("SumSylls")
        if pd.notna(value):
            try:
                syll_count = int(float(value))
            except (TypeError, ValueError):
                syll_count = estimate_syllables(w)

    if syll_count < 1:
        syll_count = estimate_syllables(w)

    return {
        "word_len": len(letters),
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "vowel_ratio": (vowel_count / max(len(letters), 1)),
        "syll_count": syll_count,
        "ends_with_vowel": int(bool(letters) and letters[-1] in vowels),
        "ends_with_consonant": int(bool(letters) and letters[-1] not in vowels),
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
# CARICAMENTO DATASET Q2STRESS
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
            df = pd.read_csv(
                path, sep="\t", engine="python", encoding=encoding, on_bad_lines="skip"
            )
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Impossibile leggere il file Q2Stress: {path}")

def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        word = row.get("word")
        if word is None: continue
        word = norm_word(word)
        if not word: continue

        target = row.get("StressPattern")
        if pd.isna(target): continue
        try:
            target = int(float(target))
        except (TypeError, ValueError):
            continue

        if target not in {0, 1, 2, 3}: continue

        features = build_word_features(word, row)
        features["target"] = target
        rows.append(features)

    if not rows:
        raise ValueError("\nNon sono state trovate righe valide nel dataset Q2Stress.")
    return pd.DataFrame(rows)


# ============================================================
# CARICAMENTO DATASET PHONITALIAR DIRECT TSV
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
    print("[phonItaliaR] Attenzione: Impossibile leggere il TSV.")
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
            if col_norm == name.strip().lower().replace(" ", "").replace("_", ""):
                return col
    return None


def phonitalia_stress_to_class(stressed_syllable: object, total_syllables: object) -> Optional[int]:
    """Converte il campo PhonItaliaR `StressedSyllable` nel formato interno {0,1,2,3}."""
    try:
        stressed = int(float(stressed_syllable))
        total = int(float(total_syllables))
    except (TypeError, ValueError):
        return None

    if stressed < 1 or total < 1:
        return None

    # `StressedSyllable` è 1-based: 1 = prima sillaba, N = ultima sillaba.
    # Il modello usa 0 = ultima, 1 = penultima, 2 = antepenultima, 3 = preantepenultima.
    class_index = total - stressed
    if class_index < 0:
        class_index = 0
    if class_index > 3:
        class_index = 3
    return int(class_index)


def build_phonitalia_dict(df: pd.DataFrame) -> dict:
    """
    Costruisce un dizionario (O(1) lookup) per validare le parole
    direttamente dal dataframe caricato, usando il formato reale di PhonItaliaR.
    """
    phon_dict: dict[str, int] = {}
    if df is None:
        return phon_dict

    word_col = detect_column(df, ["word", "parola", "wordform", "orthography"])
    syll_col = detect_column(df, ["sumsylls", "sum_sylls", "nsyllables", "numsyllables", "syllables"])
    stress_col = detect_column(df, ["stressedsyllable", "stressed_syllable", "stressedsyllable", "stress", "stresspattern", "accento"])

    if not word_col or not syll_col or not stress_col:
        return phon_dict

    for _, row in df.iterrows():
        word = norm_word(row.get(word_col, ""))
        if not word:
            continue

        target = phonitalia_stress_to_class(row.get(stress_col), row.get(syll_col))
        if target is None:
            continue

        current = phon_dict.get(word)
        if current is None:
            phon_dict[word] = target
            continue

        # Se ci sono duplicati per la stessa parola, preferiamo la classe più frequente
        # o la prima occorrenza. In PhonItaliaR i duplicati sono quasi sempre ripetizioni
        # dello stesso lemma con la stessa posizione d'accento.
        if current != target:
            phon_dict[word] = target

    return phon_dict


def enrich_with_phonitalia_dataset(frame: pd.DataFrame, df_phon: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_phon is None:
        return frame

    word_col = detect_column(df_phon, ["word", "parola", "wordform", "orthography"])
    syll_col = detect_column(df_phon, ["sumsylls", "sum_sylls", "nsyllables", "numsyllables", "syllables"])
    stress_col = detect_column(df_phon, ["stressedsyllable", "stressed_syllable", "stressedsyllable", "stress", "stresspattern", "accento"])

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

        features = build_word_features(word)
        features["target"] = target
        added.append(features)

    if not added:
        return frame

    extra = pd.DataFrame(added)
    combined = pd.concat([frame, extra], ignore_index=True).drop_duplicates(subset=["word_len", "vowel_count", "target"])
    print(f"[phonItaliaR dataset] Utilizzato per arricchire il dataset.")
    print(f"[Training] Righe complessive: {len(combined)}")
    return combined


# ============================================================
# TRAINING
# ============================================================

def train_model(frame: pd.DataFrame) -> Pipeline:
    X = frame.drop(columns=["target"]).copy()
    y = frame["target"]

    for col in ["last1", "last2", "last3", "last4"]:
        if col in X.columns: X[col] = X[col].astype(str)

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols: transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    class_counts = y.value_counts()
    if len(class_counts) >= 2 and class_counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("\n" + "=" * 60)
        print("VALUTAZIONE RANDOM FOREST")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    else:
        model.fit(X, y)
    return model


# ============================================================
# PREDIZIONE MODELLO E ANALISI
# ============================================================

STRESS_LABELS = {
    0: "ultima sillaba",
    1: "penultima sillaba",
    2: "antepenultima sillaba",
    3: "preantepenultima sillaba",
}

def predict_stress(word: str, model: Pipeline) -> dict:
    features = pd.DataFrame([build_word_features(word)])
    prediction = int(model.predict(features)[0])
    probabilities = {}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = model.named_steps["classifier"].classes_
        probabilities = {int(cls): float(prob) for cls, prob in zip(classes, proba)}

    confidence = probabilities.get(prediction, 0.0)
    return {
        "word": word,
        "prediction": prediction,
        "label": STRESS_LABELS.get(prediction, "sconosciuta"),
        "confidence": confidence,
    }


def analyze_word(word: str, model: Pipeline, phon_dict: dict) -> dict:
    model_result = predict_stress(word, model)
    w_norm = norm_word(word)
    
    phon_position = phon_dict.get(w_norm)
    phon_label = STRESS_LABELS.get(phon_position, "non disponibile") if phon_position is not None else "non disponibile"

    agreement = (model_result["prediction"] == phon_position) if phon_position is not None else None

    return {
        "word": word,
        "model_position": model_result["prediction"],
        "model_label": model_result["label"],
        "confidence": model_result["confidence"],
        "phon_position": phon_position,
        "phon_label": phon_label,
        "agreement": agreement,
    }


# ============================================================
# LETTURA LISTA PAROLE & OUTPUT
# ============================================================

def load_word_list(path: Path) -> list[str]:
    if not path.exists(): raise FileNotFoundError(f"Lista parole non trovata: {path}")
    words = []
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            word = norm_word(line.split("\t")[0].strip())
            if word: words.append(word)
    return words

def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("RISULTATI INDIVIDUAZIONE ACCENTO TONICO")
    print("=" * 100)
    print(f"{'PAROLA':<20}{'MODELLO':<26}{'PHONITALIAR':<26}{'CONF.':<8}{'OK':<5}")
    print("-" * 100)
    for res in results:
        conf = f"{res['confidence'] * 100:.1f}%"
        ok = "SI" if res["agreement"] is True else ("NO" if res["agreement"] is False else "-")
        print(f"{res['word']:<20}{res['model_label']:<26}{res['phon_label']:<26}{conf:<8}{ok:<5}")
    print("=" * 100)

def print_statistics(results: list[dict]) -> None:
    comparisons = [r for r in results if r["agreement"] is not None]
    if not comparisons:
        print("\nNessun confronto disponibile con phonItaliaR.")
        return
    correct = sum(r["agreement"] for r in comparisons)
    total = len(comparisons)
    print("\n" + "=" * 60)
    print("CONFRONTO MODELLO / PHONITALIAR TSV")
    print("=" * 60)
    print(f"Parole confrontate: {total}")
    print(f"Predizioni corrette: {correct}")
    print(f"Accordo: {correct / total * 100:.2f}%")


# ============================================================
# ARGUMENT PARSER & MAIN
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Classificatore Accenti con Q2Stress + phonItaliaR TSV.")
    parser.add_argument("word_list", nargs="?", default=str(DEFAULT_WORD_LIST), help="File contenente le parole.")
    parser.add_argument("--q2stress", default=None, help="Percorso Q2Stress.")
    parser.add_argument("--phonitalia-dataset", default=None, help="Percorso file TSV di phonItaliaR.")
    parser.add_argument("--output", default="risultati_accenti.csv", help="CSV di output.")
    parser.add_argument("--no-phonitalia", action="store_true", help="Disattiva controllo con phonItaliaR.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("\n" + "=" * 70)
    print("ACCENTI - RICONOSCIMENTO DELL'ACCENTO TONICO (Lettura TSV Diretta)")
    print("=" * 70)

    # 1. Caricamento Q2Stress
    df_q2stress = load_q2stress_dataframe(args.q2stress)
    frame = build_training_frame(df_q2stress)
    
    # 2. Caricamento phonItaliaR TSV
    phon_dict = {}
    if not args.no_phonitalia:
        df_phon = load_phonitalia_dataset(args.phonitalia_dataset)
        if df_phon is not None:
            frame = enrich_with_phonitalia_dataset(frame, df_phon)
            phon_dict = build_phonitalia_dict(df_phon)
            print(f"[phonItaliaR] Caricate {len(phon_dict)} parole nel dizionario di test.")

    # 3. Training
    print("\n[MODELLO] Addestramento Random Forest...")
    model = train_model(frame)
    print("[MODELLO] Addestramento completato.")

    # 4. Inferenza
    words = load_word_list(Path(args.word_list))
    if not words: raise ValueError("La lista delle parole è vuota.")
    
    print(f"\n[INPUT] Analisi in corso per {len(words)} parole...\n")
    results = [analyze_word(word, model, phon_dict) for word in words]

    # 5. Output
    print_results(results)
    print_statistics(results)
    
    pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nAnalisi completata. File salvato in {args.output}")

if __name__ == "__main__":
    main()