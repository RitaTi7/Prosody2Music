from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re
import unicodedata
import argparse
import json

import pandas as pd
import pyphen
import pretty_midi


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

Q2STRESS_DIR = PROJECT_DIR / "Q2Stress"
PHONITALIA_DIR = PROJECT_DIR / "phonitaliaR"

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MIDI = OUTPUT_DIR / "rhythmic_skeleton.mid"
SEMANTIC_PROFILE_JSON = OUTPUT_DIR / "semantic_profile.json"


# Tempo musicale
BPM = 100

# Nota provvisoria.
# Nella fase Music Transformer NON useremo più questa altezza.
PLACEHOLDER_PITCH = 60  # C4

# Durate in beat
STRESSED_DURATION = 0.50
UNSTRESSED_DURATION = 0.25

# Velocity MIDI
STRESSED_VELOCITY = 110
UNSTRESSED_VELOCITY = 55

# Pause dovute alla punteggiatura
PAUSES = {
    ",": 0.25,
    ";": 0.50,
    ":": 0.50,
    ".": 1.00,
    "!": 1.00,
    "?": 1.00,
}


# ============================================================
# STRUTTURE DATI
# ============================================================

@dataclass
class Syllable:

    text: str
    word: str

    # indice 0-based
    index_in_word: int

    stressed: bool

    # da quale fonte abbiamo ottenuto lo stress
    stress_source: str


@dataclass
class WordAnalysis:

    word: str
    syllables: list[Syllable]

    stressed_index: Optional[int]

    stress_source: str


@dataclass
class RhythmicEvent:

    syllable: Optional[Syllable]

    start: float
    duration: float

    velocity: int

    pitch: int

    # True se è una pausa
    is_rest: bool = False


# ============================================================
# UTILITY TESTO
# ============================================================

def normalize_word(word: str) -> str:
    """
    Normalizza una parola per il lookup nei dataset.

    Esempi:

        "Musica" -> "musica"
        "CITTÀ"  -> "città"
        "perché" -> "perché"
    """

    word = word.strip().lower()

    # normalizzazione Unicode
    word = unicodedata.normalize(
        "NFC",
        word
    )

    return word


# ============================================================
# SILLABAZIONE
# ============================================================

class ItalianSyllabifier:

    def __init__(self):

        self.hyphenator = pyphen.Pyphen(
            lang="it_IT"
        )

    def syllabify(self, word: str) -> list[str]:

        word = normalize_word(word)

        if not word:
            return []

        result = self.hyphenator.inserted(word)

        syllables = result.split("-")

        if not syllables:
            return [word]

        return syllables


# ============================================================
# LOADER GENERICO DEI FILE
# ============================================================

def load_data_files(directory: Path) -> list[pd.DataFrame]:

    """
    Cerca automaticamente file di dati dentro una cartella.

    Supporta:

        CSV
        TSV
        XLS/XLSX
        RDA
        RData

    Restituisce una lista di DataFrame.
    """

    dataframes = []

    if not directory.exists():

        print(
            f"[WARNING] Cartella non trovata: {directory}"
        )

        return dataframes

    print()
    print("=" * 70)
    print(f"RICERCA DATASET: {directory}")
    print("=" * 70)

    # --------------------------------------------------------
    # CSV / TSV
    # --------------------------------------------------------

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:

            # CSV
            if suffix == ".csv":

                print(f"[CSV] {path}")

                df = pd.read_csv(
                    path,
                    sep=None,
                    engine="python"
                )

                if not df.empty:
                    dataframes.append(df)

            # TSV / TXT
            elif suffix in {".tsv", ".tab"}:

                print(f"[TSV] {path}")

                df = pd.read_csv(
                    path,
                    sep="\t"
                )

                if not df.empty:
                    dataframes.append(df)

            # Excel
            elif suffix in {".xls", ".xlsx"}:

                print(f"[EXCEL] {path}")

                excel = pd.ExcelFile(path)

                for sheet in excel.sheet_names:

                    df = pd.read_excel(
                        path,
                        sheet_name=sheet
                    )

                    if not df.empty:
                        dataframes.append(df)

        except Exception as e:

            print(
                f"[WARNING] Errore leggendo {path}: {e}"
            )

    # --------------------------------------------------------
    # RDATA / RDA
    # --------------------------------------------------------

    r_files = list(directory.rglob("*.rda"))

    r_files += list(directory.rglob("*.RData"))

    if r_files:

        try:

            import pyreadr

        except ImportError:

            print(
                "\n[WARNING] Trovati file RData/RDA ma "
                "pyreadr non è installato."
            )

            print(
                "Installa con:"
            )

            print(
                "pip install pyreadr"
            )

        else:

            for path in r_files:

                print(f"[RDATA] {path}")

                try:

                    result = pyreadr.read_r(
                        str(path)
                    )

                    for key, df in result.items():

                        if df is not None and not df.empty:

                            print(
                                f"    oggetto R: {key} "
                                f"{df.shape}"
                            )

                            dataframes.append(df)

                except Exception as e:

                    print(
                        f"[WARNING] "
                        f"Impossibile leggere {path}: {e}"
                    )

    return dataframes


# ============================================================
# INDIVIDUAZIONE DATAFRAME PHONITALIA
# ============================================================

def find_phonitalia_dataframe(
    dataframes: list[pd.DataFrame]
) -> Optional[pd.DataFrame]:

    """
    Cerca il DataFrame phonItalia.

    Il dataset documentato contiene colonne come:

        word
        PhoneSyll
        SumSylls
        StressedSyllable
    """

    required_columns = {
        "word",
        "StressedSyllable"
    }

    for df in dataframes:

        columns = set(df.columns)

        if required_columns.issubset(columns):

            print()
            print(
                "[OK] Trovato dataset phonItalia."
            )

            print(
                f"Righe: {len(df)}"
            )

            print(
                f"Colonne: {len(df.columns)}"
            )

            return df

    return None


# ============================================================
# PHONITALIA LOOKUP
# ============================================================

class PhonItaliaLookup:

    def __init__(
        self,
        dataframe: pd.DataFrame
    ):

        self.df = dataframe.copy()

        # ----------------------------------------------------
        # Normalizzazione parola
        # ----------------------------------------------------

        self.df["_word_normalized"] = (
            self.df["word"]
            .astype(str)
            .map(normalize_word)
        )

        # ----------------------------------------------------
        # Dizionario:
        #
        # parola -> record
        # ----------------------------------------------------

        self.lookup = {}

        for _, row in self.df.iterrows():

            word = row["_word_normalized"]

            if not word:
                continue

            self.lookup[word] = row

    def get_record(
        self,
        word: str
    ) -> Optional[pd.Series]:

        word = normalize_word(word)

        return self.lookup.get(word)

    def get_stress(
        self,
        word: str
    ) -> Optional[int]:

        record = self.get_record(word)

        if record is None:
            return None

        value = record["StressedSyllable"]

        if pd.isna(value):
            return None

        try:

            value = int(value)

        except (ValueError, TypeError):

            return None

        # ----------------------------------------------------
        # IMPORTANTE
        #
        # Il dataset documenta StressedSyllable come indice
        # numerico della sillaba accentata.
        #
        # In questa pipeline assumiamo indice 1-based.
        # Lo convertiamo quindi a 0-based.
        # ----------------------------------------------------

        return value - 1

    def get_syllable_count(
        self,
        word: str
    ) -> Optional[int]:

        """
        Numero di sillabe secondo phonItalia (colonna SumSylls,
        se presente nel dataset).

        Serve a validare che la sillabazione ortografica di
        Pyphen sia coerente con quella (fonologica) da cui
        proviene l'indice di accento: se il conteggio non
        combacia, l'indice restituito da get_stress potrebbe
        riferirsi a una sillaba diversa da quella che finirà
        nella lista prodotta da Pyphen, ed è più sicuro non
        fidarsi di quel dato.
        """

        record = self.get_record(word)

        if record is None:
            return None

        if "SumSylls" not in record.index:
            return None

        value = record["SumSylls"]

        if pd.isna(value):
            return None

        try:
            return int(value)

        except (ValueError, TypeError):
            return None

    def get_syllables(
        self,
        word: str
    ) -> Optional[list[str]]:

        record = self.get_record(word)

        if record is None:
            return None

        # ----------------------------------------------------
        # PhoneSyll contiene la rappresentazione fonologica
        # con confini sillabici.
        #
        # Tuttavia per il nostro MIDI vogliamo le sillabe
        # ortografiche, quindi useremo Pyphen.
        # ----------------------------------------------------

        return None

    def print_example(
        self,
        word: str
    ):

        record = self.get_record(word)

        if record is None:

            print(
                f"{word}: NON TROVATA"
            )

            return

        print()
        print(f"Parola: {word}")

        for column in [
            "word",
            "PhoneSyll",
            "SumSylls",
            "StressedSyllable"
        ]:

            if column in record.index:

                print(
                    f"{column}: {record[column]}"
                )


# ============================================================
# Q2STRESS
# ============================================================

class Q2StressLookup:

    """
    Q2Stress non va trattato semplicemente come un dizionario
    parola -> stress.

    È un database di cue distribuzionali per lo stress,
    derivato in parte da PhonItalia.

    Per la prima versione lo carichiamo e lo utilizziamo
    come fonte secondaria / validazione.

    Se il file locale contiene direttamente una colonna
    con la posizione dello stress, questa classe la sfrutta.
    """

    POSSIBLE_WORD_COLUMNS = [
        "word",
        "Word",
        "WORD",
        "wordSpell",
        "stimulus",
        "Stimulus"
    ]

    POSSIBLE_STRESS_COLUMNS = [
        "StressedSyllable",
        "stressedSyllable",
        "stress",
        "Stress",
        "stress_position",
        "StressPosition"
    ]

    def __init__(
        self,
        dataframes: list[pd.DataFrame]
    ):

        self.lookup = {}

        self.dataframe = None

        # ----------------------------------------------------
        # Cerca un DataFrame con parola + stress
        # ----------------------------------------------------

        for df in dataframes:

            word_column = self._find_column(
                df,
                self.POSSIBLE_WORD_COLUMNS
            )

            stress_column = self._find_column(
                df,
                self.POSSIBLE_STRESS_COLUMNS
            )

            if word_column and stress_column:

                self.dataframe = df

                print(
                    "[OK] Q2Stress contiene "
                    "una tabella direttamente utilizzabile."
                )

                for _, row in df.iterrows():

                    word = normalize_word(
                        str(row[word_column])
                    )

                    value = row[stress_column]

                    if not word or pd.isna(value):
                        continue

                    try:
                        stress = int(value)

                    except (ValueError, TypeError):
                        continue

                    self.lookup[word] = stress

                break

        if not self.lookup:

            print(
                "[INFO] Q2Stress caricato, "
                "ma non è stato trovato un campo "
                "diretto parola -> stress."
            )

    @staticmethod
    def _find_column(
        df: pd.DataFrame,
        candidates: list[str]
    ) -> Optional[str]:

        for candidate in candidates:

            if candidate in df.columns:
                return candidate

        return None

    def get_stress(
        self,
        word: str
    ) -> Optional[int]:

        word = normalize_word(word)

        value = self.lookup.get(word)

        if value is None:
            return None

        # Possibile conversione 1-based -> 0-based.
        #
        # Anche qui la gestione definitiva dipenderà
        # dal file Q2Stress effettivamente trovato.

        return value - 1


# ============================================================
# ANALISI DELLA PAROLA
# ============================================================

class ItalianWordAnalyzer:

    def __init__(
        self,
        phonitalia: Optional[PhonItaliaLookup],
        q2stress: Optional[Q2StressLookup]
    ):

        self.phonitalia = phonitalia
        self.q2stress = q2stress

        self.syllabifier = pyphen.Pyphen(
            lang="it_IT"
        )

    def syllabify(
        self,
        word: str
    ) -> list[str]:

        word = normalize_word(word)

        result = self.syllabifier.inserted(
            word
        )

        if not result:
            return [word]

        return result.split("-")

    def analyze(
        self,
        word: str
    ) -> WordAnalysis:

        word = normalize_word(word)

        syllables = self.syllabify(word)

        # ----------------------------------------------------
        # PRIMA SCELTA: phonItalia
        # ----------------------------------------------------

        stress_index = None
        source = "none"

        if self.phonitalia is not None:

            stress_index = (
                self.phonitalia.get_stress(word)
            )

            if stress_index is not None:

                # ------------------------------------------------
                # Controllo di coerenza sillabica
                #
                # L'indice di accento in phonItalia si riferisce
                # alla SUA sillabazione (fonologica), non a
                # quella ortografica prodotta da Pyphen. Se il
                # numero di sillabe non coincide, l'indice
                # potrebbe puntare a una sillaba diversa da
                # quella prevista: meglio scartare il dato
                # piuttosto che accentare la sillaba sbagliata
                # senza accorgersene.
                # ------------------------------------------------

                expected_count = (
                    self.phonitalia.get_syllable_count(word)
                )

                if (
                    expected_count is not None
                    and expected_count != len(syllables)
                ):

                    stress_index = None

                else:
                    source = "phonitaliaR"

        # ----------------------------------------------------
        # SE NON TROVATA:
        # Q2Stress
        # ----------------------------------------------------

        if stress_index is None:

            if self.q2stress is not None:

                stress_index = (
                    self.q2stress.get_stress(word)
                )

                if stress_index is not None:
                    source = "Q2Stress"

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if stress_index is None:

            stress_index = (
                self.fallback_stress(
                    word,
                    syllables
                )
            )

            source = "fallback"

        # ----------------------------------------------------
        # Controllo validità
        # ----------------------------------------------------

        if (
            stress_index < 0
            or stress_index >= len(syllables)
        ):

            stress_index = self.fallback_stress(
                word,
                syllables
            )

            source = "fallback"

        # ----------------------------------------------------
        # Crea sillabe
        # ----------------------------------------------------

        syllable_objects = []

        for i, syllable in enumerate(
            syllables
        ):

            syllable_objects.append(
                Syllable(
                    text=syllable,
                    word=word,
                    index_in_word=i,
                    stressed=(
                        i == stress_index
                    ),
                    stress_source=source
                )
            )

        return WordAnalysis(
            word=word,
            syllables=syllable_objects,
            stressed_index=stress_index,
            stress_source=source
        )

    @staticmethod
    def fallback_stress(
        word: str,
        syllables: list[str]
    ) -> int:

        # ----------------------------------------------------
        # Accento grafico
        # ----------------------------------------------------

        accented = "àèéìòóù"

        for i, syllable in enumerate(
            syllables
        ):

            if any(
                char in accented
                for char in syllable.lower()
            ):

                return i

        # ----------------------------------------------------
        # Monosillabi
        # ----------------------------------------------------

        if len(syllables) == 1:
            return 0

        # ----------------------------------------------------
        # Default italiano:
        # penultima sillaba
        # ----------------------------------------------------

        return len(syllables) - 2


# ============================================================
# TOKENIZZAZIONE
# ============================================================

def tokenize_text(
    text: str
) -> list[str]:

    """
    Mantiene sia parole che punteggiatura.

    Gestisce l'apostrofo di elisione italiano (es. "dell'anima",
    "un'idea", "l'amico") tenendolo attaccato alla parola, così
    che il lookup nei dataset e la sillabazione operino sul token
    corretto invece che su frammenti spezzati ("dell" + "anima").

    Sia l'apostrofo dritto (') sia quello tipografico (') sono
    accettati.
    """

    return re.findall(
        r"\w+(?:['’]\w+)*|[,.!?;:]",
        text,
        flags=re.UNICODE
    )


# ============================================================
# ANALISI SEMANTICA → PROFILO MUSICALE
# ============================================================

EMOTION_LEXICON = {
    "triste": -1.0, "tristezza": -1.0, "solo": -0.8, "solitudine": -1.0,
    "piangere": -1.0, "pianto": -1.0, "lacrima": -0.9, "dolore": -1.0,
    "sofferenza": -1.0, "morte": -1.0, "morto": -1.0, "addio": -0.9,
    "paura": -0.8, "terrore": -1.0, "ansia": -0.8, "disperazione": -1.0,
    "buio": -0.7, "notte": -0.3, "perdita": -0.8,
    "felice": 1.0, "felicità": 1.0, "gioia": 1.0, "gioioso": 1.0,
    "amore": 0.9, "amare": 0.9, "sorriso": 0.8, "ridere": 0.9,
    "festa": 0.9, "vittoria": 1.0, "vincere": 0.9, "speranza": 0.7,
    "sole": 0.7, "luce": 0.7, "libertà": 0.8, "insieme": 0.4,
    "calma": 0.2, "pace": 0.4, "sereno": 0.5, "serenità": 0.5,
}

ENERGY_WORDS = {
    "corri": 1.0, "correre": 1.0, "scappa": 1.0, "scappare": 1.0,
    "lotta": 0.9, "combatti": 1.0, "combattere": 1.0, "battaglia": 0.9,
    "esplodi": 1.0, "esplosione": 1.0, "urla": 0.9, "urlare": 0.9,
    "salta": 0.9, "saltare": 0.9, "balla": 0.9, "ballare": 0.9,
    "festa": 0.8, "vittoria": 0.8, "coraggio": 0.7, "forza": 0.8,
    "veloce": 1.0, "presto": 0.9, "subito": 0.8, "adesso": 0.5,
}

MOVEMENT_WORDS = {
    "corri": 1.0, "correre": 1.0, "scappa": 1.0, "scappare": 1.0,
    "cammina": 0.5, "camminare": 0.5, "vola": 0.8, "volare": 0.8,
    "salta": 0.9, "saltare": 0.9, "balla": 1.0, "ballare": 1.0,
    "muovi": 0.9, "muovere": 0.9, "fuggi": 1.0, "fuggire": 1.0,
    "cade": 0.7, "cadere": 0.7, "sale": 0.6, "salire": 0.6,
    "scende": 0.6, "scendere": 0.6, "ritorna": 0.4, "tornare": 0.4,
}


def _normalize_semantic_token(token: str) -> str:
    token = token.lower().replace("’", "'")
    token = re.sub(r"[^\w']+", "", token, flags=re.UNICODE)
    return token.strip("'")


def _lexicon_value(words: list[str], lexicon: dict[str, float]) -> float:
    hits = [lexicon[w] for w in words if w in lexicon]
    return sum(hits) / len(hits) if hits else 0.0


def analyze_semantic_profile(text: str) -> dict:
    """Analisi semantica offline: valenza, energia, movimento e tensione."""
    tokens = tokenize_text(text)
    words = [_normalize_semantic_token(t) for t in tokens if t not in PAUSES]

    valence = _lexicon_value(words, EMOTION_LEXICON)
    explicit_energy = _lexicon_value(words, ENERGY_WORDS)
    movement = max(0.0, min(1.0, _lexicon_value(words, MOVEMENT_WORDS)))
    energy = max(0.0, min(1.0, 0.35 + 0.45 * abs(valence) + 0.35 * explicit_energy))

    exclamations = text.count("!")
    questions = text.count("?")
    energy = max(0.0, min(1.0, energy + 0.12 * min(exclamations, 3)))
    if movement == 0.0:
        movement = max(0.0, min(1.0, energy * 0.35))

    tension = max(0.0, min(1.0, 0.20 + 0.35 * abs(valence)
                             + 0.30 * movement + 0.10 * min(questions, 3)
                             + 0.08 * min(exclamations, 3)))
    intensity = max(0.0, min(1.0, 0.20 + 0.50 * abs(valence) + 0.30 * explicit_energy))

    return {
        "valence": round(max(-1.0, min(1.0, valence)), 3),
        "energy": round(energy, 3),
        "movement": round(movement, 3),
        "tension": round(tension, 3),
        "intensity": round(intensity, 3),
        "scale": "minor" if valence < -0.20 else "major",
        "register_shift": int(round(9 * valence + 4 * (energy - 0.5))),
        "source": "offline_italian_semantic_lexicon",
    }


def build_semantic_profile(text: str, analyses) -> dict:
    phrase_texts = []
    current = []
    for item in analyses:
        if isinstance(item, str):
            if current:
                phrase_texts.append(" ".join(current))
                current = []
        else:
            current.append(item.word)
    if current:
        phrase_texts.append(" ".join(current))

    return {
        "version": 1,
        "text": text.strip(),
        "global": analyze_semantic_profile(text),
        "phrases": [
            {"text": phrase, "profile": analyze_semantic_profile(phrase)}
            for phrase in phrase_texts
        ],
    }


def save_semantic_profile(profile: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


# ============================================================
# TESTO → ANALISI
# ============================================================

def analyze_text(
    text: str,
    analyzer: ItalianWordAnalyzer
):

    analyses = []

    tokens = tokenize_text(text)

    for token in tokens:

        # Punteggiatura
        if token in PAUSES:

            analyses.append(token)

            continue

        # Parola
        analysis = analyzer.analyze(
            token
        )

        analyses.append(
            analysis
        )

    return analyses


# ============================================================
# STAMPA ANALISI
# ============================================================

def print_analysis(
    analyses
):

    print()
    print("=" * 70)
    print("ANALISI LINGUISTICA")
    print("=" * 70)

    for item in analyses:

        if isinstance(item, str):

            print(
                f"[PAUSA {item}]"
            )

            continue

        syllables_text = []

        for syllable in item.syllables:

            if syllable.stressed:

                syllables_text.append(
                    syllable.text.upper()
                )

            else:

                syllables_text.append(
                    syllable.text
                )

        print(
            f"{item.word:15} "
            f"{'-'.join(syllables_text):25} "
            f"stress={item.stressed_index} "
            f"source={item.stress_source}"
        )


# ============================================================
# ANALISI → RITMO
# ============================================================

def build_rhythm(
    analyses
) -> list[RhythmicEvent]:

    events = []

    current_time = 0.0

    for item in analyses:

        # ----------------------------------------------------
        # PUNTEGGIATURA
        # ----------------------------------------------------

        if isinstance(item, str):

            pause = PAUSES[item]

            events.append(
                RhythmicEvent(
                    syllable=None,
                    start=current_time,
                    duration=pause,
                    velocity=0,
                    pitch=PLACEHOLDER_PITCH,
                    is_rest=True
                )
            )

            current_time += pause

            continue

        # ----------------------------------------------------
        # SILLABE
        # ----------------------------------------------------

        for syllable in item.syllables:

            if syllable.stressed:

                duration = STRESSED_DURATION
                velocity = STRESSED_VELOCITY

            else:

                duration = UNSTRESSED_DURATION
                velocity = UNSTRESSED_VELOCITY

            events.append(
                RhythmicEvent(
                    syllable=syllable,
                    start=current_time,
                    duration=duration,
                    velocity=velocity,
                    pitch=PLACEHOLDER_PITCH,
                    is_rest=False
                )
            )

            current_time += duration

    return events


# ============================================================
# STAMPA RITMO
# ============================================================

def print_rhythm(
    events: list[RhythmicEvent]
):

    print()
    print("=" * 70)
    print("SCHELETRO RITMICO")
    print("=" * 70)

    for event in events:

        if event.is_rest:

            print(
                f"{event.start:6.2f} "
                f"PAUSA "
                f"{event.duration:.2f}"
            )

            continue

        syllable = event.syllable

        accent = (
            "FORTE"
            if syllable.stressed
            else "debole"
        )

        print(
            f"{event.start:6.2f} "
            f"{syllable.text:10} "
            f"{accent:6} "
            f"dur={event.duration:.2f} "
            f"vel={event.velocity}"
        )


# ============================================================
# RITMO → MIDI
# ============================================================

def create_midi(
    events: list[RhythmicEvent],
    output_path: Path,
    target_duration_seconds: Optional[float] = None,
    bpm: float = BPM,
):

    """
    Scrive gli eventi ritmici come MIDI.

    Se `target_duration_seconds` è None (default), viene usato il
    tempo fisso `bpm` (configurato in cima al file).

    Se invece l'utente specifica una durata desiderata, ricalcoliamo
    il tempo (BPM effettivo) in modo che l'intero brano, con lo
    stesso numero di sillabe/pause già calcolato, occupi esattamente
    quella durata in secondi. Le proporzioni ritmiche restano
    identiche (una sillaba accentata dura sempre il doppio di una
    non accentata): cambia solo la velocità complessiva di
    esecuzione.
    """

    total_beats = 0.0

    if events:
        last_event = events[-1]
        total_beats = last_event.start + last_event.duration

    if target_duration_seconds is not None and total_beats > 0:

        effective_bpm = 60.0 * total_beats / target_duration_seconds

        print(
            f"\n[INFO] Durata richiesta: {target_duration_seconds:.2f}s "
            f"-> tempo ricalcolato a {effective_bpm:.1f} BPM "
            f"({total_beats:.2f} beat totali)."
        )

    else:
        effective_bpm = bpm

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=effective_bpm
    )

    # --------------------------------------------------------
    # Strumento provvisorio
    # --------------------------------------------------------

    program = (
        pretty_midi
        .instrument_name_to_program(
            "Acoustic Grand Piano"
        )
    )

    instrument = pretty_midi.Instrument(
        program=program,
        name="Rhythmic Skeleton"
    )

    seconds_per_beat = 60.0 / effective_bpm

    # --------------------------------------------------------
    # Aggiungi note
    # --------------------------------------------------------

    for event in events:

        # Le pause non diventano note MIDI
        if event.is_rest:
            continue

        start = (
            event.start
            * seconds_per_beat
        )

        end = (
            event.start
            + event.duration
        ) * seconds_per_beat

        note = pretty_midi.Note(
            velocity=event.velocity,
            pitch=event.pitch,
            start=start,
            end=end
        )

        instrument.notes.append(
            note
        )

    midi.instruments.append(
        instrument
    )

    midi.write(
        str(output_path)
    )

    total_seconds = total_beats * seconds_per_beat

    print()
    print(
        f"[OK] MIDI creato: {output_path}"
    )
    print(
        f"     Tempo: {effective_bpm:.1f} BPM | "
        f"Durata: {total_seconds:.2f}s"
    )


# ============================================================
# FUNZIONE DI DEBUG
# ============================================================

def test_words(
    analyzer: ItalianWordAnalyzer
):

    words = [
        "musica",
        "importante",
        "città",
        "perché",
        "tavolo",
        "italiano",
        "amore",
        "casa",
        "telefono"
    ]

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    for word in words:

        result = analyzer.analyze(
            word
        )

        syllables = []

        for syllable in result.syllables:

            if syllable.stressed:
                syllables.append(
                    syllable.text.upper()
                )

            else:
                syllables.append(
                    syllable.text
                )

        print(
            f"{word:15} "
            f"{'-'.join(syllables):25} "
            f"stress={result.stressed_index} "
            f"source={result.stress_source}"
        )


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Testo italiano -> scheletro ritmico MIDI"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDI",
        help=(
            "Durata totale desiderata del brano, in secondi. "
            "Se omesso, viene usato il tempo fisso configurato "
            f"(BPM={BPM})."
        ),
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Testo da mettere in musica (sovrascrive quello di default nello script).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print()
    print("=" * 70)
    print("TEXT → RHYTHM")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CARICA PHONITALIA
    # --------------------------------------------------------

    print()
    print("Caricamento phonitaliaR...")

    phonitalia_dfs = load_data_files(
        PHONITALIA_DIR
    )

    phonitalia_df = find_phonitalia_dataframe(
        phonitalia_dfs
    )

    if phonitalia_df is None:

        print()
        print(
            "[ERRORE] Non ho trovato il dataframe "
            "phonitalia."
        )

        print()
        print(
            "DataFrame trovati:"
        )

        for df in phonitalia_dfs:

            print(
                list(df.columns)
            )

        return

    phonitalia = PhonItaliaLookup(
        phonitalia_df
    )

    # --------------------------------------------------------
    # 2. CARICA Q2STRESS
    # --------------------------------------------------------

    print()
    print("Caricamento Q2Stress...")

    q2stress_dfs = load_data_files(
        Q2STRESS_DIR
    )

    q2stress = Q2StressLookup(
        q2stress_dfs
    )

    # --------------------------------------------------------
    # 3. ANALYZER
    # --------------------------------------------------------

    analyzer = ItalianWordAnalyzer(
        phonitalia=phonitalia,
        q2stress=q2stress
    )

    # --------------------------------------------------------
    # 4. TEST DATASET
    # --------------------------------------------------------

    test_words(
        analyzer
    )

    # --------------------------------------------------------
    # 5. TESTO
    # --------------------------------------------------------

    text = args.text or """
Un testo di esempio, con alcune parole accentate e punteggiatura.
Può contenere frasi brevi, frasi lunghe, esclamazioni! E interrogazioni? Tutto ciò serve a testare la pipeline di conversione da testo a MIDI."""

    print()
    print("=" * 70)
    print("TESTO")
    print("=" * 70)

    print(text)

    # --------------------------------------------------------
    # 6. TESTO → STRESS
    # --------------------------------------------------------

    analyses = analyze_text(
        text,
        analyzer
    )

    print_analysis(
        analyses
    )

    # --------------------------------------------------------
    # 7. STRESS → RITMO
    # --------------------------------------------------------

    events = build_rhythm(
        analyses
    )

    print_rhythm(
        events
    )

    # --------------------------------------------------------
    # 8. TESTO → PROFILO SEMANTICO
    # --------------------------------------------------------

    semantic_profile = build_semantic_profile(text, analyses)
    save_semantic_profile(semantic_profile, SEMANTIC_PROFILE_JSON)

    print()
    print("PROFILO SEMANTICO")
    print("=" * 70)
    gp = semantic_profile["global"]
    print(
        f"valence={gp['valence']:+.2f} | energy={gp['energy']:.2f} | "
        f"movement={gp['movement']:.2f} | tension={gp['tension']:.2f} | "
        f"scale={gp['scale']}"
    )
    print(f"[OK] Profilo semantico salvato in: {SEMANTIC_PROFILE_JSON}")

    # --------------------------------------------------------
    # 9. RITMO → MIDI
    # --------------------------------------------------------

    create_midi(
        events,
        OUTPUT_MIDI,
        target_duration_seconds=args.duration,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()