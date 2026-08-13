import pandas as pd
import spacy
import mido

from mido import MidiFile, MidiTrack, Message, MetaMessage

import re
import math


# ============================================================
# CONFIGURAZIONE
# ============================================================

TICKS_PER_BEAT = 480
BPM = 105

# Canale MIDI 10 = percussioni
PERCUSSION_CHANNEL = 9


# ============================================================
# CARICAMENTO SPACY
# ============================================================

try:
    nlp = spacy.load("it_core_news_sm")

except OSError:
    print()
    print("ERRORE: modello italiano di spaCy non trovato.")
    print()
    print("Installa con:")
    print()
    print("python -m spacy download it_core_news_sm")
    print()
    raise


# ============================================================
# GENERAL MIDI - PERCUSSIONI
# ============================================================

KICK = 36
SNARE = 38
CLOSED_HIHAT = 42
OPEN_HIHAT = 46
LOW_TOM = 45
HIGH_TOM = 50
CRASH = 49


# ============================================================
# SCALA
# ============================================================

C_MAJOR = [
    60,  # C4
    62,  # D4
    64,  # E4
    65,  # F4
    67,  # G4
    69,  # A4
    71,  # B4
    72,  # C5
]


# ============================================================
# ACCENTI
# ============================================================

ACCENT_POSITIONS = {
    "ultima sillaba": 0,
    "penultima sillaba": 1,
    "antepenultima sillaba": 2,
    "preantepenultima sillaba": 3,
}


# ============================================================
# STRUMENTI GENERAL MIDI
# ============================================================

INSTRUMENTS = {

    1: ("Pianoforte", 0),

    2: ("Pianoforte elettrico", 4),

    3: ("Chitarra acustica", 24),

    4: ("Basso", 32),

    5: ("Violino", 40),

    6: ("Violoncello", 42),

    7: ("Flauto", 73),

    8: ("Clarinetto", 71),

    9: ("Sax", 65),

    10: ("Tromba", 56),

    11: ("Vibrafono", 11),

}


# ============================================================
# UTILITÀ
# ============================================================

def safe_float(value):

    try:

        if pd.isna(value):
            return None

        return float(value)

    except:

        return None


def safe_bool(value):

    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    return str(value).lower() in [
        "true",
        "1",
        "yes",
    ]


def accent_index(label):

    if label is None:
        return 0

    return ACCENT_POSITIONS.get(
        str(label),
        0
    )


# ============================================================
# STIMA SILLABE
# ============================================================

def estimate_syllables(word):

    """
    Stima molto semplice del numero di sillabe.

    In futuro possiamo sostituirla con un
    sillabificatore italiano/fonetico vero.
    """

    word = word.lower()

    vowels = "aeiouàèéìòóù"

    count = 0
    previous_vowel = False

    for char in word:

        if char in vowels:

            if not previous_vowel:
                count += 1

            previous_vowel = True

        else:

            previous_vowel = False

    return max(count, 1)


# ============================================================
# WORD EVENT
# ============================================================

class WordEvent:

    def __init__(
        self,
        word,
        model_label,
        model_position,
        confidence,
        phon_label,
        phon_position,
        agreement,
        token,
        index,
        total_words,
    ):

        # ----------------------------------------------------
        # Informazioni originali
        # ----------------------------------------------------

        self.word = word

        self.model_label = model_label

        self.model_position = model_position

        self.confidence = confidence

        self.phon_label = phon_label

        self.phon_position = phon_position

        self.agreement = agreement

        # ----------------------------------------------------
        # Informazioni spaCy
        # ----------------------------------------------------

        self.lemma = token.lemma_

        self.pos = token.pos_

        self.tag = token.tag_

        self.dep = token.dep_

        self.head = token.head.text

        self.head_lemma = token.head.lemma_

        self.morphology = str(
            token.morph
        )

        # ----------------------------------------------------
        # Posizione nella frase
        # ----------------------------------------------------

        self.index = index

        self.total_words = total_words

        self.is_first = index == 0

        self.is_last = (
            index == total_words - 1
        )

        # ----------------------------------------------------
        # Informazioni lessicali
        # ----------------------------------------------------

        self.word_length = len(word)

        self.syllables = estimate_syllables(
            word
        )

        # ----------------------------------------------------
        # Punteggiatura
        # ----------------------------------------------------

        self.punctuation_before = ""

        self.punctuation_after = ""

        # ----------------------------------------------------
        # Informazioni musicali
        # ----------------------------------------------------

        self.rhythmic_weight = 1.0

        self.melodic_tension = 0.0

        self.duration = TICKS_PER_BEAT

        self.model_note = 60

        self.phon_note = 60

        self.drum_note = CLOSED_HIHAT

        # ----------------------------------------------------
        # Contesto sintattico
        # ----------------------------------------------------

        self.syntactic_role = self.get_syntactic_role()

        self.is_root = (
            self.dep == "ROOT"
        )

        self.is_subject = (
            self.dep in [
                "nsubj",
                "nsubj:pass"
            ]
        )

        self.is_object = (
            self.dep in [
                "obj",
                "iobj"
            ]
        )

        self.is_modifier = (
            self.dep in [
                "amod",
                "advmod",
                "nmod"
            ]
        )

        # ----------------------------------------------------
        # Contesto POS
        # ----------------------------------------------------

        self.is_noun = self.pos == "NOUN"

        self.is_verb = self.pos in [
            "VERB",
            "AUX"
        ]

        self.is_adjective = (
            self.pos == "ADJ"
        )

        self.is_adverb = (
            self.pos == "ADV"
        )

        self.is_pronoun = (
            self.pos == "PRON"
        )


    # ========================================================
    # RUOLO SINTATTICO
    # ========================================================

    def get_syntactic_role(self):

        mapping = {

            "ROOT":
                "radice",

            "nsubj":
                "soggetto",

            "nsubj:pass":
                "soggetto",

            "obj":
                "oggetto",

            "iobj":
                "oggetto_indiretto",

            "amod":
                "modificatore",

            "advmod":
                "avverbio",

            "nmod":
                "complemento",

            "obl":
                "complemento",

            "conj":
                "coordinazione",

            "cc":
                "congiunzione",

            "det":
                "determinante",

            "punct":
                "punteggiatura",

        }

        return mapping.get(
            self.dep,
            self.dep
        )


# ============================================================
# ESTRAZIONE TOKEN SPAcy
# ============================================================

def analyze_sentence(sentence):

    """
    Analizza una frase con spaCy.
    """

    doc = nlp(
        sentence
    )

    tokens = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_punct:
            continue

        tokens.append(token)

    return doc, tokens


# ============================================================
# CONFRONTO DATASET / SPACY
# ============================================================

def create_events_from_sentence(
    sentence,
    dataframe
):

    """
    Associa ogni parola del dataset
    al relativo token spaCy.

    Il caso ideale è:

        dataset.word == token.text
    """

    doc, tokens = analyze_sentence(
        sentence
    )

    events = []

    total = min(
        len(tokens),
        len(dataframe)
    )

    for i in range(total):

        token = tokens[i]

        row = dataframe.iloc[i]

        event = WordEvent(

            word=token.text,

            model_label=row.get(
                "model_label"
            ),

            model_position=safe_float(
                row.get(
                    "model_position"
                )
            ),

            confidence=safe_float(
                row.get(
                    "confidence"
                )
            ),

            phon_label=(
                row.get(
                    "phon_label"
                )
                if not pd.isna(
                    row.get(
                        "phon_label"
                    )
                )
                else None
            ),

            phon_position=safe_float(
                row.get(
                    "phon_position"
                )
            ),

            agreement=safe_bool(
                row.get(
                    "agreement"
                )
            ),

            token=token,

            index=i,

            total_words=total,
        )

        events.append(
            event
        )

    return events


# ============================================================
# CONTESTO FONOLOGICO
# ============================================================

def analyze_phonology(event):

    accent = accent_index(
        event.model_label
    )

    event.rhythmic_weight = {

        0: 1.30,

        1: 1.15,

        2: 1.00,

        3: 0.85,

    }.get(
        accent,
        1.0
    )

    # Errore del modello
    if event.agreement is False:

        event.melodic_tension += 0.35

    # Confidence
    if event.confidence is not None:

        event.melodic_tension += (
            1.0 - event.confidence
        ) * 0.25


# ============================================================
# CONTESTO LESSICALE
# ============================================================

def analyze_lexical(event):

    syllables = event.syllables

    if syllables <= 1:

        event.duration = (
            TICKS_PER_BEAT
        )

    elif syllables == 2:

        event.duration = (
            TICKS_PER_BEAT
        )

    elif syllables == 3:

        event.duration = int(
            TICKS_PER_BEAT * 1.25
        )

    else:

        event.duration = int(
            TICKS_PER_BEAT * 1.5
        )


# ============================================================
# CONTESTO SINTATTICO
# ============================================================

def analyze_syntax(event):

    # --------------------------------------------------------
    # Verbo = movimento
    # --------------------------------------------------------

    if event.is_verb:

        event.melodic_tension += 0.20

        event.rhythmic_weight *= 1.10


    # --------------------------------------------------------
    # Soggetto = stabilità
    # --------------------------------------------------------

    if event.is_subject:

        event.melodic_tension -= 0.10


    # --------------------------------------------------------
    # Oggetto = sviluppo
    # --------------------------------------------------------

    if event.is_object:

        event.melodic_tension += 0.10


    # --------------------------------------------------------
    # Root = elemento strutturale
    # --------------------------------------------------------

    if event.is_root:

        event.rhythmic_weight *= 1.20


    # --------------------------------------------------------
    # Modificatori = decorazione
    # --------------------------------------------------------

    if event.is_modifier:

        event.melodic_tension += 0.05


    # --------------------------------------------------------
    # Parti del discorso
    # --------------------------------------------------------

    if event.is_adjective:

        event.melodic_tension += 0.05

    if event.is_adverb:

        event.melodic_tension += 0.08


# ============================================================
# CONTESTO PROSODICO
# ============================================================

def analyze_prosody(event):

    # Prima parola
    if event.is_first:

        event.melodic_tension -= 0.15


    # Ultima parola
    if event.is_last:

        event.melodic_tension -= 0.30

        event.duration = int(
            event.duration * 1.5
        )


# ============================================================
# CONTESTO LOCALE
# ============================================================

def analyze_local_context(events):

    for i, event in enumerate(events):

        current = accent_index(
            event.model_label
        )

        # --------------------------------------------
        # Parola precedente
        # --------------------------------------------

        if i > 0:

            previous = accent_index(
                events[i - 1].model_label
            )

            difference = (
                current - previous
            )

            if difference > 0:

                event.melodic_tension += 0.10

            elif difference < 0:

                event.melodic_tension -= 0.10


        # --------------------------------------------
        # Parola successiva
        # --------------------------------------------

        if i < len(events) - 1:

            following = accent_index(
                events[i + 1].model_label
            )

            if following > current:

                event.melodic_tension += 0.05


# ============================================================
# STATO MUSICALE GLOBALE
# ============================================================

class MusicalState:

    def __init__(self):

        self.previous_note = 60

        self.tension = 0.0

        self.phrase_position = 0


    def update(self, event):

        self.tension += (
            event.melodic_tension
        )

        # Decadimento della tensione
        self.tension *= 0.85

        self.tension = max(
            -1.0,
            min(1.0, self.tension)
        )

        self.phrase_position += 1


# ============================================================
# GENERAZIONE NOTA MODELLO
# ============================================================

def generate_model_note(
    event,
    state
):

    accent = accent_index(
        event.model_label
    )

    # ----------------------------------------
    # Base accentuale
    # ----------------------------------------

    degree = {

        0: 0,

        1: 2,

        2: 4,

        3: 7,

    }.get(
        accent,
        0
    )


    # ----------------------------------------
    # Numero sillabe
    # ----------------------------------------

    degree += (
        event.syllables - 1
    )


    # ----------------------------------------
    # POS
    # ----------------------------------------

    if event.is_verb:

        degree += 1

    elif event.is_subject:

        degree -= 1

    elif event.is_object:

        degree += 2


    # ----------------------------------------
    # Sintassi
    # ----------------------------------------

    if event.is_root:

        degree += 1


    # ----------------------------------------
    # Tensione globale
    # ----------------------------------------

    degree += int(
        state.tension * 2
    )


    degree %= len(C_MAJOR)

    note = C_MAJOR[
        degree
    ]


    # ----------------------------------------
    # Evita salti enormi
    # ----------------------------------------

    difference = (
        note -
        state.previous_note
    )

    if difference > 7:

        note -= 12

    elif difference < -7:

        note += 12


    # ----------------------------------------
    # Stato
    # ----------------------------------------

    state.previous_note = note

    return note


# ============================================================
# GENERAZIONE NOTA FONETICA
# ============================================================

def generate_phon_note(
    event,
    state
):

    if event.phon_label is None:

        return state.previous_note


    accent = accent_index(
        event.phon_label
    )


    degree = {

        0: 0,

        1: 2,

        2: 4,

        3: 7,

    }.get(
        accent,
        0
    )


    # La sintassi influenza anche
    # la seconda voce

    if event.is_verb:

        degree += 1

    elif event.is_object:

        degree += 2


    degree %= len(C_MAJOR)

    note = C_MAJOR[
        degree
    ]


    return note


# ============================================================
# PERCUSSIONI
# ============================================================

def generate_drum(event):

    accent = accent_index(
        event.model_label
    )

    # Errore modello
    if event.agreement is False:

        if accent == 0:

            return LOW_TOM

        return HIGH_TOM


    # Sintassi
    if event.is_verb:

        return SNARE


    if event.is_subject:

        return CLOSED_HIHAT


    if event.is_object:

        return HIGH_TOM


    # Accento
    if accent == 0:

        return KICK

    if accent == 1:

        return SNARE

    if accent == 2:

        return CLOSED_HIHAT

    return OPEN_HIHAT


# ============================================================
# ANALISI COMPLETA
# ============================================================

def analyze_events(events):

    # ----------------------------------------
    # Contesto individuale
    # ----------------------------------------

    for event in events:

        analyze_phonology(event)

        analyze_lexical(event)

        analyze_syntax(event)

        analyze_prosody(event)

        event.drum_note = generate_drum(
            event
        )


    # ----------------------------------------
    # Contesto tra parole
    # ----------------------------------------

    analyze_local_context(
        events
    )


    return events


# ============================================================
# TRACCIA PERCUSSIONI
# ============================================================

def create_percussion_track(
    events
):

    track = MidiTrack()

    track.append(
        MetaMessage(
            "track_name",
            name="Percussioni"
        )
    )


    for event in events:

        velocity = int(
            70 +
            40 *
            min(
                event.rhythmic_weight,
                1.4
            )
        )

        velocity = max(
            1,
            min(
                127,
                velocity
            )
        )


        track.append(
            Message(
                "note_on",
                channel=PERCUSSION_CHANNEL,
                note=event.drum_note,
                velocity=velocity,
                time=0
            )
        )


        track.append(
            Message(
                "note_off",
                channel=PERCUSSION_CHANNEL,
                note=event.drum_note,
                velocity=0,
                time=event.duration
            )
        )


    return track


# ============================================================
# TRACCIA MODELLO
# ============================================================

def create_model_track(
    events,
    instrument,
    channel
):

    track = MidiTrack()

    track.append(
        MetaMessage(
            "track_name",
            name="Modello - Sintassi + Accento"
        )
    )


    track.append(
        Message(
            "program_change",
            channel=channel,
            program=instrument,
            time=0
        )
    )


    state = MusicalState()


    for event in events:

        note = generate_model_note(
            event,
            state
        )


        # Confidence
        if event.confidence is None:

            velocity = 80

        else:

            velocity = int(
                45 +
                event.confidence * 80
            )


        # Agreement
        if event.agreement is False:

            velocity -= 20


        velocity = max(
            20,
            min(
                127,
                velocity
            )
        )


        track.append(
            Message(
                "note_on",
                channel=channel,
                note=note,
                velocity=velocity,
                time=0
            )
        )


        track.append(
            Message(
                "note_off",
                channel=channel,
                note=note,
                velocity=0,
                time=event.duration
            )
        )


        state.update(
            event
        )


    return track


# ============================================================
# TRACCIA FONETICA
# ============================================================

def create_phonetic_track(
    events,
    instrument,
    channel
):

    track = MidiTrack()

    track.append(
        MetaMessage(
            "track_name",
            name="Fonetica"
        )
    )


    track.append(
        Message(
            "program_change",
            channel=channel,
            program=instrument,
            time=0
        )
    )


    state = MusicalState()


    for event in events:

        note = generate_phon_note(
            event,
            state
        )


        if event.agreement is True:

            velocity = 90

        elif event.agreement is False:

            velocity = 60

        else:

            velocity = 45


        track.append(
            Message(
                "note_on",
                channel=channel,
                note=note,
                velocity=velocity,
                time=0
            )
        )


        track.append(
            Message(
                "note_off",
                channel=channel,
                note=note,
                velocity=0,
                time=event.duration
            )
        )


        state.update(
            event
        )


    return track


# ============================================================
# CREAZIONE MIDI
# ============================================================

def create_midi(
    events,
    selected_instruments
):

    midi = MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    # ----------------------------------------
    # Tempo
    # ----------------------------------------

    tempo_track = MidiTrack()

    tempo_track.append(
        MetaMessage(
            "track_name",
            name="Tempo"
        )
    )

    tempo_track.append(
        MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(BPM)
        )
    )

    midi.tracks.append(
        tempo_track
    )


    # ----------------------------------------
    # Percussioni
    # ----------------------------------------

    midi.tracks.append(
        create_percussion_track(
            events
        )
    )


    # ----------------------------------------
    # Strumenti
    # ----------------------------------------

    for i, instrument in enumerate(
        selected_instruments
    ):

        channel = i

        # Il canale 9 è riservato alle percussioni
        if channel >= 9:

            channel += 1


        # Primo strumento = modello
        if i == 0:

            midi.tracks.append(
                create_model_track(
                    events,
                    instrument,
                    channel
                )
            )


        # Secondo strumento = fonetica
        elif i == 1:

            midi.tracks.append(
                create_phonetic_track(
                    events,
                    instrument,
                    channel
                )
            )


        # Altri strumenti = modello
        else:

            midi.tracks.append(
                create_model_track(
                    events,
                    instrument,
                    channel
                )
            )


    return midi


# ============================================================
# MENU STRUMENTI
# ============================================================

def choose_instruments():

    print()
    print("=" * 50)
    print("STRUMENTI")
    print("=" * 50)

    for number, (name, program) in INSTRUMENTS.items():

        print(
            f"{number:2d}. {name}"
        )

    print()
    print(
        "Scegli uno o più strumenti."
    )
    print(
        "Esempio: 1,5,7"
    )
    print()

    choice = input("> ")

    selected = []

    for item in choice.split(","):

        try:

            number = int(
                item.strip()
            )

            if number in INSTRUMENTS:

                selected.append(
                    INSTRUMENTS[number][1]
                )

        except ValueError:

            pass


    return selected


# ============================================================
# STAMPA ANALISI
# ============================================================

def print_analysis(events):

    print()
    print("=" * 100)
    print("ANALISI LINGUISTICA")
    print("=" * 100)

    for event in events:

        print(
            f"{event.word:15s} "
            f"lemma={event.lemma:12s} "
            f"POS={event.pos:6s} "
            f"dep={event.dep:10s} "
            f"ruolo={event.syntactic_role:20s} "
            f"accento={event.model_label}"
        )


# ============================================================
# DATASET
# ============================================================

def load_dataset(filename):

    df = pd.read_csv(
        filename
    )

    required_columns = [

        "word",

        "model_position",

        "model_label",

        "confidence",

        "phon_position",

        "phon_label",

        "agreement",

    ]


    missing = [

        c

        for c in required_columns

        if c not in df.columns

    ]


    if missing:

        raise ValueError(
            "Colonne mancanti: "
            +
            ", ".join(missing)
        )


    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TEXT → NLP → CONTEXT → MUSIC")
    print("=" * 70)


    # ----------------------------------------
    # CSV
    # ----------------------------------------

    filename = input(
        "\nNome del CSV: "
    )


    try:

        df = load_dataset(
            filename
        )

    except Exception as error:

        print()
        print(
            "Errore:",
            error
        )

        return


    print(
        f"\nRighe del dataset: {len(df)}"
    )


    # ----------------------------------------
    # FRASE
    # ----------------------------------------

    if "sentence" in df.columns:

        sentence = str(
            df.iloc[0]["sentence"]
        )

        print(
            "\nFrase trovata nel dataset:"
        )

        print(
            sentence
        )

    else:

        print()
        print(
            "Il CSV non contiene la colonna 'sentence'."
        )

        print(
            "Inserisci la frase corrispondente:"
        )

        sentence = input(
            "> "
        )


    # ----------------------------------------
    # Controllo parole
    # ----------------------------------------

    doc, tokens = analyze_sentence(
        sentence
    )


    print()
    print(
        f"Parole riconosciute da spaCy: {len(tokens)}"
    )

    print(
        f"Righe nel dataset: {len(df)}"
    )


    if len(tokens) != len(df):

        print()
        print(
            "ATTENZIONE:"
        )

        print(
            "Il numero di parole della frase "
            "non coincide con le righe del dataset."
        )

        print(
            "Il programma utilizzerà "
            "il numero minore."
        )


    # ----------------------------------------
    # Creazione eventi
    # ----------------------------------------

    events = create_events_from_sentence(
        sentence,
        df
    )


    # ----------------------------------------
    # Analisi contesti
    # ----------------------------------------

    events = analyze_events(
        events
    )


    # ----------------------------------------
    # Visualizzazione
    # ----------------------------------------

    print_analysis(
        events
    )


    # ----------------------------------------
    # Strumenti
    # ----------------------------------------

    selected = choose_instruments()


    if not selected:

        print(
            "\nNessuno strumento selezionato."
        )

        return


    # ----------------------------------------
    # MIDI
    # ----------------------------------------

    midi = create_midi(
        events,
        selected
    )


    output = (
        "testo_musicale_context.mid"
    )


    midi.save(
        output
    )


    print()
    print("=" * 70)

    print(
        "MIDI creato:"
    )

    print(
        output
    )

    print("=" * 70)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":

    main()