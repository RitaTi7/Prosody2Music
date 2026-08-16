from __future__ import annotations

from pathlib import Path
import argparse
import random
import json
import pretty_midi

# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "output"

RHYTHMIC_MIDI = INPUT_DIR / "rhythmic_skeleton.mid"
FINAL_MELODY_MIDI = INPUT_DIR / "melody.mid"
SEMANTIC_PROFILE_JSON = INPUT_DIR / "semantic_profile.json"

# Impostazioni melodia
START_PITCH = 60  # C4 (Do centrale)
SCALE = [0, 2, 4, 5, 7, 9, 11]  # Intervalli della scala Maggiore (Do maggiore)
# Se preferisci una scala minore, usa: [0, 2, 3, 5, 7, 8, 10]

# Impostazioni arrangiamento
#
# L'arrangiamento è definito da due scelte indipendenti che
# l'utente può combinare liberamente da riga di comando:
#
#   --genre   sceglie le timbriche (chi suona)
#   --rhythm  sceglie quanto la musica è marcata/ritmata (come suona)
#
ADD_ACCOMPANIMENT = True

DEFAULT_GENRE = "orchestrale"
DEFAULT_RHYTHM = "moderato"

# ------------------------------------------------------------
# GENERI: definiscono gli strumenti usati.
# ------------------------------------------------------------
#
# melody / bass / pad: strumenti principali (nomi General MIDI).
# horn, timpani: elementi extra tipici del genere orchestrale
#   (assenti negli altri generi).
# percussion_kit: "drumkit" per una vera batteria (kick/snare/hihat),
#   oppure il numero di nota GM di una singola percussione intonata
#   allo stile del genere (es. tamburello per il vocale), o None se
#   il genere non usa percussione "da groove" (l'orchestrale la
#   sostituisce con i timpani).
GENRE_PRESETS = {
    "orchestrale": {
        "description": "Fiato solista + archi, cinematico",
        "melody": "Oboe",
        "bass": "Contrabass",
        "pad": "String Ensemble 1",
        "horn": "French Horn",
        "timpani": "Timpani",
        "percussion_kit": None,
    },
    "vocale": {
        "description": "Coro/voce sintetica in primo piano",
        "melody": "Choir Aahs",
        "bass": "Fretless Bass",
        "pad": "Pad 4 (choir)",
        "horn": None,
        "timpani": None,
        "percussion_kit": 54,  # Tambourine: leggero, adatto a un accompagnamento vocale
    },
    "acustico": {
        "description": "Intimo, chitarra e legni",
        "melody": "Flute",
        "bass": "Acoustic Bass",
        "pad": "Acoustic Guitar (nylon)",
        "horn": None,
        "timpani": None,
        "percussion_kit": 70,  # Maracas: texture organica, non invadente
    },
    "pop": {
        "description": "Moderno, elettrico, più ritmato",
        "melody": "Electric Piano 1",
        "bass": "Electric Bass (finger)",
        "pad": "Synth Strings 1",
        "horn": None,
        "timpani": None,
        "percussion_kit": "drumkit",  # vera batteria: cassa/rullante/hi-hat
    },
}

# ------------------------------------------------------------
# LIVELLI RITMICI: definiscono quanto l'arrangiamento è marcato.
# ------------------------------------------------------------
#
# percussion_density: "nessuna" | "leggera" (solo sillabe accentate)
#   | "piena" (ogni sillaba, con accenti più forti su quelle toniche).
# staccato: quanto accorciare le note della melodia (0 = legato pieno,
#   0.2 = molto staccato/puntato).
# bass_pattern: "sostenuto" (un'unica nota lunga per frase, come nel
#   preset orchestrale iniziale) oppure "walking" (il basso si muove
#   su ogni sillaba accentata, alternando fondamentale e quinta, per
#   una sensazione più ritmica e propulsiva).
# timpani_extra: se True, oltre al colpo a inizio frase i timpani
#   accentano anche le sillabe toniche (solo rilevante nel genere
#   orchestrale, dove non c'è una percussion_kit "da groove").
RHYTHM_PRESETS = {
    "libero": {
        "description": "Rubato, quasi senza pulsazione",
        "percussion_density": "nessuna",
        "staccato": 0.0,
        "bass_pattern": "sostenuto",
        "timpani_extra": False,
    },
    "moderato": {
        "description": "Pulsazione presente ma leggera",
        "percussion_density": "leggera",
        "staccato": 0.05,
        "bass_pattern": "sostenuto",
        "timpani_extra": False,
    },
    "marcato": {
        "description": "Molto ritmato, groove in evidenza",
        "percussion_density": "piena",
        "staccato": 0.20,
        "bass_pattern": "walking",
        "timpani_extra": True,
    },
}

# Soglia di velocity sopra la quale una sillaba è considerata
# accentata: coerente con STRESSED_VELOCITY (110) / UNSTRESSED_VELOCITY
# (55) definiti in DaTestoAMIDI.py.
STRESS_VELOCITY_THRESHOLD = 80

# Giro armonico diatonico usato per il pad/basso: I - IV - V - I,
# espresso come gradi 0-based della scala (0=I, 3=IV, 4=V).
# Un accordo per ogni frase individuata nel testo (vedi is_phrase_end).
CHORD_PROGRESSION_DEGREES = [0, 3, 4, 0]


def resolve_arrangement(genre: str, rhythm: str) -> dict:
    """
    Combina un genere e un livello ritmico nell'insieme completo di
    impostazioni usato per costruire l'arrangiamento. Le due scelte
    sono indipendenti: qualunque genere può essere reso più o meno
    ritmato.
    """
    if genre not in GENRE_PRESETS:
        raise ValueError(
            f"Genere sconosciuto: {genre!r}. "
            f"Disponibili: {', '.join(GENRE_PRESETS)}"
        )

    if rhythm not in RHYTHM_PRESETS:
        raise ValueError(
            f"Livello ritmico sconosciuto: {rhythm!r}. "
            f"Disponibili: {', '.join(RHYTHM_PRESETS)}"
        )

    arrangement = {**GENRE_PRESETS[genre], **RHYTHM_PRESETS[rhythm]}
    arrangement["genre"] = genre
    arrangement["rhythm"] = rhythm
    return arrangement

# ============================================================
# LETTURA DELLO SCHELETRO
# ============================================================


# Se tra la fine di una nota e l'inizio della successiva c'è un
# vuoto maggiore di questa soglia (in secondi), lo interpretiamo
# come una pausa dovuta alla punteggiatura (create_midi, nella
# prima fase, non scrive le pause come note: restano solo come
# "buchi" nel tempo). Usiamo questo segnale per marcare la fine
# di una frase musicale.
PHRASE_GAP_THRESHOLD = 0.05  # secondi


def load_rhythmic_skeleton(midi_path: Path):
    """
    Legge il MIDI prodotto dalla prima fase.
    Restituisce: [(start, duration, velocity, is_phrase_end, is_stressed), ...]

    is_phrase_end è True quando dopo questa nota il testo
    originale aveva una pausa di punteggiatura (virgola, punto,
    ecc.): la ricostruiamo dal vuoto temporale lasciato nel MIDI,
    dato che le pause non vengono scritte come note.

    is_stressed è True per le sillabe accentate, dedotto dalla
    velocity (vedi STRESS_VELOCITY_THRESHOLD): serve a decidere
    dove mettere gli accenti ritmici (basso "walking",
    percussioni, timpani) senza dover rileggere il testo originale.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.end - note.start, note.velocity))

    notes.sort(key=lambda x: x[0])

    enriched = []
    for i, (start, duration, velocity) in enumerate(notes):
        is_phrase_end = False

        if i + 1 < len(notes):
            next_start = notes[i + 1][0]
            gap = next_start - (start + duration)
            is_phrase_end = gap > PHRASE_GAP_THRESHOLD
        else:
            # l'ultima nota del testo è sempre una fine di frase
            is_phrase_end = True

        is_stressed = velocity > STRESS_VELOCITY_THRESHOLD

        enriched.append((start, duration, velocity, is_phrase_end, is_stressed))

    return enriched

# ============================================================
# GENERAZIONE MELODICA (Alternativa a Magenta)
# ============================================================

def _nearest_stable_index(allowed_pitches, current_idx, start_pitch):
    """
    Trova, tra le altezze consentite, l'indice del grado stabile
    (tonica o dominante, in una qualunque ottava) più vicino
    all'indice corrente. Usato per far "atterrare" la melodia su
    una nota di riposo a fine frase, un po' come una cadenza.
    """
    stable_offsets = {0, 7}  # tonica e dominante (5° grado), in semitoni

    stable_indices = [
        idx
        for idx, pitch in enumerate(allowed_pitches)
        if (pitch - start_pitch) % 12 in stable_offsets
    ]

    if not stable_indices:
        return current_idx

    return min(stable_indices, key=lambda idx: abs(idx - current_idx))


def _build_allowed_pitches(center_pitch: int, scale: list[int]):
    """Costruisce tre ottave della scala attorno al registro desiderato."""
    allowed = []
    base_octave = (center_pitch // 12) * 12
    for octave in [-1, 0, 1, 2]:
        for offset in scale:
            allowed.append(base_octave + offset + 12 * octave)
    return sorted(set(p for p in allowed if 0 <= p <= 127))


def _closest_index(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def _profile_to_scale(profile: dict) -> list[int]:
    return [0, 2, 3, 5, 7, 8, 10] if profile.get("scale") == "minor" else SCALE


def _profile_step_weights(profile: dict):
    """Converte movimento/energia/valenza in una distribuzione di passi."""
    movement = float(profile.get("movement", 0.0))
    energy = float(profile.get("energy", 0.0))
    valence = float(profile.get("valence", 0.0))

    # Movimento basso = note congiunte/ripetute; movimento alto = salti.
    repeat = max(4.0, 30.0 - 22.0 * movement)
    small = 40.0 - 18.0 * movement
    jump = 4.0 + 20.0 * movement + 8.0 * energy
    down = small
    up = small

    # La valenza dà una direzione morbida: positiva tende a salire,
    # negativa a scendere, senza trasformare la melodia in una scala.
    direction = 10.0 * abs(valence) + 6.0 * energy
    if valence > 0.15:
        up += direction
        down = max(5.0, down - direction * 0.45)
    elif valence < -0.15:
        down += direction
        up = max(5.0, up - direction * 0.45)

    return [jump * 0.65, down, repeat, up, jump]


def generate_algorithmic_pitches(
    num_notes: int,
    start_pitch: int = 60,
    phrase_ends: list[bool] | None = None,
    seed: int | None = None,
    semantic_profiles: list[dict] | None = None,
):
    """Genera altezze condizionate sia dalla prosodia sia dal significato.

    Ogni frase può avere scala, registro, movimento, energia e direzione
    differenti. Se il profilo semantico non è disponibile, il comportamento
    resta compatibile con il Random Walk precedente.
    """
    rng = random.Random(seed)
    if phrase_ends is None:
        phrase_ends = [False] * num_notes
    if semantic_profiles is None or not semantic_profiles:
        semantic_profiles = [{
            "valence": 0.0, "energy": 0.35, "movement": 0.0,
            "tension": 0.25, "intensity": 0.0, "scale": "major",
            "register_shift": 0,
        }]

    # Costruiamo gli intervalli di frase direttamente dai marcatori già
    # ricavati dalle pause di punteggiatura.
    phrase_ranges = []
    start = 0
    for i in range(num_notes):
        if i < len(phrase_ends) and phrase_ends[i]:
            phrase_ranges.append((start, i + 1))
            start = i + 1
    if start < num_notes:
        phrase_ranges.append((start, num_notes))

    pitches = [start_pitch] * num_notes
    previous_pitch = start_pitch

    for phrase_idx, (phrase_start, phrase_end) in enumerate(phrase_ranges):
        profile = semantic_profiles[min(phrase_idx, len(semantic_profiles) - 1)]
        scale = _profile_to_scale(profile)
        register_shift = int(profile.get("register_shift", 0))
        center_pitch = max(36, min(84, start_pitch + register_shift))
        allowed_pitches = _build_allowed_pitches(center_pitch, scale)

        # Avvio vicino al registro precedente, ma attratto verso il registro
        # emotivo della frase.
        target_pitch = int(round(0.65 * center_pitch + 0.35 * previous_pitch))
        current_idx = _closest_index(allowed_pitches, target_pitch)
        weights = _profile_step_weights(profile)
        movement = float(profile.get("movement", 0.0))
        tension = float(profile.get("tension", 0.0))

        for i in range(phrase_start, phrase_end):
            if i < len(phrase_ends) and phrase_ends[i]:
                # La tensione tende a lasciare dominante/nota stabile;
                # le frasi molto tese possono evitare una risoluzione completa.
                stable_offsets = {0, 7}
                stable = [
                    idx for idx, pitch in enumerate(allowed_pitches)
                    if (pitch - start_pitch) % 12 in stable_offsets
                ]
                if stable:
                    if tension < 0.65:
                        current_idx = min(stable, key=lambda idx: abs(idx - current_idx))
                    else:
                        # Tensione alta: avvicinamento alla stabilità, non piena risoluzione.
                        current_idx = min(stable, key=lambda idx: abs(idx - current_idx))
                        if rng.random() < tension * 0.45:
                            current_idx = max(0, min(len(allowed_pitches) - 1, current_idx + rng.choice([-1, 1])))

            pitches[i] = allowed_pitches[current_idx]
            step = rng.choices([-2, -1, 0, 1, 2], weights=weights)[0]
            current_idx = max(0, min(len(allowed_pitches) - 1, current_idx + step))

        previous_pitch = pitches[phrase_end - 1]

    return pitches


# ============================================================
# ARMONIA / ACCOMPAGNAMENTO
# ============================================================

def scale_degree_pitch(degree: int, base_pitch: int, scale=SCALE) -> int:
    """
    Converte un grado della scala (anche oltre la singola ottava,
    es. grado 9) nell'altezza MIDI assoluta corrispondente,
    a partire da `base_pitch`.

    Esempio con scala maggiore: degree=0 -> tonica, degree=2 ->
    terza, degree=4 -> quinta (i tre gradi di una triade
    diatonica costruita per terze).
    """
    octave_shift, step = divmod(degree, len(scale))
    return base_pitch + 12 * octave_shift + scale[step]


def build_phrases(rhythmic_notes) -> list[list[int]]:
    """
    Raggruppa gli indici di `rhythmic_notes` in frasi, usando i
    marcatori `is_phrase_end` prodotti da `load_rhythmic_skeleton`
    (che a loro volta derivano dalle pause di punteggiatura del
    testo originale). Ogni frase diventa più avanti un accordo.
    """
    phrases = []
    current: list[int] = []

    for i, note in enumerate(rhythmic_notes):
        current.append(i)

        if note[3]:  # is_phrase_end
            phrases.append(current)
            current = []

    if current:
        phrases.append(current)

    return phrases


def build_accompaniment_instruments(
    rhythmic_notes,
    arrangement: dict,
    start_pitch: int = START_PITCH,
    scale=SCALE,
):
    """
    Costruisce basso, pad armonico e (a seconda del genere/livello
    ritmico scelti) corno, timpani e percussione per accompagnare
    la melodia.

    Un accordo diatonico (radice + terza + quinta) viene assegnato
    a ciascuna frase individuata nel testo, seguendo il giro
    armonico definito in CHORD_PROGRESSION_DEGREES: in questo modo
    l'armonia cambia in corrispondenza delle pause di punteggiatura,
    proprio come la cadenza melodica.
    """
    phrases = build_phrases(rhythmic_notes)

    bass_program = pretty_midi.instrument_name_to_program(arrangement["bass"])
    pad_program = pretty_midi.instrument_name_to_program(arrangement["pad"])

    bass = pretty_midi.Instrument(program=bass_program, name="Bass")
    pad = pretty_midi.Instrument(program=pad_program, name="Pad")

    horn = None
    if arrangement["horn"]:
        horn_program = pretty_midi.instrument_name_to_program(arrangement["horn"])
        horn = pretty_midi.Instrument(program=horn_program, name="Horn Swell")

    timpani = None
    if arrangement["timpani"]:
        timpani_program = pretty_midi.instrument_name_to_program(arrangement["timpani"])
        timpani = pretty_midi.Instrument(program=timpani_program, name="Timpani")

    percussion_density = arrangement["percussion_density"]
    percussion_kit = arrangement["percussion_kit"]

    percussion = None
    if percussion_density != "nessuna" and percussion_kit is not None:
        percussion = pretty_midi.Instrument(
            program=0, is_drum=True, name="Percussion"
        )

    for phrase_idx, indices in enumerate(phrases):

        degree = CHORD_PROGRESSION_DEGREES[
            phrase_idx % len(CHORD_PROGRESSION_DEGREES)
        ]

        first_note = rhythmic_notes[indices[0]]
        last_note = rhythmic_notes[indices[-1]]

        phrase_start = first_note[0]
        phrase_end = last_note[0] + last_note[1]
        phrase_duration = phrase_end - phrase_start

        stressed_indices = [
            i for i in indices if rhythmic_notes[i][4]  # is_stressed
        ]

        # -- Basso --
        if arrangement["bass_pattern"] == "walking" and stressed_indices:

            # Si muove su ogni sillaba accentata, alternando
            # fondamentale e quinta: più propulsivo di un'unica
            # nota sostenuta.
            for step, i in enumerate(stressed_indices):

                note_start, note_duration, _v, _pe, _st = rhythmic_notes[i]

                bass_degree = degree if step % 2 == 0 else degree + 4
                bass_pitch = scale_degree_pitch(bass_degree, start_pitch - 12, scale)

                if step + 1 < len(stressed_indices):
                    note_end = rhythmic_notes[stressed_indices[step + 1]][0]
                else:
                    note_end = phrase_end

                note_end = max(note_start + 0.05, note_end - 0.03)

                bass.notes.append(
                    pretty_midi.Note(
                        velocity=80,
                        pitch=max(0, min(127, bass_pitch)),
                        start=note_start,
                        end=note_end,
                    )
                )

        else:
            # Pattern "sostenuto": un'unica nota lunga per frase.
            bass_pitch = scale_degree_pitch(degree, start_pitch - 12, scale)

            bass.notes.append(
                pretty_midi.Note(
                    velocity=75,
                    pitch=max(0, min(127, bass_pitch)),
                    start=phrase_start,
                    end=max(phrase_start + 0.05, phrase_end - 0.02),
                )
            )

        # -- Pad: triade completa (radice, terza, quinta), sostenuta --
        for chord_tone_degree in (degree, degree + 2, degree + 4):

            pad_pitch = scale_degree_pitch(
                chord_tone_degree, start_pitch - 12, scale
            )

            pad.notes.append(
                pretty_midi.Note(
                    velocity=48,
                    pitch=max(0, min(127, pad_pitch)),
                    start=phrase_start,
                    end=phrase_end,
                )
            )

        # -- Corno: entra sostenuto verso la fine della frase, per
        #    dare risalto alla cadenza (piccolo "swell" orchestrale) --
        if horn is not None:

            swell_duration = min(1.5, phrase_duration * 0.4)
            swell_start = max(phrase_start, phrase_end - swell_duration)

            horn_pitch = scale_degree_pitch(degree, start_pitch - 12, scale)

            horn.notes.append(
                pretty_midi.Note(
                    velocity=55,
                    pitch=max(0, min(127, horn_pitch)),
                    start=swell_start,
                    end=phrase_end,
                )
            )

        # -- Timpano: colpo a inizio frase, con accenti extra sulle
        #    sillabe toniche se il livello ritmico lo richiede --
        if timpani is not None:

            timpani_pitch = scale_degree_pitch(degree, start_pitch - 24, scale)

            timpani.notes.append(
                pretty_midi.Note(
                    velocity=60,
                    pitch=max(0, min(127, timpani_pitch)),
                    start=phrase_start,
                    end=phrase_start + 0.3,
                )
            )

            if arrangement["timpani_extra"]:
                for i in stressed_indices:
                    note_start, note_duration, _v, _pe, _st = rhythmic_notes[i]

                    if note_start == phrase_start:
                        continue  # già coperta dal colpo di apertura

                    timpani.notes.append(
                        pretty_midi.Note(
                            velocity=50,
                            pitch=max(0, min(127, timpani_pitch)),
                            start=note_start,
                            end=note_start + min(0.25, note_duration),
                        )
                    )

        # -- Percussione: densità e voce dipendono da genere e
        #    livello ritmico scelti --
        if percussion is not None:

            target_indices = (
                indices if percussion_density == "piena" else stressed_indices
            )

            for i in target_indices:
                note_start, note_duration, _velocity, _is_end, is_stressed = (
                    rhythmic_notes[i]
                )

                if percussion_kit == "drumkit":
                    # Vera batteria: cassa+rullante sulle sillabe
                    # toniche, hi-hat sulle altre (backbeat semplice).
                    if is_stressed:
                        for drum_note in (36, 38):  # Kick, Snare
                            percussion.notes.append(
                                pretty_midi.Note(
                                    velocity=95,
                                    pitch=drum_note,
                                    start=note_start,
                                    end=note_start + 0.08,
                                )
                            )
                    else:
                        percussion.notes.append(
                            pretty_midi.Note(
                                velocity=45,
                                pitch=42,  # Closed Hi-Hat
                                start=note_start,
                                end=note_start + 0.05,
                            )
                        )
                else:
                    # Percussione singola intonata al genere (es.
                    # tamburello per il vocale, maracas per l'acustico).
                    velocity = 65 if is_stressed else 35

                    percussion.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=percussion_kit,
                            start=note_start,
                            end=note_start + min(0.08, note_duration),
                        )
                    )

    return bass, pad, horn, timpani, percussion


# ============================================================
# RIALLINEAMENTO RITMO + ALTEZZE
# ============================================================

def align_pitches_to_rhythm(
    rhythmic_notes,
    generated_pitches,
    output_path: Path,
    arrangement: dict,
    add_accompaniment: bool = ADD_ACCOMPANIMENT,
):
    """
    Applica le altezze generate agli onset/durate del testo, e
    aggiunge (facoltativamente) basso + pad armonico + eventuali
    corno/timpani/percussione, secondo l'arrangiamento scelto
    (genere + livello ritmico), per un risultato multi-strumento
    più simile a un vero arrangiamento che a un singolo filo
    melodico.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    program = pretty_midi.instrument_name_to_program(arrangement["melody"])
    instrument = pretty_midi.Instrument(program=program, name="AI Melody")

    staccato = arrangement["staccato"]

    for i, (start, duration, velocity, _is_phrase_end, _is_stressed) in enumerate(rhythmic_notes):
        pitch = generated_pitches[i % len(generated_pitches)]
        pitch = max(0, min(127, pitch))

        played_duration = duration * (1.0 - staccato)
        played_duration = max(played_duration, 0.05)

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + played_duration
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)

    instruments_used = [arrangement["melody"]]

    if add_accompaniment:
        bass, pad, horn, timpani, percussion = build_accompaniment_instruments(
            rhythmic_notes, arrangement, START_PITCH, SCALE
        )

        midi.instruments.append(bass)
        midi.instruments.append(pad)
        instruments_used += [arrangement["bass"], arrangement["pad"]]

        if horn is not None:
            midi.instruments.append(horn)
            instruments_used.append(arrangement["horn"])

        if timpani is not None:
            midi.instruments.append(timpani)
            instruments_used.append(arrangement["timpani"])

        if percussion is not None:
            midi.instruments.append(percussion)
            perc_label = (
                "Batteria" if arrangement["percussion_kit"] == "drumkit"
                else "Percussione"
            )
            instruments_used.append(perc_label)

    midi.write(str(output_path))
    print(f"\n[OK] Melodia finale salvata in: {output_path}")
    print(
        f"     Genere: {arrangement['genre']} | "
        f"Ritmo: {arrangement['rhythm']}"
    )
    print(f"     Strumenti: {', '.join(instruments_used)}")

# ============================================================
# STAMPA MELODIA
# ============================================================

def print_melody(rhythmic_notes, pitches):
    print("\n" + "=" * 70)
    print("MELODIA GENERATA (ALGORITMICA)")
    print("=" * 70)

    for i, (rhythmic, pitch) in enumerate(zip(rhythmic_notes, pitches)):
        start, duration, velocity, is_phrase_end, is_stressed = rhythmic
        note_name = pretty_midi.note_number_to_name(pitch)
        accent = "*" if is_stressed else " "
        marker = " | fine frase" if is_phrase_end else ""
        print(f"{i:3} t={start:6.2f} dur={duration:4.2f} {note_name:4}{accent} vel={velocity}{marker}")

# ============================================================
# MAIN
# ============================================================

def load_semantic_profiles(path: Path) -> list[dict]:
    """Legge i profili prodotti da DaTestoAMIDI.py."""
    if not path.exists():
        print(f"[INFO] Nessun profilo semantico trovato: {path}")
        print("       Uso la generazione melodica neutra.")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = [item["profile"] for item in data.get("phrases", [])]
        if profiles:
            print(f"[OK] Profili semantici caricati: {len(profiles)} frasi")
        return profiles
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[ATTENZIONE] Profilo semantico non leggibile: {exc}")
        return []


def parse_args():

    parser = argparse.ArgumentParser(
        description="Scheletro ritmico -> melodia + arrangiamento MIDI"
    )

    genre_help = " | ".join(
        f"{name} ({cfg['description']})" for name, cfg in GENRE_PRESETS.items()
    )

    rhythm_help = " | ".join(
        f"{name} ({cfg['description']})" for name, cfg in RHYTHM_PRESETS.items()
    )

    parser.add_argument(
        "--genre",
        choices=list(GENRE_PRESETS),
        default=DEFAULT_GENRE,
        help=f"Timbrica dell'arrangiamento. Opzioni: {genre_help}",
    )

    parser.add_argument(
        "--rhythm",
        choices=list(RHYTHM_PRESETS),
        default=DEFAULT_RHYTHM,
        help=f"Quanto l'arrangiamento è marcato/ritmato. Opzioni: {rhythm_help}",
    )

    parser.add_argument(
        "--no-accompaniment",
        action="store_true",
        help="Genera solo la linea melodica, senza basso/pad/altri strumenti.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per la generazione casuale della melodia (riproducibilità).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    arrangement = resolve_arrangement(args.genre, args.rhythm)

    print("\n" + "=" * 70)
    print("RHYTHM → ALGORITHMIC GENERATOR → MELODY")
    print("=" * 70)
    print(
        f"\nGenere: {args.genre} ({GENRE_PRESETS[args.genre]['description']})"
    )
    print(
        f"Ritmo:  {args.rhythm} ({RHYTHM_PRESETS[args.rhythm]['description']})"
    )

    if not RHYTHMIC_MIDI.exists():
        print(f"\n[ERRORE] Non trovo: {RHYTHMIC_MIDI}\nEsegui prima lo script della prosodia.")
        return

    # 1. Carica ritmo
    rhythmic_notes = load_rhythmic_skeleton(RHYTHMIC_MIDI)
    print(f"\n[OK] Sillabe/eventi ritmici trovati: {len(rhythmic_notes)}")

    # 2. Carica il significato del testo analizzato nella fase precedente
    semantic_profiles = load_semantic_profiles(SEMANTIC_PROFILE_JSON)

    # 3. Genera altezze condizionate dal profilo semantico
    num_notes_needed = len(rhythmic_notes)
    phrase_ends = [note[3] for note in rhythmic_notes]
    generated_pitches = generate_algorithmic_pitches(
        num_notes_needed,
        START_PITCH,
        phrase_ends=phrase_ends,
        seed=args.seed,
        semantic_profiles=semantic_profiles,
    )
    print(f"[OK] Generate {len(generated_pitches)} altezze musicali.")

    # 4. Mostra risultato
    print_melody(rhythmic_notes, generated_pitches)

    # 5. Combina e salva
    align_pitches_to_rhythm(
        rhythmic_notes,
        generated_pitches,
        FINAL_MELODY_MIDI,
        arrangement=arrangement,
        add_accompaniment=not args.no_accompaniment,
    )

if __name__ == "__main__":
    main()