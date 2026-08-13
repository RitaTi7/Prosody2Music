#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generatore musicale da CSV linguistico (Versione Completa - Genere & Umanizzazione).

Migliorie incluse:
- Selezione del Genere (BPM, dinamica e strumenti adattivi).
- Respiro Umano: silenzi sulle virgole e articolazione (staccato/legato).
- Varianza Melodica: arpeggi per parole forti, gradi congiunti per parole deboli.
- Varianza Ritmica: parole corte veloci, parole lunghe lente.

Uso:
    python GeneratoreMusicale.py risultati.csv --genere jazz --output brano.mid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import pandas as pd
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# ============================================================
# CONFIGURAZIONE E PROFILI DI GENERE
# ============================================================

TICKS_PER_BEAT = 480
PERCUSSION_CHANNEL = 9
GRID_DURATIONS = [120, 240, 480, 960]

# Strumenti array: [Melodia, Armonizzazione, Tappeto/Accordi, Basso]
GENRES = {
    "classico": {
        "bpm": 70,
        "velocity_base": 70, 
        "instruments": [68, 40, 52, 43], # Oboe (68), Violino (40), Pad/Coro (52), Contrabbasso (43)
    },
    "pop": {
        "bpm": 120,
        "velocity_base": 100, 
        "instruments": [4, 81, 88, 38],  # Piano Elettrico (4), Synth Lead (81), Synth Pad (88), Synth Bass (38)
    },
    "jazz": {
        "bpm": 85,
        "velocity_base": 85, 
        "instruments": [11, 26, 0, 32],  # Vibrafono (11), Chitarra Jazz (26), Pianoforte (0), Basso Acustico (32)
    },
    "acustico": {
        "bpm": 100,
        "velocity_base": 90, 
        "instruments": [24, 0, 42, 33],  # Chitarra Acustica (24), Pianoforte (0), Violoncello (42), Basso Elettrico (33)
    }
}

# Scale e Accordi
SCALE_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72] # C Maj
SCALE_MINOR = [69, 71, 72, 74, 76, 77, 79, 81] # A Min

CHORDS = {
    "C": [60, 64, 67],    # I Mag
    "Dm": [62, 65, 69],   # ii min
    "E": [64, 68, 71],    # V (modo minore armonico)
    "F": [65, 69, 72],    # IV Mag
    "G": [67, 71, 74],    # V Mag
    "Am": [69, 72, 76]    # vi min / i min
}

ACCENT_INDEX = {
    "ultima sillaba": 0,
    "penultima sillaba": 1,
    "antepenultima sillaba": 2,
    "preantepenultima sillaba": 3,
}

# ============================================================
# STRUTTURA DATI
# ============================================================

@dataclass
class WordEvent:
    word: str
    sentence: str
    model_position: int
    model_label: str
    confidence: float
    phon_position: Optional[int]
    phon_label: str
    agreement: Optional[bool]
    lemma: str
    pos: str
    tag: str
    morphology: str
    dependency: str
    syntactic_role: str
    head: str
    head_lemma: str
    head_token_index: int
    syntactic_depth: int
    sentence_id: int
    sentence_word_index: int
    sentence_word_count: int
    sentence_position: str
    is_first_in_sentence: bool
    is_last_in_sentence: bool
    previous_word: str
    next_word: str
    punctuation_before: str
    punctuation_after: str
    has_pause_after: bool
    has_sentence_end: bool
    is_question_end: bool
    is_exclamation_end: bool

    semantic_category: str = "nessuna"
    is_named_entity: bool = False
    semantic_polarity: float = 0.0
    semantic_polarity_label: str = "neutro"
    semantic_similarity_to_sentence: float = 0.0
    semantic_salience: float = 0.5

    # Parametri musicali generati
    duration: int = TICKS_PER_BEAT
    velocity: int = 80
    
    # Contesto Armonico Assegnato
    chord_name: str = "C"
    chord_notes: List[int] = field(default_factory=lambda: [60, 64, 67])
    scale: List[int] = field(default_factory=lambda: SCALE_MAJOR)

# ============================================================
# UTILITÀ DI PARSING
# ============================================================

def is_true(value) -> bool:
    if pd.isna(value): return False
    if isinstance(value, bool): return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "vero"}

def safe_float(value, default=0.0) -> float:
    try: return default if pd.isna(value) else float(value)
    except: return default

def safe_int(value, default=0) -> int:
    try: return default if pd.isna(value) else int(float(value))
    except: return default

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def quantize_duration(duration: float) -> int:
    return min(GRID_DURATIONS, key=lambda x: abs(x - duration))

def row_to_event(row: pd.Series) -> WordEvent:
    agreement = is_true(row["agreement"]) if not pd.isna(row.get("agreement", pd.NA)) else None
    phon_position = safe_int(row["phon_position"], default=0) if not pd.isna(row.get("phon_position", pd.NA)) else None
    return WordEvent(
        word=str(row.get("word", "")),
        sentence=str(row.get("sentence", "")),
        model_position=safe_int(row.get("model_position", 0)),
        model_label=str(row.get("model_label", "")),
        confidence=safe_float(row.get("confidence", 0.0)),
        phon_position=phon_position,
        phon_label=str(row.get("phon_label", "")),
        agreement=agreement,
        lemma=str(row.get("lemma", "")),
        pos=str(row.get("pos", "")),
        tag=str(row.get("tag", "")),
        morphology=str(row.get("morphology", "")),
        dependency=str(row.get("dependency", "")),
        syntactic_role=str(row.get("syntactic_role", "")),
        head=str(row.get("head", "")),
        head_lemma=str(row.get("head_lemma", "")),
        head_token_index=safe_int(row.get("head_token_index", 0)),
        syntactic_depth=safe_int(row.get("syntactic_depth", 0)),
        sentence_id=safe_int(row.get("sentence_id", 0)),
        sentence_word_index=safe_int(row.get("sentence_word_index", 0)),
        sentence_word_count=max(1, safe_int(row.get("sentence_word_count", 1))),
        sentence_position=str(row.get("sentence_position", "")),
        is_first_in_sentence=is_true(row.get("is_first_in_sentence", False)),
        is_last_in_sentence=is_true(row.get("is_last_in_sentence", False)),
        previous_word=str(row.get("previous_word", "")),
        next_word=str(row.get("next_word", "")),
        punctuation_before=str(row.get("punctuation_before", "")),
        punctuation_after=str(row.get("punctuation_after", "")),
        has_pause_after=is_true(row.get("has_pause_after", False)),
        has_sentence_end=is_true(row.get("has_sentence_end", False)),
        is_question_end=is_true(row.get("is_question_end", False)),
        is_exclamation_end=is_true(row.get("is_exclamation_end", False)),
        semantic_category=str(row.get("semantic_category", "nessuna")),
        is_named_entity=is_true(row.get("is_named_entity", False)),
        semantic_polarity=safe_float(row.get("semantic_polarity", 0.0)),
        semantic_polarity_label=str(row.get("semantic_polarity_label", "neutro")),
        semantic_similarity_to_sentence=safe_float(row.get("semantic_similarity_to_sentence", 0.0)),
        semantic_salience=safe_float(row.get("semantic_salience", 0.5))
    )

# ============================================================
# MOTORE ARMONICO
# ============================================================

def analyze_sentence_harmony(events: list[WordEvent]):
    groups: dict[int, list[WordEvent]] = {}
    for event in events:
        groups.setdefault(event.sentence_id, []).append(event)

    for sentence_id, sentence_events in groups.items():
        total_polarity = sum(e.semantic_polarity for e in sentence_events)
        
        pos_progressions = [
            ["C", "G", "Am", "F"],
            ["C", "F", "C", "G"],
            ["F", "G", "C", "Am"]
        ]
        neg_progressions = [
            ["Am", "F", "Dm", "E"],
            ["Dm", "Am", "E", "Am"],
            ["Am", "G", "F", "E"]
        ]
        
        if total_polarity >= 0:
            progression = pos_progressions[sentence_id % len(pos_progressions)]
            scale = SCALE_MAJOR
        else:
            progression = neg_progressions[sentence_id % len(neg_progressions)]
            scale = SCALE_MINOR
            
        n_words = len(sentence_events)
        words_per_chord = max(1, n_words // len(progression))
        
        for i, event in enumerate(sentence_events):
            chord_idx = min(i // words_per_chord, len(progression) - 1)
            event.chord_name = progression[chord_idx]
            event.chord_notes = CHORDS[event.chord_name]
            event.scale = scale

# ============================================================
# MELODIA E RITMO
# ============================================================

def calculate_duration(event: WordEvent) -> int:
    # Groove base: le particelle sono veloci, le altre normali/lente
    if event.pos in {"ADP", "DET", "PRON", "CCONJ"} or len(event.word) <= 3:
        duration = TICKS_PER_BEAT // 4  # Sedicesimo
    elif event.is_last_in_sentence or event.has_pause_after:
        duration = TICKS_PER_BEAT * 2   # Metà
    else:
        duration = TICKS_PER_BEAT // 2  # Ottavo
        
    # Enfasi emotiva allarga il tempo (rubato)
    if abs(event.semantic_polarity) > 0.3:
        duration = int(duration * 1.5)
        
    return quantize_duration(duration)

def generate_note(event: WordEvent, previous_note: int, voice_offset_idx: int = 0) -> int:
    chord = event.chord_notes
    is_strong = event.syntactic_role in {"radice", "soggetto"} or event.pos in {"VERB", "NOUN", "ADJ"}
    
    if is_strong:
        base_idx = ACCENT_INDEX.get(event.model_label, 0)
        octave_shift = 12 if abs(event.semantic_polarity) > 0.4 else 0
        target_idx = (base_idx + voice_offset_idx) % len(chord)
        note = chord[target_idx] + octave_shift
    else:
        closest_scale_notes = sorted(event.scale, key=lambda x: abs(x - previous_note))
        note = closest_scale_notes[voice_offset_idx % len(closest_scale_notes)]
    
    if event.is_question_end and voice_offset_idx == 0:
        note = max(chord) + 12 if max(chord) < 72 else max(chord)
        
    return clamp(note, 48, 84)

# ============================================================
# TRACCE MIDI
# ============================================================

def create_melodic_track(events: list[WordEvent], program: int, channel: int, name: str, voice_idx: int = 0) -> MidiTrack:
    track = MidiTrack()
    track.append(MetaMessage("track_name", name=name))
    track.append(Message("program_change", channel=channel, program=program, time=0))

    previous_note = 60
    accumulated_delay = 0

    for event in events:
        note = generate_note(event, previous_note, voice_offset_idx=voice_idx)
        previous_note = note
        
        # Articolazione (Gate)
        gate = 0.5 if (len(event.word) <= 3 or event.pos in {"ADP", "DET"}) else 0.9
        actual_play_time = int(event.duration * gate)
        rest_time = event.duration - actual_play_time
        
        # Dinamica
        is_strong = event.syntactic_role in {"radice", "soggetto"}
        base_velocity = min(115, event.velocity + (15 if is_strong else -15))
        final_velocity = clamp(base_velocity - (15 if voice_idx > 0 else 0), 30, 115)
        
        # Pause per punteggiatura (Respiro)
        if event.punctuation_before in {",", ";", ":"}:
            accumulated_delay += TICKS_PER_BEAT // 2
        elif event.is_first_in_sentence and event.sentence_word_index == 0:
            accumulated_delay += TICKS_PER_BEAT

        track.append(Message("note_on", channel=channel, note=note, velocity=final_velocity, time=accumulated_delay))
        track.append(Message("note_off", channel=channel, note=note, velocity=0, time=actual_play_time))
        
        accumulated_delay = rest_time

    return track

def create_chord_track(events: list[WordEvent], program: int, channel: int) -> MidiTrack:
    track = MidiTrack()
    track.append(MetaMessage("track_name", name="Tappeto Armonico"))
    track.append(Message("program_change", channel=channel, program=program, time=0))
    
    chord_duration_acc = 0
    
    for idx, event in enumerate(events):
        chord_duration_acc += event.duration
        next_chord = events[idx+1].chord_name if idx + 1 < len(events) else None
        
        if next_chord != event.chord_name or idx == len(events)-1:
            velocity = 50 
            notes_to_play = [n - 12 for n in event.chord_notes]
            
            for n in notes_to_play:
                track.append(Message("note_on", channel=channel, note=n, velocity=velocity, time=0))
            
            for i, n in enumerate(notes_to_play):
                time_val = chord_duration_acc if i == 0 else 0
                track.append(Message("note_off", channel=channel, note=n, velocity=0, time=time_val))
                
            chord_duration_acc = 0

    return track

def create_bass_track(events: list[WordEvent], program: int, channel: int) -> MidiTrack:
    track = MidiTrack()
    track.append(MetaMessage("track_name", name="Basso"))
    track.append(Message("program_change", channel=channel, program=program, time=0))

    for event in events:
        root_note = event.chord_notes[0] - 24
        play_bass = event.syntactic_role in {"radice", "soggetto"} or event.has_sentence_end or event.is_first_in_sentence
        velocity = clamp(event.velocity - 10, 40, 100) if play_bass else 0
        
        if play_bass:
            track.append(Message("note_on", channel=channel, note=root_note, velocity=velocity, time=0))
            track.append(Message("note_off", channel=channel, note=root_note, velocity=0, time=event.duration))
        else:
            track.append(Message("note_on", channel=channel, note=root_note, velocity=0, time=0))
            track.append(Message("note_off", channel=channel, note=root_note, velocity=0, time=event.duration))
            
    return track

def create_midi(events: list[WordEvent], selected_instruments: list[int], target_bpm: int, output_path: Path):
    midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    
    trk_tempo = MidiTrack()
    trk_tempo.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(target_bpm)))
    midi.tracks.append(trk_tempo)
    
    channel_idx = 0
    for idx, program in enumerate(selected_instruments):
        if channel_idx == PERCUSSION_CHANNEL: channel_idx += 1 
        
        name = f"Traccia {idx+1} (Prog {program})"
        
        if idx == 0:
            midi.tracks.append(create_melodic_track(events, program, channel_idx, f"{name} - Melodia", voice_idx=0))
        elif idx == 1:
            midi.tracks.append(create_melodic_track(events, program, channel_idx, f"{name} - Armonizzazione", voice_idx=1))
        elif idx == 2:
            midi.tracks.append(create_chord_track(events, program, channel_idx))
        elif idx == 3:
            midi.tracks.append(create_bass_track(events, program, channel_idx))
            
        channel_idx += 1
        
    midi.save(output_path)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Genera un file MIDI da un'analisi linguistica CSV.")
    parser.add_argument("csv", help="Il file CSV di input generato dallo script NLP")
    parser.add_argument("--output", default="brano_generato.mid", help="Nome del file MIDI di output")
    parser.add_argument("--genere", choices=GENRES.keys(), default="acustico", 
                        help="Scegli il genere musicale (classico, pop, jazz, acustico)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    events = [row_to_event(row) for _, row in df.iterrows()]
    
    genre_profile = GENRES[args.genere]
    target_bpm = genre_profile["bpm"]
    selected_instruments = genre_profile["instruments"]
    base_velocity = genre_profile["velocity_base"]
    
    analyze_sentence_harmony(events)
    
    for e in events:
        e.duration = calculate_duration(e)
        e.velocity = base_velocity + int(e.semantic_salience * 15)
    
    create_midi(events, selected_instruments, target_bpm, Path(args.output))
    print("-" * 50)
    print(f"🎵 MIDI generato con successo!")
    print(f"🎹 Genere selezionato : {args.genere.upper()}")
    print(f"⏱️  BPM impostati      : {target_bpm}")
    print(f"💾 File salvato come  : {args.output}")
    print("-" * 50)

if __name__ == "__main__":
    main()