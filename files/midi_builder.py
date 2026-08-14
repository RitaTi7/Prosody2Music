"""
midi_builder.py — Costruisce file MIDI da Melody e Harmony.

Interfaccia:
    build_midi(melody, harmony, tempo=120, out_path="output.mid")
    
Input:
  - melody: Melody object con lista di Note (pitch, duration, velocity)
  - harmony: Harmony object con lista di Chord (pitches, duration)
  - tempo: BPM
  - out_path: percorso file di output MIDI
"""

import os


def build_midi(melody, harmony, tempo=120, out_path="output.mid"):
    """
    Crea un file MIDI da Melody e Harmony.
    
    Per ogni nota della melodia, aggiunge gli accordi della sezione corrente
    su una traccia separata.
    
    Esporta in formato MIDI su out_path.
    """
    try:
        import pretty_midi
    except ImportError:
        print("[midi_builder] Errore: pretty_midi non è installato")
        print("[midi_builder] Installa con: pip install pretty_midi")
        return out_path
    
    # Crea un nuovo file MIDI
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Traccia per la melodia (programma 0 = piano)
    melody_track = pretty_midi.Instrument(program=0, is_drum=False, name="Melody")
    pm.instruments.append(melody_track)
    
    # Traccia per l'armonia (programma 33 = acoustic bass)
    harmony_track = pretty_midi.Instrument(program=33, is_drum=False, name="Harmony")
    pm.instruments.append(harmony_track)
    
    # Aggiungi le note della melodia
    current_time = 0.0
    beat_duration = 60.0 / tempo  # durata di un beat in secondi
    
    for note in melody.notes:
        start_time = current_time
        duration_sec = note.duration * beat_duration
        end_time = start_time + duration_sec
        
        midi_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=start_time,
            end=end_time,
        )
        melody_track.notes.append(midi_note)
        current_time = end_time
    
    # Aggiungi gli accordi (armonia)
    current_time = 0.0
    for chord in harmony.chords:
        start_time = current_time
        duration_sec = chord.duration * beat_duration
        end_time = start_time + duration_sec
        
        for pitch in chord.pitches:
            midi_note = pretty_midi.Note(
                velocity=60,
                pitch=pitch,
                start=start_time,
                end=end_time,
            )
            harmony_track.notes.append(midi_note)
        
        current_time = end_time
    
    # Crea la directory di output se non esiste
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    
    # Salva il file MIDI
    pm.write(out_path)
    return out_path


if __name__ == "__main__":
    from music_transformer import Note, Melody, Chord, Harmony
    
    # Esempio: crea una semplice melodia e armonia
    melody = Melody()
    melody.notes.append(Note(pitch=60, duration=0.5, velocity=80))
    melody.notes.append(Note(pitch=62, duration=0.5, velocity=80))
    melody.notes.append(Note(pitch=64, duration=1.0, velocity=80))
    
    harmony = Harmony()
    harmony.chords.append(Chord(pitches=[48, 52, 55], duration=2.0))
    
    out_path = build_midi(melody, harmony, tempo=120, out_path="test.mid")
    print(f"MIDI salvato in: {out_path}")
