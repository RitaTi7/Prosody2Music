from __future__ import annotations

from pathlib import Path
import random
import pretty_midi

# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "output"

RHYTHMIC_MIDI = INPUT_DIR / "rhythmic_skeleton.mid"
FINAL_MELODY_MIDI = INPUT_DIR / "melody.mid"

# Impostazioni melodia
START_PITCH = 60  # C4 (Do centrale)
SCALE = [0, 2, 4, 5, 7, 9, 11]  # Intervalli della scala Maggiore (Do maggiore)
# Se preferisci una scala minore, usa: [0, 2, 3, 5, 7, 8, 10]

# ============================================================
# LETTURA DELLO SCHELETRO
# ============================================================

def load_rhythmic_skeleton(midi_path: Path):
    """
    Legge il MIDI prodotto dalla prima fase.
    Restituisce: [(start, duration, velocity), ...]
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.end - note.start, note.velocity))

    notes.sort(key=lambda x: x[0])
    return notes

# ============================================================
# GENERAZIONE MELODICA (Alternativa a Magenta)
# ============================================================

def generate_algorithmic_pitches(num_notes: int, start_pitch: int = 60):
    """
    Genera una sequenza di note musicalmente coerenti usando un
    "Random Walk" (passeggiata casuale) vincolato a una scala musicale.
    """
    # 1. Costruiamo la scala su 3 ottave per avere range
    allowed_pitches = []
    base_notes = [start_pitch + i for i in SCALE]
    
    for octave in [-1, 0, 1]:  # Un'ottava sotto, ottava base, un'ottava sopra
        allowed_pitches.extend([note + (12 * octave) for note in base_notes])
        
    allowed_pitches = sorted(list(set(allowed_pitches)))
    
    # 2. Troviamo l'indice della nota di partenza
    current_idx = allowed_pitches.index(start_pitch) if start_pitch in allowed_pitches else len(allowed_pitches)//2
    
    pitches = []
    
    # 3. Generiamo le note passo per passo
    for _ in range(num_notes):
        pitches.append(allowed_pitches[current_idx])
        
        # Sceglie il prossimo passo (salto)
        # Pesi: [Pausa, Grado giù, Grado su, Terza giù, Terza su]
        # Favorisce i movimenti per grado congiunto (piccoli passi)
        step = random.choices([-2, -1, 0, 1, 2], weights=[5, 35, 10, 35, 15])[0]
        current_idx += step
        
        # Mantiene l'indice all'interno dei limiti della nostra scala
        current_idx = max(0, min(len(allowed_pitches) - 1, current_idx))
        
    return pitches

# ============================================================
# RIALLINEAMENTO RITMO + ALTEZZE
# ============================================================

def align_pitches_to_rhythm(rhythmic_notes, generated_pitches, output_path: Path):
    """
    Applica le altezze generate agli onset/durate del testo.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    instrument = pretty_midi.Instrument(program=program, name="AI Melody")

    for i, (start, duration, velocity) in enumerate(rhythmic_notes):
        pitch = generated_pitches[i % len(generated_pitches)]
        pitch = max(0, min(127, pitch))

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + duration
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(str(output_path))
    print(f"\n[OK] Melodia finale salvata in: {output_path}")

# ============================================================
# STAMPA MELODIA
# ============================================================

def print_melody(rhythmic_notes, pitches):
    print("\n" + "=" * 70)
    print("MELODIA GENERATA (ALGORITMICA)")
    print("=" * 70)

    for i, (rhythmic, pitch) in enumerate(zip(rhythmic_notes, pitches)):
        start, duration, velocity = rhythmic
        note_name = pretty_midi.note_number_to_name(pitch)
        print(f"{i:3} t={start:6.2f} dur={duration:4.2f} {note_name:4} vel={velocity}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("RHYTHM → ALGORITHMIC GENERATOR → MELODY")
    print("=" * 70)

    if not RHYTHMIC_MIDI.exists():
        print(f"\n[ERRORE] Non trovo: {RHYTHMIC_MIDI}\nEsegui prima lo script della prosodia.")
        return

    # 1. Carica ritmo
    rhythmic_notes = load_rhythmic_skeleton(RHYTHMIC_MIDI)
    print(f"\n[OK] Sillabe/eventi ritmici trovati: {len(rhythmic_notes)}")

    # 2. Genera altezze (Pitches) con algoritmo invece che con Magenta
    num_notes_needed = len(rhythmic_notes)
    generated_pitches = generate_algorithmic_pitches(num_notes_needed, START_PITCH)
    print(f"[OK] Generate {len(generated_pitches)} altezze musicali.")

    # 3. Mostra risultato
    print_melody(rhythmic_notes, generated_pitches)

    # 4. Combina e salva
    align_pitches_to_rhythm(rhythmic_notes, generated_pitches, FINAL_MELODY_MIDI)

if __name__ == "__main__":
    main()