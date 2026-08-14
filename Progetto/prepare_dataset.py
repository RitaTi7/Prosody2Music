from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pretty_midi

MIN_PITCH = 36
MAX_PITCH = 84


def choose_melodic_track(midi: pretty_midi.PrettyMIDI):
    candidates = []
    for inst in midi.instruments:
        if inst.is_drum or not inst.notes:
            continue
        pitches = [n.pitch for n in inst.notes]
        # Preferisce tracce con molte note e registro melodico plausibile.
        avg_pitch = sum(pitches) / len(pitches)
        score = len(pitches) * (1.0 - min(abs(avg_pitch - 66.0), 36.0) / 36.0)
        candidates.append((score, inst))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def note_features(notes, bpm: float):
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return None

    # Convertiamo i tempi in beat usando il primo tempo del MIDI.
    sec_per_beat = 60.0 / max(1.0, bpm)
    events = []
    prev_start = notes[0].start
    velocities = [n.velocity for n in notes]
    median_vel = float(np.median(velocities))

    for i, n in enumerate(notes):
        start_beat = n.start / sec_per_beat
        duration = max(0.0625, (n.end - n.start) / sec_per_beat)
        delta = max(0.0, start_beat - prev_start / sec_per_beat)
        # Proxy dell'accento: velocity alta o posizione metrica forte.
        position = start_beat % 4.0
        metric_accent = 1.0 if position < 0.08 else 0.0
        stress = 1.0 if (n.velocity >= median_vel or metric_accent) else 0.0
        phrase_end = 1.0 if i == len(notes) - 1 else 0.0

        events.append([
            min(duration / 2.0, 1.0),
            min(delta / 2.0, 1.0),
            n.velocity / 127.0,
            stress,
            phrase_end,
            0.0,  # valence: assente nel pretraining musicale
            0.0,  # energy
            0.0,  # movement
            0.0,  # tension
        ])
        prev_start = n.start

    targets = [max(MIN_PITCH, min(MAX_PITCH, n.pitch)) for n in notes]
    return np.asarray(events, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepara sequenze ritmo->melodia da una cartella MIDI")
    parser.add_argument("midi_dir", type=Path, help="Cartella con MIDI Lakh")
    parser.add_argument("--output", type=Path, default=Path("dataset/rhythm_melody.npz"))
    parser.add_argument("--max-files", type=int, default=0, help="0 = tutti")
    parser.add_argument("--max-notes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    paths = list(args.midi_dir.rglob("*.mid")) + list(args.midi_dir.rglob("*.midi"))
    random.shuffle(paths)
    if args.max_files:
        paths = paths[:args.max_files]

    X, Y = [], []
    skipped = 0

    for idx, path in enumerate(paths, 1):
        try:
            midi = pretty_midi.PrettyMIDI(str(path))
            track = choose_melodic_track(midi)
            if track is None or len(track.notes) < 8:
                skipped += 1
                continue
            tempo = float(np.median(midi.get_tempo_changes()[1])) if len(midi.get_tempo_changes()[1]) else 100.0
            result = note_features(track.notes[:args.max_notes], tempo)
            if result is None:
                skipped += 1
                continue
            x, y = result
            X.append(x)
            Y.append(y)
        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {path}: {exc}")

        if idx % 1000 == 0:
            print(f"Processati {idx}/{len(paths)} MIDI...")

    if not X:
        raise RuntimeError("Nessun esempio valido trovato.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=np.array(X, dtype=object), Y=np.array(Y, dtype=object))
    print(f"[OK] Dataset: {args.output}")
    print(f"     Esempi: {len(X)} | scartati: {skipped}")


if __name__ == "__main__":
    main()