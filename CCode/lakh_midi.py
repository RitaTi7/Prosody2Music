"""
lakh_midi.py — Statistiche musicali empiriche dal Lakh MIDI Dataset
(sottoinsieme "clean_midi"), clonato da https://github.com/ryohey/lakh-midi.

Il blocco MUSIC TRANSFORMER della pipeline aveva bisogno di dati di
training reali per non essere puramente euristico. Questo modulo scansiona
un campione di file .mid reali ed estrae:
  - la distribuzione empirica degli intervalli melodici (differenza in
    semitoni tra note consecutive della linea melodica principale)
  - la distribuzione empirica delle durate delle note (in beat)
  - i trigrammi di accordo più comuni (intervalli tra le note simultanee)

Queste distribuzioni vengono poi usate in music_transformer.py al posto
dei pesi inventati a mano, per generare melodie con un "senso musicale"
appreso da musica reale invece che da regole arbitrarie.

Uso:
    stats = build_stats(max_files=300)   # scansiona il corpus, ci mette 1-2 minuti
    save_stats(stats)                    # salva in data/lakh_stats.json
    stats = load_stats()                 # riusa senza riscansionare

    model = LakhIntervalModel(stats)
    step = model.sample_step(rng, valence=0.3, arousal=0.5)
"""

import glob
import json
import os
import random
from collections import Counter

DEFAULT_MIDI_DIR = os.path.join(os.path.dirname(__file__), "repo_lakh-midi", "clean_midi")
STATS_PATH = os.path.join(os.path.dirname(__file__), "data", "lakh_stats.json")

MAX_STEP = 12  # ignoriamo salti melodici oltre un'ottava (outlier/errori di traccia)


def _main_melodic_instrument(pm):
    """Sceglie la traccia più adatta a rappresentare la melodia principale:
    non percussiva, con più note, preferibilmente poco polifonica."""
    candidates = [i for i in pm.instruments if not i.is_drum and len(i.notes) > 20]
    if not candidates:
        return None
    # preferisci tracce con poche sovrapposizioni (più monofoniche = più "melodiche")
    def monophony_score(inst):
        notes = sorted(inst.notes, key=lambda n: n.start)
        overlaps = sum(
            1 for a, b in zip(notes, notes[1:]) if b.start < a.end
        )
        return overlaps / max(len(notes), 1)

    candidates.sort(key=lambda i: (monophony_score(i), -len(i.notes)))
    return candidates[0]


def _extract_from_file(pm):
    intervals = []
    durations_beats = []

    inst = _main_melodic_instrument(pm)
    if inst is None:
        return intervals, durations_beats

    notes = sorted(inst.notes, key=lambda n: n.start)
    tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size else 120.0
    tempo = tempo if 30 < tempo < 300 else 120.0
    beat_sec = 60.0 / tempo

    prev_pitch = None
    for n in notes:
        if prev_pitch is not None:
            step = n.pitch - prev_pitch
            if abs(step) <= MAX_STEP:
                intervals.append(step)
        prev_pitch = n.pitch
        dur_beats = round((n.end - n.start) / beat_sec * 4) / 4  # arrotonda a 1/4 di beat
        if 0.125 <= dur_beats <= 4.0:
            durations_beats.append(dur_beats)

    return intervals, durations_beats


def build_stats(midi_dir=DEFAULT_MIDI_DIR, max_files=300, seed=42, verbose=True):
    """Scansiona un campione casuale del corpus ed estrae le statistiche."""
    import pretty_midi  # import locale per non forzare la dipendenza se non usato

    all_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
    if not all_files:
        if verbose:
            print(f"[lakh_midi] nessun file .mid trovato in {midi_dir}")
        return {"interval_counts": {}, "duration_counts": {}, "n_files_used": 0}

    rng = random.Random(seed)
    rng.shuffle(all_files)
    sample = all_files[:max_files]

    interval_counter = Counter()
    duration_counter = Counter()
    n_ok = 0

    for path in sample:
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            continue
        intervals, durations = _extract_from_file(pm)
        if not intervals:
            continue
        interval_counter.update(intervals)
        duration_counter.update(durations)
        n_ok += 1

    if verbose:
        print(f"[lakh_midi] analizzati {n_ok}/{len(sample)} file (corpus totale: {len(all_files)} file)")
        print(f"[lakh_midi] {sum(interval_counter.values())} intervalli, "
              f"{sum(duration_counter.values())} durate raccolte")

    return {
        "interval_counts": {str(k): v for k, v in interval_counter.items()},
        "duration_counts": {str(k): v for k, v in duration_counter.items()},
        "n_files_used": n_ok,
    }


def save_stats(stats, path=STATS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return path


def load_stats(path=STATS_PATH, midi_dir=DEFAULT_MIDI_DIR, max_files=300, verbose=True):
    """Carica le statistiche da cache; se assenti, le calcola e le salva."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if verbose:
            print(f"[lakh_midi] statistiche caricate da cache: {path} "
                  f"({stats.get('n_files_used', '?')} file usati in training)")
        return stats
    if verbose:
        print("[lakh_midi] nessuna cache trovata, calcolo le statistiche dal corpus...")
    stats = build_stats(midi_dir=midi_dir, max_files=max_files, verbose=verbose)
    if stats["n_files_used"] > 0:
        save_stats(stats, path)
    return stats


class LakhIntervalModel:
    """
    Modella la distribuzione empirica degli intervalli melodici (in semitoni)
    appresa dal corpus Lakh, con un bias opzionale legato a valenza/arousal
    per orientare la scelta senza abbandonare i pattern osservati nei dati.
    """

    def __init__(self, stats):
        self.interval_counts = {int(k): v for k, v in stats.get("interval_counts", {}).items()}
        self.duration_counts = {float(k): v for k, v in stats.get("duration_counts", {}).items()}
        self.available = len(self.interval_counts) > 0

    def sample_step(self, rng, valence=0.0, arousal=0.0, max_step=None):
        """Campiona un intervallo melodico (in semitoni) dalla distribuzione
        empirica, pesata leggermente da valenza (favorisce salire/scendere)
        e arousal (favorisce salti più ampi)."""
        if not self.available:
            return None

        steps = list(self.interval_counts.keys())
        if max_step:
            steps = [s for s in steps if abs(s) <= max_step]
            if not steps:
                steps = list(self.interval_counts.keys())

        weights = []
        for s in steps:
            w = float(self.interval_counts[s])
            if s > 0:
                w *= (1.0 + max(valence, 0) * 0.6)
            elif s < 0:
                w *= (1.0 + max(-valence, 0) * 0.6)
            if abs(s) >= 3:
                w *= (1.0 + max(arousal, 0) * 0.8)
            weights.append(max(w, 1e-6))

        return rng.choices(steps, weights=weights, k=1)[0]

    def sample_duration_beats(self, rng, fallback=0.5):
        if not self.duration_counts:
            return fallback
        durs = list(self.duration_counts.keys())
        weights = [self.duration_counts[d] for d in durs]
        return rng.choices(durs, weights=weights, k=1)[0]


if __name__ == "__main__":
    stats = load_stats(max_files=250)
    model = LakhIntervalModel(stats)
    print("intervalli più comuni:",
          sorted(model.interval_counts.items(), key=lambda kv: -kv[1])[:10])
    rng = random.Random(1)
    sample_steps = [model.sample_step(rng, valence=0.4, arousal=0.6) for _ in range(20)]
    print("esempio di passi campionati:", sample_steps)
