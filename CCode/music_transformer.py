"""
music_transformer.py — Blocco generativo in sola INFERENZA ONLINE (Fase 2)
"""

import random
from dataclasses import dataclass, field

try:
    from lakh_midi import LakhIntervalModel, load_stats as load_lakh_stats
    _LAKH_AVAILABLE = True
except ImportError:
    _LAKH_AVAILABLE = False

try:
    from transformer_melody import load_inference_model, generate as _deep_generate
    _DEEP_TRANSFORMER_AVAILABLE = True
except ImportError:
    _DEEP_TRANSFORMER_AVAILABLE = False

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

PROGRESSIONS = {
    "major": [[0, 3, 4, 0], [5, 3, 0, 4], [0, 4, 5, 3]],
    "minor": [[0, 5, 6, 0], [0, 3, 4, 0], [0, 6, 3, 4]],
    "dorian": [[0, 3, 0, 4]],
    "phrygian": [[0, 1, 0, 6]],
    "lydian": [[0, 4, 0, 3]],
}


@dataclass
class Note:
    pitch: int
    duration: float
    velocity: int = 80


@dataclass
class Melody:
    notes: list = field(default_factory=list)


@dataclass
class Chord:
    pitches: list
    duration: float


@dataclass
class Harmony:
    chords: list = field(default_factory=list)


def choose_mode(emotion: dict) -> str:
    v, a = emotion["valence"], emotion["arousal"]
    if v >= 0.35 and a <= 0.15:
        return "lydian"
    if v >= 0.15:
        return "major"
    if v <= -0.45 and a >= 0.35:
        return "phrygian"
    if a >= 0.35:
        return "dorian"
    return "minor"


def choose_tempo(emotion: dict) -> float:
    return round(70 + emotion["arousal"] * 50)


def choose_root(text_seed: str) -> int:
    roots = [60, 62, 63, 65, 67, 69, 70]
    h = sum(ord(c) for c in text_seed) if text_seed else 0
    return roots[h % len(roots)]


class MusicTransformer:
    def __init__(self, seed=None, use_lakh=True, use_deep_transformer=True, verbose=True):
        self.rng = random.Random(seed)
        self.lakh_model = None
        if use_lakh and _LAKH_AVAILABLE:
            stats = load_lakh_stats(verbose=False)
            model = LakhIntervalModel(stats)
            if model.available:
                self.lakh_model = model

        # CARICAMENTO INFERENZA ONLINE (Nessun training al volo)
        self.deep_model = None
        if use_deep_transformer and _DEEP_TRANSFORMER_AVAILABLE:
            self.deep_model = load_inference_model()

    def get_model_stats(self) -> dict:
        if self.deep_model is not None:
            stats = {"type": "Deep Transformer (Pre-trained Offline)"}
            stats.update(getattr(self.deep_model, "stats", {}))
            return stats
        elif self.lakh_model is not None:
            return {"type": "Lakh Bigram Model (Statistical Fallback)"}
        else:
            return {"type": "Heuristic Random Walk (Fallback)"}

    def _deep_interval_sequence(self, n_intervals, valence, arousal):
        if self.deep_model is None or n_intervals <= 0:
            return None
        seed = self.rng.randint(0, 2**31 - 1)
        try:
            return _deep_generate(self.deep_model, valence=valence, arousal=arousal,
                                   length=n_intervals, seed=seed)
        except Exception:
            return None

    def _melodic_step_semitones(self, valence, arousal):
        if self.lakh_model is not None:
            step = self.lakh_model.sample_step(self.rng, valence=valence, arousal=arousal, max_step=12)
            if step is not None:
                return step

        max_abs = 1 + round(arousal * 2)
        step_choices = [s for s in range(-max_abs, max_abs + 1) if s != 0] or [0]
        weights = [(1.0 + valence if s > 0 else 1.0 - valence) for s in step_choices]
        return self.rng.choices(step_choices, weights=weights, k=1)[0]

    def _nearest_scale_degree(self, scale, root, midi_pitch):
        octave_shift, semitone_in_octave = divmod(midi_pitch - root, 12)
        best_idx = min(range(len(scale)), key=lambda i: abs(scale[i] - semitone_in_octave))
        return best_idx + octave_shift * len(scale)

    def _degree_to_midi(self, root, scale, degree):
        octave_shift, scale_idx = divmod(degree, len(scale))
        return root + scale[scale_idx] + 12 * octave_shift

    def generate(self, poem_analysis, emotion, text_seed=""):
        mode = choose_mode(emotion)
        scale = SCALES[mode]
        root = choose_root(text_seed)
        tempo = choose_tempo(emotion)

        melody = Melody()
        harmony = Harmony()
        prog = self.rng.choice(PROGRESSIONS[mode])

        n_syllables_total = sum(len(verse["rhythm"]) for verse in poem_analysis)
        
        # FASE ONLINE: Generazione con il modello generale
        deep_queue = self._deep_interval_sequence(
            n_syllables_total, emotion["valence"], emotion["arousal"]
        )
        deep_idx = 0

        current_pitch = root
        chord_idx = 0

        for verse in poem_analysis:
            rhythm = verse["rhythm"]

            for dur_units in rhythm:
                if deep_queue is not None and deep_idx < len(deep_queue):
                    step = deep_queue[deep_idx]
                    deep_idx += 1
                else:
                    step = self._melodic_step_semitones(emotion["valence"], emotion["arousal"])
                
                candidate_pitch = current_pitch + step
                
                # Scale Snapping
                degree = self._nearest_scale_degree(scale, root, candidate_pitch)
                degree = max(-3, min(len(scale) * 2, degree))
                pitch = self._degree_to_midi(root, scale, degree)
                current_pitch = pitch
                
                duration_beats = 0.5 * dur_units
                velocity = 70 + (15 if dur_units == 2 else 0)
                melody.notes.append(Note(pitch=pitch, duration=duration_beats, velocity=velocity))

            # Armonia
            chord_degree = prog[chord_idx % len(prog)]
            chord_root = self._degree_to_midi(root - 12, scale, chord_degree)
            third = self._degree_to_midi(root - 12, scale, chord_degree + 2)
            fifth = self._degree_to_midi(root - 12, scale, chord_degree + 4)
            verse_duration = sum(0.5 * d for d in rhythm)
            harmony.chords.append(Chord(pitches=[chord_root, third, fifth], duration=verse_duration))
            chord_idx += 1

        melody_source = "deep_transformer" if deep_queue is not None else "fallback"
        meta = {
            "mode": mode,
            "root": root,
            "tempo": tempo,
            "melody_source": melody_source,
            "model_stats": self.get_model_stats()
        }
        return melody, harmony, meta