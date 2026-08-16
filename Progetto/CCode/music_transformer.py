"""
music_transformer.py — Blocco generativo: (ritmo, emotion embedding) -> (melodia, armonia)

Nota di design: un vero "Music Transformer" (Huang et al.) è un modello
sequence-to-sequence attention-based addestrato su grandi corpora MIDI.
Qui non abbiamo un modello addestrato né dati di training, quindi questo
modulo implementa uno strato generativo *condizionato* — una policy
stocastica pesata dall'emotion embedding — che gioca lo stesso ruolo
architetturale nella pipeline (prende rhythm+embedding, produce note),
e può essere sostituito 1:1 da un vero transformer addestrato in futuro
senza toccare il resto della pipeline (l'interfaccia è la stessa:
generate(rhythm, emotion) -> Melody, Harmony).
"""

import random
from dataclasses import dataclass, field

# --- Scale musicali (intervalli in semitoni dalla tonica) ---
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],       # minore ma più "luminoso": arousal alto + valenza bassa
    "phrygian": [0, 1, 3, 5, 7, 8, 10],     # cupo/teso: valenza molto bassa
    "lydian": [0, 2, 4, 6, 7, 9, 11],       # sognante: valenza alta + arousal basso
}

# progressioni di accordi (gradi della scala, 0-based) per modalità
PROGRESSIONS = {
    "major": [[0, 3, 4, 0], [5, 3, 0, 4], [0, 4, 5, 3]],
    "minor": [[0, 5, 6, 0], [0, 3, 4, 0], [0, 6, 3, 4]],
    "dorian": [[0, 3, 0, 4]],
    "phrygian": [[0, 1, 0, 6]],
    "lydian": [[0, 4, 0, 3]],
}


@dataclass
class Note:
    pitch: int       # MIDI note number
    duration: float  # in beat
    velocity: int = 80


@dataclass
class Melody:
    notes: list = field(default_factory=list)


@dataclass
class Chord:
    pitches: list       # lista di note MIDI
    duration: float


@dataclass
class Harmony:
    chords: list = field(default_factory=list)


def choose_mode(emotion: dict) -> str:
    """Sceglie la scala musicale in base a valenza/arousal/tenerezza."""
    v, a, t = emotion["valence"], emotion["arousal"], emotion["tenderness"]
    if v >= 0.35 and a <= 0.15:
        return "lydian"        # sereno, sognante
    if v >= 0.15:
        return "major"         # positivo
    if v <= -0.45 and a >= 0.35:
        return "phrygian"      # cupo e teso
    if a >= 0.35:
        return "dorian"        # intenso ma non del tutto negativo
    return "minor"             # malinconico di default


def choose_tempo(emotion: dict) -> float:
    """BPM in base all'arousal."""
    a = emotion["arousal"]
    return round(70 + a * 50)  # 55..~130 bpm circa, capped implicitamente


def choose_root(text_seed: str) -> int:
    """Sceglie una tonica (MIDI base ottava 4) in modo deterministico dal testo."""
    roots = [60, 62, 63, 65, 67, 69, 70]  # C D Eb F G A Bb (varietà ma consonanti)
    h = sum(ord(c) for c in text_seed) if text_seed else 0
    return roots[h % len(roots)]


class MusicTransformer:
    """
    Interfaccia: generate(poem_analysis, emotion) -> (Melody, Harmony, meta)
    poem_analysis: output di prosody.analyze_poem()
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _melodic_step(self, scale, degree, valence, arousal):
        """Sceglie il prossimo grado della scala (random-walk pesato)."""
        # ampiezza del salto proporzionale all'arousal
        max_step = 1 + round(arousal * 2)  # 1..3 gradi
        step_choices = list(range(-max_step, max_step + 1))
        step_choices.remove(0) if 0 in step_choices and len(step_choices) > 1 else None

        # bias: valenza alta -> preferenza per salire, bassa -> scendere
        weights = []
        for s in step_choices:
            w = 1.0
            if valence >= 0 and s > 0:
                w += valence
            if valence < 0 and s < 0:
                w += -valence
            weights.append(w)

        step = self.rng.choices(step_choices, weights=weights, k=1)[0]
        new_degree = degree + step
        # mantieni il grado in un ambito ragionevole (2 ottave)
        new_degree = max(-3, min(len(scale) * 2, new_degree))
        return new_degree

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
        progressions = PROGRESSIONS[mode]
        prog = self.rng.choice(progressions)

        degree = 0  # partenza dalla tonica
        chord_idx = 0

        for verse in poem_analysis:
            syllables = verse["syllables"]
            rhythm = verse["rhythm"]  # 2=tonica, 1=atona (in "impulsi")

            # --- MELODIA: una nota per sillaba ---
            for dur_units in rhythm:
                degree = self._melodic_step(scale, degree, emotion["valence"], emotion["arousal"])
                pitch = self._degree_to_midi(root, scale, degree)
                duration_beats = 0.5 * dur_units  # impulso=1 -> croma, =2 -> semiminima
                velocity = 70 + (15 if dur_units == 2 else 0)
                melody.notes.append(Note(pitch=pitch, duration=duration_beats, velocity=velocity))

            # --- ARMONIA: un accordo per verso ---
            chord_degree = prog[chord_idx % len(prog)]
            chord_root = self._degree_to_midi(root - 12, scale, chord_degree)  # un'ottava sotto la melodia
            third = self._degree_to_midi(root - 12, scale, chord_degree + 2)
            fifth = self._degree_to_midi(root - 12, scale, chord_degree + 4)
            verse_duration = sum(0.5 * d for d in rhythm)
            harmony.chords.append(Chord(pitches=[chord_root, third, fifth], duration=verse_duration))
            chord_idx += 1

        meta = {"mode": mode, "root": root, "tempo": tempo}
        return melody, harmony, meta


if __name__ == "__main__":
    from prosody import analyze_poem
    from emotion import analyze_emotion

    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    pa = analyze_poem(demo)
    em = analyze_emotion(demo)
    mt = MusicTransformer(seed=42)
    melody, harmony, meta = mt.generate(pa, em, text_seed=demo)
    print("meta:", meta)
    print("melody notes:", [(n.pitch, n.duration) for n in melody.notes])
    print("harmony chords:", [(c.pitches, c.duration) for c in harmony.chords])
