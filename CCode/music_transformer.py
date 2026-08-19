"""
music_transformer.py — Blocco generativo: (ritmo, emotion embedding) -> (melodia, armonia)

Nota di design: la scelta del prossimo intervallo melodico segue una catena
di fallback a tre livelli, in ordine di preferenza:

  1. deep transformer autoregressivo (transformer_melody.py), addestrato da
     zero su sequenze reali del corpus Lakh MIDI, condizionato dall'emotion
     embedding — usato per l'INTERA sequenza di intervalli di una poesia in
     un colpo solo (non nota per nota), per sfruttare davvero la self-attention
     sulle note precedenti;
  2. se torch non è installato o non c'è/non si riesce a costruire una cache
     del modello: modello bigramma sulla distribuzione empirica Lakh
     (LakhIntervalModel, lakh_midi.py) — nota per nota;
  3. se neanche Lakh è disponibile: random-walk euristico pesato da
     valenza/arousal.

La pipeline non si rompe mai: ogni livello mancante fa degradare la qualità
della melodia, non la esecuzione.
"""

import random
from dataclasses import dataclass, field

try:
    from lakh_midi import LakhIntervalModel, load_stats as load_lakh_stats
    _LAKH_AVAILABLE = True
except ImportError:
    _LAKH_AVAILABLE = False

try:
    from transformer_melody import load_or_train_model, generate as _deep_generate
    _DEEP_TRANSFORMER_AVAILABLE = True
except ImportError:
    _DEEP_TRANSFORMER_AVAILABLE = False

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

    def __init__(self, seed=None, use_lakh=True, use_deep_transformer=True, verbose=True):
        self.rng = random.Random(seed)
        self.lakh_model = None
        if use_lakh and _LAKH_AVAILABLE:
            stats = load_lakh_stats(verbose=False)
            model = LakhIntervalModel(stats)
            if model.available:
                self.lakh_model = model

        # Livello 1 (preferito): deep transformer autoregressivo. Se manca
        # una cache addestrata (data/melody_transformer.pt) load_or_train_model
        # prova ad addestrarla al volo dal corpus Lakh (richiede repo_lakh-midi/
        # accanto agli script, vedi lakh_midi.py); se anche quello non è
        # disponibile (né torch, né dati) ritorna None e si scende al livello
        # 2 (bigramma) senza errori. verbose=True di default per non
        # nascondere un eventuale training al primo avvio, che può richiedere
        # qualche minuto su CPU.
        self.deep_model = None
        if use_deep_transformer and _DEEP_TRANSFORMER_AVAILABLE:
            self.deep_model = load_or_train_model(verbose=verbose)

    def _deep_interval_sequence(self, n_intervals, valence, arousal):
        """Pre-genera in un colpo solo l'intera sequenza di intervalli per
        la poesia (o None se il deep transformer non è disponibile), così
        ogni intervallo è condizionato dall'attenzione su TUTTI quelli
        generati prima nella stessa sequenza — a differenza del bigramma,
        che vede solo l'ultimo passo."""
        if self.deep_model is None or n_intervals <= 0:
            return None
        seed = self.rng.randint(0, 2**31 - 1)
        try:
            return _deep_generate(self.deep_model, valence=valence, arousal=arousal,
                                   length=n_intervals, seed=seed)
        except Exception:
            # qualunque problema a inferenza (es. cache corrotta) non deve
            # far crollare la pipeline: si ricade sul bigramma/euristica
            return None

    def _melodic_step_semitones(self, valence, arousal):
        """
        Sceglie il prossimo intervallo melodico in SEMITONI.
        Se è disponibile un modello allenato sul corpus Lakh MIDI, campiona
        dalla distribuzione empirica di intervalli reali (pesata da
        valenza/arousal); altrimenti ricade su un random-walk euristico.
        """
        max_step = 12  # non oltre un'ottava per salto
        if self.lakh_model is not None:
            step = self.lakh_model.sample_step(self.rng, valence=valence, arousal=arousal, max_step=max_step)
            if step is not None:
                return step

        # --- fallback euristico (nessun dato Lakh disponibile) ---
        max_abs = 1 + round(arousal * 2)
        step_choices = [s for s in range(-max_abs, max_abs + 1) if s != 0] or [0]
        weights = []
        for s in step_choices:
            w = 1.0
            if valence >= 0 and s > 0:
                w += valence
            if valence < 0 and s < 0:
                w += -valence
            weights.append(w)
        return self.rng.choices(step_choices, weights=weights, k=1)[0]

    def _nearest_scale_degree(self, scale, root, midi_pitch):
        """Arrotonda un pitch MIDI cromatico al grado di scala più vicino,
        così gli intervalli campionati dal corpus (cromatici) restano
        comunque dentro la tonalità scelta dall'emotion embedding."""
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
        progressions = PROGRESSIONS[mode]
        prog = self.rng.choice(progressions)

        # Livello 1: se il deep transformer è disponibile, genera QUI l'intera
        # sequenza di intervalli per tutta la poesia in un solo colpo (una
        # chiamata autoregressiva che vede l'attenzione su tutte le note
        # precedenti), invece che nota per nota dentro il loop sottostante.
        n_syllables_total = sum(len(verse["rhythm"]) for verse in poem_analysis)
        deep_queue = self._deep_interval_sequence(
            n_syllables_total, emotion["valence"], emotion["arousal"]
        )
        deep_idx = 0

        current_pitch = root  # partenza dalla tonica
        chord_idx = 0

        for verse in poem_analysis:
            syllables = verse["syllables"]
            rhythm = verse["rhythm"]  # 2=tonica, 1=atona (in "impulsi")

            # --- MELODIA: una nota per sillaba ---
            for dur_units in rhythm:
                if deep_queue is not None and deep_idx < len(deep_queue):
                    step = deep_queue[deep_idx]
                    deep_idx += 1
                else:
                    # coda del deep transformer esaurita o non disponibile:
                    # livello 2 (bigramma Lakh) / livello 3 (euristica)
                    step = self._melodic_step_semitones(emotion["valence"], emotion["arousal"])
                candidate_pitch = current_pitch + step
                # aggancia il pitch cromatico campionato alla scala della modalità scelta
                degree = self._nearest_scale_degree(scale, root, candidate_pitch)
                degree = max(-3, min(len(scale) * 2, degree))
                pitch = self._degree_to_midi(root, scale, degree)
                current_pitch = pitch
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

        if deep_queue is not None:
            melody_source = "deep_transformer" if deep_idx >= n_syllables_total else "deep_transformer+fallback"
        elif self.lakh_model is not None:
            melody_source = "lakh_bigram"
        else:
            melody_source = "heuristic"
        meta = {"mode": mode, "root": root, "tempo": tempo, "melody_source": melody_source}
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