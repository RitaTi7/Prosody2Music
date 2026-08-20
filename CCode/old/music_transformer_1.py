"""

versione con le pause
 
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
 
# PAUSE_WEIGHT vive in rhythm.py (unica fonte di verità sui pesi delle

# pause, vedi nota lì). Import opzionale con try/except, nello stesso

# spirito degli import di lakh_midi/transformer_melody sopra: se

# rhythm.py non è disponibile (es. si sta ancora usando prosody.py, che

# non ha questa tabella e nemmeno "pauses" nell'output), il transformer

# degrada semplicemente a nessuna attenuazione da pausa, senza rompersi.

try:

    from rhythm import PAUSE_WEIGHT as _PAUSE_WEIGHT

except ImportError:

    _PAUSE_WEIGHT = {}
 
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
 
# Velocity minima di sicurezza dopo l'attenuazione da pausa: sotto questa

# soglia una nota rischia di essere quasi inudibile o, in certi

# sintetizzatori/soundfont, di comportarsi in modo imprevedibile (alcuni

# trattano velocity troppo basse come note-off). Vedi _apply_pause_damping.

MIN_VELOCITY_AFTER_PAUSE_DAMPING = 20
 
 
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
 
 
def derive_bass_and_arpeggio(harmony: Harmony, step_beats: float = 0.5):

    """Deriva le tracce di basso e arpeggio dall'armonia, in BEAT (le

    stesse unità di Melody.notes/Chord.duration in tutta la pipeline).
 
    Prima questa logica esisteva DUPLICATA in due punti (main.py, solo per

    la stampa a schermo e in unità sbagliate — trattava chord.duration come

    "beat relativo" senza convertirlo col tempo reale; synth.py, per il

    rendering audio effettivo, in secondi): due copie indipendenti della

    stessa formula sono un rischio concreto di andare fuori sync la prima

    volta che una delle due viene modificata senza toccare l'altra.

    Centralizzandola qui, sia midi_builder.py (file .mid) sia synth.py

    (file .wav) partono dagli STESSI eventi — il .mid che esporti e il

    .wav che ascolti rappresentano davvero la stessa composizione.
 
    Basso: una nota per accordo, alla fondamentale abbassata di un'ottava.

    Arpeggio: ogni accordo diviso in passi di `step_beats` (default: croma),

    ciclando sulle note dell'accordo (pizzicato/arpa)."""

    bass_notes = []

    arpeggio_notes = []

    for chord in harmony.chords:

        dur = chord.duration

        root_pitch = min(chord.pitches) if chord.pitches else 48

        bass_pitch = max(28, root_pitch - 12)

        bass_notes.append(Note(pitch=bass_pitch, duration=dur, velocity=65))
 
        n_steps = max(1, int(round(dur / step_beats)))

        actual_step = dur / n_steps

        for i in range(n_steps):

            p = chord.pitches[i % len(chord.pitches)] if chord.pitches else bass_pitch

            arpeggio_notes.append(Note(pitch=p, duration=actual_step, velocity=50))

    return bass_notes, arpeggio_notes
 
 
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
 
 
def _pause_weight_map(verse: dict) -> dict:

    """

    {syllable_index: weight 0-1} per il verso corrente, ricavato dalla

    lista "pauses" di analyze_poem(). Se "pauses" manca (es. si sta

    ancora usando prosody.py invece di rhythm.py) o _PAUSE_WEIGHT è

    vuoto (rhythm.py non importabile), ritorna {} — nessuna attenuazione,

    comportamento identico a prima di questa modifica.

    """

    weights = {}

    for p in verse.get("pauses", []) or []:

        w = _PAUSE_WEIGHT.get(p.get("kind"), 0.0)

        idx = p.get("after_syllable_index")

        if idx is None or w <= 0.0:

            continue

        if w > weights.get(idx, 0.0):

            weights[idx] = w

    return weights
 
 
def _apply_pause_damping(velocity: int, weight: float, sensitivity: float) -> int:

    """

    Attenua la velocity in base al peso della pausa e alla sensibilità

    scelta (0 = disattivato, valori più alti = respiro più marcato).

    NON tocca mai la durata della nota — solo l'intensità — per non

    disallineare la melodia da armonia/basso/arpeggio (che calcolano le

    loro durate indipendentemente sommando l'intero "rhythm" del verso;

    accorciare qui la durata di una nota senza "restituire" quel tempo

    da qualche parte farebbe andare la melodia via via fuori sincrono).

    """

    if weight <= 0.0 or sensitivity <= 0.0:

        return velocity

    factor = 1.0 - (sensitivity * weight)

    return max(MIN_VELOCITY_AFTER_PAUSE_DAMPING, int(round(velocity * factor)))
 
 
class MusicTransformer:

    def __init__(self, seed=None, use_lakh=True, use_deep_transformer=True, verbose=True,

                 pause_sensitivity: float = 0.4):

        """

        pause_sensitivity: quanto le pause del testo (virgole, punti,

        sospensioni, fine verso...) si fanno sentire nella melodia, come

        leggera attenuazione della velocity della nota che le precede

        (un respiro, non un silenzio imposto — il transformer resta

        libero di costruire la linea melodica che preferisce). 0.0 la

        disattiva del tutto; 1.0 è il massimo prima del limite di

        sicurezza MIN_VELOCITY_AFTER_PAUSE_DAMPING. Default 0.4: le pause

        più forti (sospensione, fine verso) si notano appena, quelle

        deboli (virgola) quasi per niente.

        """

        self.rng = random.Random(seed)

        self.pause_sensitivity = max(0.0, min(1.0, pause_sensitivity))

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
 
    def _nearest_scale_degree(self, scale, root, midi_pitch, prefer_up=True):

        """

        Grado di scala più vicino a `midi_pitch`. In caso di pareggio tra

        due gradi equidistanti, `prefer_up` decide quale privilegiare:

        prima usava sempre min() puro, che risolve i pareggi scegliendo

        sistematicamente il grado con indice più basso in ogni ottava —

        cioè quasi sempre "lo stesso di prima" quando lo step del modello

        è piccolo (es. ±1 semitono, comune con arousal basso). Questo

        blocca la melodia sulla stessa nota per molti passi di fila.

        Qui il pareggio viene invece risolto nella direzione dello step

        (su o giù), cosa che riduce ma non elimina del tutto il rischio:

        il fix strutturale è comunque accumulare lo spostamento su un

        pitch "grezzo" indipendente da quello quantizzato (vedi generate()).

        """

        octave_shift, semitone_in_octave = divmod(midi_pitch - root, 12)

        best_idx = min(range(len(scale)), key=lambda i: (abs(scale[i] - semitone_in_octave),

                                                           0 if ((scale[i] - semitone_in_octave) >= 0) == prefer_up else 1))

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
 
        # `walk_pitch` è la posizione GREZZA (non quantizzata), somma

        # cumulativa di tutti gli step del modello. `current_pitch` è

        # invece l'ultima nota quantizzata sulla scala, usata solo per

        # l'output MIDI. PRIMA i due venivano fatti coincidere (ogni step

        # veniva sommato al pitch già quantizzato): se uno step piccolo

        # produceva un pareggio in _nearest_scale_degree, il grado

        # restava identico e la nota successiva risultava IDENTICA alla

        # precedente — con step piccoli frequenti (arousal basso) questo

        # poteva ripetersi per molte sillabe di fila, dando l'effetto di

        # una melodia bloccata su una sola nota. Facendo camminare

        # walk_pitch indipendentemente, lo spostamento si accumula

        # sempre, anche quando la quantizzazione "assorbe" un paio di

        # step consecutivi, e la nota di output torna a muoversi.

        walk_pitch = root

        current_pitch = root

        chord_idx = 0
 
        for verse in poem_analysis:

            rhythm = verse["rhythm"]

            pause_weights = _pause_weight_map(verse)
 
            for syll_idx, dur_units in enumerate(rhythm):

                if deep_queue is not None and deep_idx < len(deep_queue):

                    step = deep_queue[deep_idx]

                    deep_idx += 1

                else:

                    step = self._melodic_step_semitones(emotion["valence"], emotion["arousal"])
 
                walk_pitch += step
 
                # Scale Snapping (sulla posizione grezza cumulativa, non

                # sull'ultima nota quantizzata: vedi commento sopra)

                degree = self._nearest_scale_degree(scale, root, walk_pitch, prefer_up=(step >= 0))

                degree = max(-3, min(len(scale) * 2, degree))

                pitch = self._degree_to_midi(root, scale, degree)

                current_pitch = pitch
 
                duration_beats = 0.5 * dur_units

                velocity = 70 + (15 if dur_units == 2 else 0)
 
                # Respiro sulle pause: solo velocity, mai la durata (vedi

                # _apply_pause_damping) — nessun silenzio imposto, il

                # transformer resta libero sul resto della linea melodica.

                weight = pause_weights.get(syll_idx, 0.0)

                velocity = _apply_pause_damping(velocity, weight, self.pause_sensitivity)
 
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
 
