"""
main.py — Orchestrazione della pipeline completa:

  POESIA
    -> analisi semantica (emotion embedding)
    -> analisi prosodica (sillabe + accenti -> RHYTHM)
    -> MUSIC TRANSFORMER -> MELODIA + ARMONIA
    -> MIDI
    -> Synth/Audio (WAV)

Uso:
    python3 main.py                      # usa la poesia demo inclusa
    python3 main.py mia_poesia.txt        # legge da file
"""

import argparse
import sys
import os
from visualizer import plot_melody_and_rhythm, plot_emotion_space
from prosody import analyze_poem
from emotion import analyze_emotion
from music_transformer import MusicTransformer
from midi_builder import build_midi
from synth import mix_and_export
from instruments import INSTRUMENT_PRESETS, validate_instrument, choose_by_emotion

DEMO_POEM = """Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita"""

# Limiti di sicurezza per i parametri esposti all'utente. Non esponiamo mai
# la durata delle singole note: sono derivate dal ritmo prosodico e
# sincronizzate 1:1 con l'armonia (la durata di ogni accordo è la somma
# delle durate delle note del verso corrispondente). Un tempo o uno scale
# fuori range possono causare divisioni per numeri assurdi in
# midi_builder.py/synth.py (beat_sec = 60/tempo) o allocazioni enormi in
# synth.py (l'array audio è lungo quanto la durata totale in campioni).
TEMPO_MIN, TEMPO_MAX = 40, 180          # bpm
DURATION_SCALE_MIN, DURATION_SCALE_MAX = 0.25, 4.0  # moltiplicatore globale


def _validate_tempo(value):
    """Converte e clampa un tempo fornito dall'utente. Solleva ValueError
    con un messaggio chiaro se il valore non è un numero utilizzabile."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"tempo non valido: {value!r} (deve essere un numero, es. 90)")
    if v <= 0 or v != v:  # v != v -> NaN
        raise ValueError(f"tempo non valido: {v} (deve essere un numero positivo)")
    clamped = max(TEMPO_MIN, min(TEMPO_MAX, v))
    if clamped != v:
        print(f"[main] tempo {v} fuori range [{TEMPO_MIN},{TEMPO_MAX}] bpm, portato a {clamped}")
    return clamped


def _validate_duration_scale(value):
    """Converte e clampa il moltiplicatore globale di durata. Si applica
    in modo identico a melodia e armonia, quindi non rompe mai la
    sincronia tra le due (a differenza di modificare le durate nota per
    nota, che invece la romperebbe)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"duration_scale non valido: {value!r} (deve essere un numero, es. 1.0)")
    if v <= 0 or v != v:
        raise ValueError(f"duration_scale non valido: {v} (deve essere un numero positivo)")
    clamped = max(DURATION_SCALE_MIN, min(DURATION_SCALE_MAX, v))
    if clamped != v:
        print(f"[main] duration_scale {v} fuori range [{DURATION_SCALE_MIN},{DURATION_SCALE_MAX}], portato a {clamped}")
    return clamped


def apply_duration_scale(melody, harmony, scale):
    """Applica lo stesso moltiplicatore a tutte le note e a tutti gli
    accordi, cosicché melodia e armonia restino perfettamente sincronizzate
    (l'armonia è calcolata come somma delle durate della melodia dello
    stesso verso: scalare entrambe allo stesso modo preserva quella somma)."""
    for note in melody.notes:
        note.duration *= scale
    for chord in harmony.chords:
        chord.duration *= scale


def run_pipeline(text, out_dir=".", basename="output", seed=42, verbose=True,
                  tempo_override=None, duration_scale=1.0,
                  melody_instrument=None, harmony_instrument=None):
    os.makedirs(out_dir, exist_ok=True)
    duration_scale = _validate_duration_scale(duration_scale)
    if melody_instrument is not None:
        melody_instrument = validate_instrument(melody_instrument)
    if harmony_instrument is not None:
        harmony_instrument = validate_instrument(harmony_instrument)

    # 1) Analisi prosodica: sillabe + accenti -> ritmo
    poem_analysis = analyze_poem(text)

    # 2) Analisi semantica: emotion embedding
    emotion = analyze_emotion(text)

    if verbose:
        print("=== ANALISI PROSODICA (PhonItalia + Q2Stress + euristica) ===")
        for v in poem_analysis:
            syll_str = " | ".join(
                (s["text"].upper() if s["stressed"] else s["text"])
                for s in v["syllables"]
            )
            sources = [s["stress_source"] for s in v["syllables"] if s["stressed"]]
            print(f"  {v['text']}")
            print(f"    sillabe: {syll_str}")
            print(f"    ritmo:   {v['rhythm']}")
            print(f"    fonti accento: {sources}")
        print()
        print("=== EMOTION EMBEDDING ===")
        print(f"  valenza: {emotion['valence']:+.2f}  "
              f"arousal: {emotion['arousal']:+.2f}  "
              f"tenerezza: {emotion['tenderness']:+.2f}")
        print(f"  parole rilevate: {emotion['matched']}")
        print()

    # 3) MUSIC TRANSFORMER: rhythm + emotion -> melodia + armonia
    mt = MusicTransformer(seed=seed, verbose=verbose)
    melody, harmony, meta = mt.generate(poem_analysis, emotion, text_seed=text)

    # tempo: se l'utente ne specifica uno, sostituisce quello scelto
    # automaticamente dall'arousal, ma resta clampato a un range sicuro
    if tempo_override is not None:
        meta["tempo"] = _validate_tempo(tempo_override)

    # scale di durata: moltiplicatore globale, applicato a melodia e
    # armonia insieme (mai alle singole note separatamente, vedi apply_duration_scale)
    if duration_scale != 1.0:
        apply_duration_scale(melody, harmony, duration_scale)

    # strumenti: se l'utente non ne specifica, si scelgono automaticamente
    # in base all'emotion embedding (coerenti col tono della poesia)
    if melody_instrument is None or harmony_instrument is None:
        auto_mel, auto_harm = choose_by_emotion(emotion)
        melody_instrument = melody_instrument or auto_mel
        harmony_instrument = harmony_instrument or auto_harm

    if verbose:
        print("=== MUSIC TRANSFORMER (output) ===")
        print(f"  modalità: {meta['mode']}   tonica MIDI: {meta['root']}   tempo: {meta['tempo']} bpm"
              + (f"   duration_scale: {duration_scale}x" if duration_scale != 1.0 else ""))
        print(f"  fonte melodia: {meta['melody_source']}")
        print(f"  strumenti: melodia={melody_instrument} ({INSTRUMENT_PRESETS[melody_instrument]['label']})"
              f"  armonia={harmony_instrument} ({INSTRUMENT_PRESETS[harmony_instrument]['label']})")
        print(f"  note melodia: {len(melody.notes)}   accordi armonia: {len(harmony.chords)}")
        print()
    # Generazione Grafici Esplicativi
    plot_melody_and_rhythm(melody, poem_analysis, save_path=os.path.join(out_dir, "melody_rhythm.png"))
    plot_emotion_space(emotion, meta["mode"], save_path=os.path.join(out_dir, "emotion_space.png"))
    # 4) MIDI
    midi_path = os.path.join(out_dir, f"{basename}.mid")
    build_midi(melody, harmony, tempo=meta["tempo"],
               melody_instrument=melody_instrument, harmony_instrument=harmony_instrument,
               out_path=midi_path)

    # 5) Synth / Audio
    wav_path = os.path.join(out_dir, f"{basename}.wav")
    mix_and_export(melody, harmony, meta["tempo"],
                    melody_instrument=melody_instrument, harmony_instrument=harmony_instrument,
                    out_path=wav_path)

    if verbose:
        print("=== OUTPUT ===")
        print(f"  MIDI: {midi_path}")
        print(f"  WAV:  {wav_path}")

    return midi_path, wav_path, meta, emotion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poesia -> Musica")
    parser.add_argument("poem_file", nargs="?", default=None,
                         help="file di testo con la poesia (un verso per riga). Se omesso, usa la demo.")
    parser.add_argument("--tempo", type=float, default=None,
                         help=f"bpm, sovrascrive quello scelto dall'emotion embedding "
                              f"(clampato a [{TEMPO_MIN},{TEMPO_MAX}])")
    parser.add_argument("--duration-scale", type=float, default=1.0,
                         help=f"moltiplicatore globale di durata "
                              f"(clampato a [{DURATION_SCALE_MIN},{DURATION_SCALE_MAX}])")
    parser.add_argument("--out-dir", default=".", help="cartella di output")
    parser.add_argument("--basename", default="output", help="nome base dei file generati")
    parser.add_argument("--seed", type=int, default=42, help="seed per la generazione")
    parser.add_argument("--melody-instrument", default=None,
                         choices=sorted(INSTRUMENT_PRESETS),
                         help="strumento per la melodia; se omesso, scelto dall'emotion embedding")
    parser.add_argument("--harmony-instrument", default=None,
                         choices=sorted(INSTRUMENT_PRESETS),
                         help="strumento per l'armonia; se omesso, scelto dall'emotion embedding")
    args = parser.parse_args()

    if args.poem_file:
        with open(args.poem_file, "r", encoding="utf-8") as f:
            poem_text = f.read()
    else:
        poem_text = DEMO_POEM
        print("(nessun file fornito, uso la poesia demo inclusa)\n")

    try:
        run_pipeline(
            poem_text,
            out_dir=args.out_dir,
            basename=args.basename,
            seed=args.seed,
            tempo_override=args.tempo,
            duration_scale=args.duration_scale,
            melody_instrument=args.melody_instrument,
            harmony_instrument=args.harmony_instrument,
        )
    except ValueError as e:
        print(f"[main] parametro non valido: {e}")
        sys.exit(1)