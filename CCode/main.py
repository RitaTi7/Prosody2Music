"""
main.py — Orchestrazione della pipeline multilivello:

  POESIA
    -> Analisi semantica (Emotion embedding)
    -> Analisi prosodica (Sillabe + Accenti -> Rhythm)
    -> MUSIC TRANSFORMER -> Melodia + Armonia
    -> SELEZIONE NLP -> Generazione Orchestra a 4 Strumenti (Lead, Pad, Basso, Arpeggio)
    -> MIDI Multitraccia
    -> Synth Stereo (WAV)

Uso:
    python3 main.py                      # Usa la poesia demo inclusa (NLP sceglie gli strumenti)
    python3 main.py mia_poesia.txt        # Legge da file
    python3 main.py mia_poesia.txt --melody-instrument clarinet        # forzi solo la melodia, l'armonia resta automatica
    python3 main.py mia_poesia.txt --melody-instrument clarinet --harmony-instrument organ --basename mia_canzone
"""

import argparse
import sys
import os
from visualizer import plot_melody_and_rhythm, plot_emotion_space
from prosody import analyze_poem
from emotion import analyze_emotion
from music_transformer import MusicTransformer, derive_bass_and_arpeggio
from midi_builder import build_midi
from synth import mix_and_export, render_with_fluidsynth
from instruments import INSTRUMENT_PRESETS
from nlp_instruments import choose_by_nlp

DEMO_POEM = """Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita"""

TEMPO_MIN, TEMPO_MAX = 40, 180          # bpm
DURATION_SCALE_MIN, DURATION_SCALE_MAX = 0.25, 4.0  # moltiplicatore globale


def _validate_tempo(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"tempo non valido: {value!r} (deve essere un numero, es. 90)")
    if v <= 0 or v != v:
        raise ValueError(f"tempo non valido: {v} (deve essere un numero positivo)")
    clamped = max(TEMPO_MIN, min(TEMPO_MAX, v))
    if clamped != v:
        print(f"[main] tempo {v} fuori range [{TEMPO_MIN},{TEMPO_MAX}] bpm, portato a {clamped}")
    return clamped


def _validate_duration_scale(value):
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
    for note in melody.notes:
        note.duration *= scale
    for chord in harmony.chords:
        chord.duration *= scale


def run_pipeline(text, out_dir="./output", basename="output", seed=None, verbose=True,
                  tempo_override=None, duration_scale=1.0, melody_instrument_override=None, harmony_instrument_override=None):
    os.makedirs(out_dir, exist_ok=True)

    music_dir = os.path.join(out_dir, "music")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(music_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    duration_scale = _validate_duration_scale(duration_scale)

    # 1) Analisi prosodica
    poem_analysis = analyze_poem(text)

    # 2) Analisi semantica (Emotion)
    emotion = analyze_emotion(text)
    print(emotion)

    if verbose:
        print("=== ANALISI PROSODICA E SEMANTICA ===")
        print(f"  Valenza: {emotion['valence']:+.2f} | Arousal: {emotion['arousal']:+.2f} | Tenerezza: {emotion['tenderness']:+.2f}")
        print()

    # 3) MUSIC TRANSFORMER
    mt = MusicTransformer(seed=seed, verbose=verbose)
    melody, harmony, meta = mt.generate(poem_analysis, emotion, text_seed=text)

    if tempo_override is not None:
        meta["tempo"] = _validate_tempo(tempo_override)

    if duration_scale != 1.0:
        apply_duration_scale(melody, harmony, duration_scale)

    # 4) Selezione Automatica degli Strumenti (NLP Zero-Shot)
    auto_lead, auto_pad = choose_by_nlp(text, emotion)
    lead_instrument = melody_instrument_override or auto_lead
    pad_instrument = harmony_instrument_override or auto_pad

    # Generazione dell'Arrangiamento Orchestrale a 4 Parti
    # Deriva basso e arpeggio dall'armonia UNA SOLA VOLTA: la stessa lista di
    # Note viene passata sia a build_midi (traccia MIDI reale) sia a
    # mix_and_export (rendering audio), così .mid e .wav rappresentano
    # esattamente lo stesso arrangiamento a 4 parti invece di due copie
    # calcolate indipendentemente (il bug che stiamo sistemando ora: prima
    # basso/arpeggio finivano SOLO nel .wav, mai nel file .mid esportato).
    bass_notes, arpeggio_notes = derive_bass_and_arpeggio(harmony)
    bass_instrument, arpeggio_instrument = "bass", "guitar"

    if verbose:
        print("=== ORGANICO ORCHESTRALE GENERATO (4 STRUMENTI) ===")
        print(f"  • Lead ({lead_instrument})       | {len(melody.notes)} note")
        print(f"  • Pad ({pad_instrument})        | {len(harmony.chords)} accordi")
        print(f"  • Basso ({bass_instrument})       | {len(bass_notes)} note")
        print(f"  • Arpeggio ({arpeggio_instrument})   | {len(arpeggio_notes)} note")
        print(f"  Tempo: {meta['tempo']} BPM | Modalità: {meta['mode']}\n")

    # Generazione Grafici
#    plot_melody_and_rhythm(melody, poem_analysis, save_path=os.path.join(plots_dir, "melody_rhythm.png"))
    plot_melody_and_rhythm(melody, poem_analysis, save_path=os.path.join(plots_dir, f"{basename}_melody_rhythm.png"))
    print("[DEBUG plot emotion in main]:", emotion)
 #   plot_emotion_space(emotion, meta["mode"], save_path=os.path.join(plots_dir, "emotion_space.png"))
    plot_emotion_space(emotion, meta["mode"], save_path=os.path.join(plots_dir, f"{basename}_emotion_space.png"))

    # 5) MIDI Multitraccia (4 tracce: melodia, armonia, basso, arpeggio)
    midi_path = os.path.join(music_dir, f"{basename}.mid")
    build_midi(melody, harmony, tempo=meta["tempo"],
               melody_instrument=lead_instrument, harmony_instrument=pad_instrument,
               bass_notes=bass_notes, arpeggio_notes=arpeggio_notes,
               bass_instrument=bass_instrument, arpeggio_instrument=arpeggio_instrument,
               out_path=midi_path)

    # 6) Synth / Audio Stereo (stesse 4 tracce di sopra)
    wav_path = os.path.join(music_dir, f"{basename}.wav")
    mix_and_export(melody, harmony, meta["tempo"],
                    melody_instrument=lead_instrument, harmony_instrument=pad_instrument,
                    bass_notes=bass_notes, arpeggio_notes=arpeggio_notes,
                    bass_instrument=bass_instrument, arpeggio_instrument=arpeggio_instrument,
                    out_path=wav_path)

    # 6.1) Synth con FluidSynth
    wav_fluid_path = os.path.join(music_dir, f"{basename}_fluid.wav")
    soundfont_path = os.path.join("fluidsynth", "soundfonts", "MuseScore_General.sf3")
    render_with_fluidsynth(midi_path=midi_path, soundfont_path=soundfont_path, out_path=wav_fluid_path)


    if verbose:
        print("=== OUTPUT GENERATI ===")
        print(f"  MIDI: {midi_path}")
        print(f"  WAV (synth):  {wav_path}")
        print(f"  WAV (FluidSynth):  {wav_fluid_path}")
        print(f"  PLOTS: {plots_dir}")

    return midi_path, wav_path, meta, emotion       # si potrebbe aggiungere anche FluidSynth path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poesia -> Orchestra Multi-Strumento")
    parser.add_argument("poem_file", nargs="?", default=None, help="File di testo della poesia")
    parser.add_argument("--tempo", type=float, default=None, help="BPM della composizione")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Moltiplicatore durata")
    parser.add_argument("--out-dir", default="./output", help="Cartella di output")
    parser.add_argument("--basename", default=None, help="Nome base dei file")
    parser.add_argument("--seed", type=int, default=None, help="Seed casualità")        #casuale
    #per la scelta degli strumenti
    parser.add_argument("--melody-instrument", choices=sorted(INSTRUMENT_PRESETS), default=None, 
                        help="strumento per la voce principale (default: scelto automaticamente dall'NLP)")
    parser.add_argument("--harmony-instrument", choices=sorted(INSTRUMENT_PRESETS), default=None,
                        help="strumento per l'accompagnamento (default: scelto automaticamente dall'NLP)")

    args = parser.parse_args()

    if args.poem_file:
        with open(args.poem_file, "r", encoding="utf-8") as f:
            poem_text = f.read()
        basename=  args.basename or os.path.splitext(os.path.basename(args.poem_file))[0]
    else:
        poem_text = DEMO_POEM
        basename=args.basename or "output"

    try:
        run_pipeline(
            poem_text,
            out_dir=args.out_dir,
            basename=basename,
            seed=args.seed,
            tempo_override=args.tempo,
            duration_scale=args.duration_scale,
            melody_instrument_override=args.melody_instrument,
            harmony_instrument_override=args.harmony_instrument,
        )
    except ValueError as e:
        print(f"[main] Errore: {e}")
        sys.exit(1)