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

import sys
import os

from prosody import analyze_poem
from emotion import analyze_emotion
from music_transformer import MusicTransformer
from midi_builder import build_midi
from synth import mix_and_export

DEMO_POEM = """Nel mezzo del cammin di nostra vita
mi ritrovai per una selva oscura
ché la diritta via era smarrita"""


def run_pipeline(text, out_dir=".", basename="output", seed=42, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

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
    mt = MusicTransformer(seed=seed)
    melody, harmony, meta = mt.generate(poem_analysis, emotion, text_seed=text)

    if verbose:
        print("=== MUSIC TRANSFORMER (output) ===")
        print(f"  modalità: {meta['mode']}   tonica MIDI: {meta['root']}   tempo: {meta['tempo']} bpm")
        print(f"  note melodia: {len(melody.notes)}   accordi armonia: {len(harmony.chords)}")
        print()

    # 4) MIDI
    midi_path = os.path.join(out_dir, f"{basename}.mid")
    build_midi(melody, harmony, tempo=meta["tempo"], out_path=midi_path)

    # 5) Synth / Audio
    wav_path = os.path.join(out_dir, f"{basename}.wav")
    mix_and_export(melody, harmony, meta["tempo"], out_path=wav_path)

    if verbose:
        print("=== OUTPUT ===")
        print(f"  MIDI: {midi_path}")
        print(f"  WAV:  {wav_path}")

    return midi_path, wav_path, meta, emotion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            poem_text = f.read()
    else:
        poem_text = DEMO_POEM
        print("(nessun file fornito, uso la poesia demo inclusa)\n")

    run_pipeline(poem_text)
