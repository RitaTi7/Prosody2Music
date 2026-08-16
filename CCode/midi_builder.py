"""
midi_builder.py — Costruisce un file .mid a partire da Melody + Harmony.
"""

import pretty_midi as pm
from instruments import INSTRUMENT_PRESETS, DEFAULT_MELODY_INSTRUMENT, DEFAULT_HARMONY_INSTRUMENT, validate_instrument


def build_midi(melody, harmony, tempo=90,
                melody_instrument=DEFAULT_MELODY_INSTRUMENT,
                harmony_instrument=DEFAULT_HARMONY_INSTRUMENT,
                out_path="output.mid"):
    """
    melody_instrument / harmony_instrument: nomi da instruments.INSTRUMENT_PRESETS
    (es. "piano", "guitar", "flute", "organ", "strings", ...)
    """
    melody_instrument = validate_instrument(melody_instrument)
    harmony_instrument = validate_instrument(harmony_instrument)
    melody_program = INSTRUMENT_PRESETS[melody_instrument]["program"]
    harmony_program = INSTRUMENT_PRESETS[harmony_instrument]["program"]

    midi = pm.PrettyMIDI(initial_tempo=tempo)
    beat_sec = 60.0 / tempo

    mel_inst = pm.Instrument(program=melody_program, name=f"Melodia ({melody_instrument})")
    t = 0.0
    for note in melody.notes:
        dur_sec = note.duration * beat_sec
        mel_inst.notes.append(pm.Note(
            velocity=note.velocity, pitch=note.pitch,
            start=t, end=t + dur_sec * 0.95  # piccolo staccato per chiarezza
        ))
        t += dur_sec
    midi.instruments.append(mel_inst)

    harm_inst = pm.Instrument(program=harmony_program, name=f"Armonia ({harmony_instrument})")
    t = 0.0
    for chord in harmony.chords:
        dur_sec = chord.duration * beat_sec
        for pitch in chord.pitches:
            harm_inst.notes.append(pm.Note(
                velocity=55, pitch=pitch, start=t, end=t + dur_sec * 0.98
            ))
        t += dur_sec
    midi.instruments.append(harm_inst)

    midi.write(out_path)
    return out_path


if __name__ == "__main__":
    from prosody import analyze_poem
    from emotion import analyze_emotion
    from music_transformer import MusicTransformer

    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    pa = analyze_poem(demo)
    em = analyze_emotion(demo)
    mt = MusicTransformer(seed=42)
    melody, harmony, meta = mt.generate(pa, em, text_seed=demo)
    path = build_midi(melody, harmony, tempo=meta["tempo"],
                       melody_instrument="flute", harmony_instrument="organ")
    print("scritto:", path)
