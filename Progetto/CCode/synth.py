"""
synth.py — Synth / Audio: renderizza Melody + Harmony in un file WAV.

Sintesi additiva semplice (parziali armoniche + inviluppo ADSR) in numpy
puro, così la pipeline produce audio ascoltabile senza dipendere da
soundfont/fluidsynth esterni (spesso non disponibili in ambienti sandbox).
"""

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100


def midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def adsr_envelope(n_samples, sr=SAMPLE_RATE, a=0.01, d=0.08, s=0.7, r=0.15):
    a_n = int(a * sr)
    d_n = int(d * sr)
    r_n = int(r * sr)
    s_n = max(0, n_samples - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, max(a_n, 1)),
        np.linspace(1, s, max(d_n, 1)),
        np.full(max(s_n, 0), s),
        np.linspace(s, 0, max(r_n, 1)),
    ])
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]


def render_tone(freq, duration_sec, velocity=80, sr=SAMPLE_RATE, timbre="piano"):
    n = max(1, int(duration_sec * sr))
    t = np.arange(n) / sr
    amp = (velocity / 127.0)

    if timbre == "piano":
        # somma di armoniche con decadimento tipico percussivo
        harmonics = [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]
        wave = np.zeros(n)
        for h, hamp in harmonics:
            wave += hamp * np.sin(2 * np.pi * freq * h * t)
        env = adsr_envelope(n, sr, a=0.005, d=0.25, s=0.25, r=0.25)
    else:  # "pad" per l'armonia: più morbido, sostenuto
        harmonics = [(1, 1.0), (2, 0.3), (3, 0.1)]
        wave = np.zeros(n)
        for h, hamp in harmonics:
            wave += hamp * np.sin(2 * np.pi * freq * h * t)
        env = adsr_envelope(n, sr, a=0.08, d=0.2, s=0.8, r=0.3)

    wave = wave / max(np.max(np.abs(wave)), 1e-9)
    return wave * env * amp


def render_track(events, sr=SAMPLE_RATE, timbre="piano"):
    """
    events: lista di (pitch_or_list, start_sec, duration_sec, velocity)
    pitch_or_list può essere un intero (nota singola) o lista (accordo)
    """
    total_len = int(sr * (max((s + d for _, s, d, _ in events), default=1.0) + 0.5))
    track = np.zeros(total_len)
    for pitches, start, dur, vel in events:
        if isinstance(pitches, int):
            pitches = [pitches]
        for p in pitches:
            freq = midi_to_freq(p)
            tone = render_tone(freq, dur, velocity=vel, sr=sr, timbre=timbre)
            start_i = int(start * sr)
            end_i = start_i + len(tone)
            if end_i > len(track):
                track = np.pad(track, (0, end_i - len(track)))
            track[start_i:end_i] += tone
    return track


def mix_and_export(melody, harmony, tempo, out_path="output.wav"):
    beat_sec = 60.0 / tempo

    mel_events = []
    t = 0.0
    for note in melody.notes:
        dur_sec = note.duration * beat_sec
        mel_events.append((note.pitch, t, dur_sec * 0.92, note.velocity))
        t += dur_sec

    harm_events = []
    t = 0.0
    for chord in harmony.chords:
        dur_sec = chord.duration * beat_sec
        harm_events.append((chord.pitches, t, dur_sec * 0.98, 45))
        t += dur_sec

    mel_track = render_track(mel_events, timbre="piano")
    harm_track = render_track(harm_events, timbre="pad")

    n = max(len(mel_track), len(harm_track))
    mel_track = np.pad(mel_track, (0, n - len(mel_track)))
    harm_track = np.pad(harm_track, (0, n - len(harm_track)))

    mix = mel_track * 0.75 + harm_track * 0.5
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9  # normalizzazione, evita clipping

    audio_i16 = (mix * 32767).astype(np.int16)
    wavfile.write(out_path, SAMPLE_RATE, audio_i16)
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
    path = mix_and_export(melody, harmony, meta["tempo"])
    print("scritto:", path)
