"""
synth.py — Rendering audio stereo multitraccia a 4 strumenti.
"""

import os
import subprocess
import numpy as np
from scipy.io import wavfile
from instruments import INSTRUMENT_PRESETS, DEFAULT_MELODY_INSTRUMENT, DEFAULT_HARMONY_INSTRUMENT

SAMPLE_RATE = 44100


TIMBRE_PROFILES = {
    "piano": {"adsr": dict(a=0.005, d=0.3, s=0.2, r=0.2)},
    "strings": {"adsr": dict(a=0.08, d=0.2, s=0.85, r=0.25)},
    "pluck": {"adsr": dict(a=0.002, d=0.25, s=0.05, r=0.1)},
    "flute": {"adsr": dict(a=0.05, d=0.1, s=0.8, r=0.15)},
    "organ": {"adsr": dict(a=0.01, d=0.02, s=0.95, r=0.05)},
    "brass": {"adsr": dict(a=0.03, d=0.12, s=0.7, r=0.15)},
    "bass": {"adsr": dict(a=0.01, d=0.15, s=0.75, r=0.15)},
}


def midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def adsr_envelope(n_samples, sr=SAMPLE_RATE, a=0.01, d=0.08, s=0.7, r=0.15):
    total_time = n_samples / sr
    req_time = a + d + r

    if req_time > total_time:
        scale = total_time / req_time
        a *= scale
        d *= scale
        r *= scale

    a_n = max(1, int(a * sr))
    d_n = max(1, int(d * sr))
    r_n = max(1, int(r * sr))
    s_n = max(0, n_samples - a_n - d_n - r_n)

    attack = np.linspace(0, 1, a_n)
    decay = np.linspace(1, s, d_n)
    sustain = np.full(s_n, s)
    release = np.linspace(s if s_n > 0 else (decay[-1] if len(decay) > 0 else 1.0), 0, r_n)

    env = np.concatenate([attack, decay, sustain, release])
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]


def render_tone(freq, duration_sec, velocity=80, sr=SAMPLE_RATE, timbre="piano"):
    n = max(1, int(duration_sec * sr))
    t = np.arange(n) / sr
    amp = (velocity / 127.0)

    profile = TIMBRE_PROFILES.get(timbre, TIMBRE_PROFILES["piano"])

    if timbre == "flute":
        vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * freq * np.cumsum(vibrato) / sr
        wave = np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.04 * np.sin(3 * phase)
        breath_noise = (np.random.rand(n) - 0.5) * 0.025
        wave += breath_noise

    elif timbre == "strings":
        detunes = [-0.004, 0.0, 0.004]
        vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
        wave = np.zeros(n)
        for d in detunes:
            f = freq * (1.0 + d)
            phase = 2 * np.pi * f * np.cumsum(vibrato) / sr
            wave += 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase) + 0.05 * np.sin(4 * phase)
        wave /= len(detunes)

    elif timbre == "pluck":
        decay_env = np.exp(-t * (freq * 0.03 + 2.5))
        wave = np.zeros(n)
        for h, hamp in [(1, 1.0), (2, 0.55), (3, 0.35), (4, 0.2), (5, 0.1)]:
            wave += hamp * np.sin(2 * np.pi * freq * h * t)
        wave *= decay_env

    elif timbre == "brass":
        bright = np.linspace(0.5, 1.0, n)
        wave = np.zeros(n)
        for h, hamp in [(1, 1.0), (2, 0.75), (3, 0.55), (4, 0.35), (5, 0.25), (6, 0.15)]:
            wave += (hamp * (bright if h > 2 else 1.0)) * np.sin(2 * np.pi * freq * h * t)

    elif timbre == "organ":
        wave = np.zeros(n)
        for h, hamp in [(1, 1.0), (2, 0.65), (3, 0.5), (4, 0.35), (5, 0.25), (6, 0.15), (8, 0.1)]:
            wave += hamp * np.sin(2 * np.pi * freq * h * t)

    elif timbre == "bass":
        decay_env = np.exp(-t * 0.6)
        wave = np.sin(2 * np.pi * freq * t) + 0.45 * np.sin(2 * np.pi * freq * 2 * t) + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
        wave *= decay_env

    else:  # piano
        decay_env = np.exp(-t * (freq * 0.008 + 1.2))
        wave = np.zeros(n)
        for h, hamp in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]:
            h_freq = freq * h * (1.0 + 0.0004 * (h ** 2))
            wave += hamp * np.sin(2 * np.pi * h_freq * t)
        wave *= decay_env

    env = adsr_envelope(n, sr, **profile["adsr"])
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave / max_val
    return wave * env * amp


def render_track_stereo(events, pan=0.0, sr=SAMPLE_RATE, timbre="piano"):
    total_len = int(sr * (max((s + d for _, s, d, _ in events), default=1.0) + 0.5))
    mono_track = np.zeros(total_len)

    for pitches, start, dur, vel in events:
        if isinstance(pitches, int):
            pitches = [pitches]
        for p in pitches:
            freq = midi_to_freq(p)
            tone = render_tone(freq, dur, velocity=vel, sr=sr, timbre=timbre)
            start_i = int(start * sr)
            end_i = start_i + len(tone)
            if end_i > len(mono_track):
                mono_track = np.pad(mono_track, (0, end_i - len(mono_track)))
            mono_track[start_i:end_i] += tone

    left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
    right_gain = np.sin((pan + 1.0) * np.pi / 4.0)

    stereo_track = np.zeros((len(mono_track), 2))
    stereo_track[:, 0] = mono_track * left_gain
    stereo_track[:, 1] = mono_track * right_gain
    return stereo_track


def mix_and_export(melody, harmony, tempo,
                    melody_instrument=DEFAULT_MELODY_INSTRUMENT,
                    harmony_instrument=DEFAULT_HARMONY_INSTRUMENT,
                    bass_notes=None, arpeggio_notes=None,
                    bass_instrument="bass", arpeggio_instrument="guitar",
                    out_path="output.wav"):
    from instruments import validate_instrument
    melody_instrument = validate_instrument(melody_instrument)
    harmony_instrument = validate_instrument(harmony_instrument)
    melody_timbre = INSTRUMENT_PRESETS[melody_instrument]["timbre"]
    harmony_timbre = INSTRUMENT_PRESETS[harmony_instrument]["timbre"]

    beat_sec = 60.0 / tempo

    # 1. Melodia Lead (Centro)
    mel_events = []
    t = 0.0
    for note in melody.notes:
        dur_sec = note.duration * beat_sec
        mel_events.append((note.pitch, t, dur_sec * 1.02, note.velocity))
        t += dur_sec

    # 2. Tappeto Armonico Pad (Sinistra)
    harm_events = []
    t = 0.0
    for chord in harmony.chords:
        dur_sec = chord.duration * beat_sec
        harm_events.append((chord.pitches, t, dur_sec * 0.98, 45))
        t += dur_sec

    # 3. Basso Profondo (Centro) e 4. Arpeggio/Pizzicato (Destra): se non
    # vengono passati esplicitamente (bass_notes/arpeggio_notes, tipicamente
    # da music_transformer.derive_bass_and_arpeggio — la STESSA fonte usata
    # da midi_builder.py per il file .mid, cosicché .mid e .wav coincidano),
    # si ricade sul calcolo diretto dall'armonia per compatibilità con le
    # chiamate esistenti che non li forniscono.
    bass_timbre = INSTRUMENT_PRESETS[validate_instrument(bass_instrument)]["timbre"]
    arpeggio_timbre = INSTRUMENT_PRESETS[validate_instrument(arpeggio_instrument)]["timbre"]

    bass_events = []
    if bass_notes is not None:
        t = 0.0
        for note in bass_notes:
            dur_sec = note.duration * beat_sec
            bass_events.append(([note.pitch], t, dur_sec * 0.95, note.velocity))
            t += dur_sec
    else:
        t = 0.0
        for chord in harmony.chords:
            dur_sec = chord.duration * beat_sec
            root_pitch = min(chord.pitches) if chord.pitches else 48
            bass_pitch = max(28, root_pitch - 12)
            bass_events.append(([bass_pitch], t, dur_sec * 0.95, 65))
            t += dur_sec

    arpeggio_events = []
    if arpeggio_notes is not None:
        t = 0.0
        for note in arpeggio_notes:
            dur_sec = note.duration * beat_sec
            arpeggio_events.append(([note.pitch], t, dur_sec * 0.85, note.velocity))
            t += dur_sec
    else:
        t = 0.0
        step_sec = 0.5 * beat_sec
        for chord in harmony.chords:
            dur_sec = chord.duration * beat_sec
            n_steps = max(1, int(round(dur_sec / step_sec)))
            actual_step = dur_sec / n_steps
            for i in range(n_steps):
                p = chord.pitches[i % len(chord.pitches)]
                arpeggio_events.append(([p], t + i * actual_step, actual_step * 0.85, 50))
            t += dur_sec

    # Mix Multitraccia Stereo
    tracks = [
        {"events": mel_events, "timbre": melody_timbre, "pan": 0.0, "volume": 0.85},
        {"events": harm_events, "timbre": harmony_timbre, "pan": -0.45, "volume": 0.45},
        {"events": bass_events, "timbre": bass_timbre, "pan": 0.0, "volume": 0.70},
        {"events": arpeggio_events, "timbre": arpeggio_timbre, "pan": 0.50, "volume": 0.50},
    ]

    rendered_tracks = []
    max_len = 0
    for conf in tracks:
        trk = render_track_stereo(conf["events"], pan=conf["pan"], timbre=conf["timbre"])
        trk *= conf["volume"]
        rendered_tracks.append(trk)
        if len(trk) > max_len:
            max_len = len(trk)

    master = np.zeros((max_len, 2))
    for trk in rendered_tracks:
        master[:len(trk), :] += trk

    peak = np.max(np.abs(master))
    if peak > 0:
        master = (master / peak) * 0.92

    audio_i16 = (master * 32767).astype(np.int16)
    wavfile.write(out_path, SAMPLE_RATE, audio_i16)
    return out_path

def _resolve_fluidsynth_executable():
    """
    Determina quale eseguibile fluidsynth usare a seconda del sistema operativo.
 
    - Su Windows: usa il binario vendorizzato nel repo (fluidsynth/bin/fluidsynth.exe),
      perché FluidSynth non ha un installer che lo registra nel PATH di sistema.
    - Su Linux/macOS: usa il comando "fluidsynth" di sistema (es. installato con
      apt/brew), cercandolo esplicitamente nel PATH con shutil.which per dare un
      errore chiaro se manca, invece di lasciare fallire subprocess.run in modo criptico.
    """
    base_dir = os.path.dirname(__file__)
 
    if platform.system() == "Windows":
        bundled = os.path.join(base_dir, "fluidsynth", "bin", "fluidsynth.exe")
        if os.path.exists(bundled):
            return bundled
        # fallback: magari è comunque nel PATH anche su Windows
        found = shutil.which("fluidsynth")
        if found:
            return found
        raise FileNotFoundError(
            "fluidsynth.exe non trovato né in fluidsynth/bin/ né nel PATH. "
            "Scarica FluidSynth per Windows e mettilo in fluidsynth/bin/ nella cartella del progetto."
        )
 
    found = shutil.which("fluidsynth")
    if not found:
        raise FileNotFoundError(
            "Comando 'fluidsynth' non trovato nel PATH. Installalo con il package "
            "manager di sistema (es. 'sudo apt install fluidsynth' su Linux, "
            "'brew install fluid-synth' su macOS)."
        )
    return found

#per la sintesi del file midi usando un sintetizzatore autonomo
def render_with_fluidsynth(midi_path, soundfont_path, out_path="output_fluid.wav", sample_rate=44100):
    base_dir = os.path.dirname(__file__)
    fluidsynth_exe = os.path.join(base_dir, "fluidsynth", "bin", "fluidsynth.exe")

    if not os.path.exists(fluidsynth_exe):
        fluidsynth_exe = "fluidsynth"

    # Le opzioni (-ni, -F, -r) DEVONO stare prima di soundfont_path e midi_path.
    # Avvolto in try/except: FluidSynth è un extra (il synth additivo in
    # mix_and_export produce comunque un WAV valido) — un binario mancante,
    # incompatibile con la piattaforma (es. .exe su Linux/Mac) o un
    # soundfont non trovato non deve far fallire l'intera generazione.
    try:
        subprocess.run([
            fluidsynth_exe,
            "-ni",
            "-F", out_path,
            "-r", str(sample_rate),
            soundfont_path,
            midi_path
        ], check=True)
        return out_path
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[synth] FluidSynth non disponibile o fallito ({e}); uso solo il synth additivo interno.")
        return None

#per la sintesi del file midi usando un sintetizzatore autonomo
#def render_with_fluidsynth(midi_path, soundfont_path, out_path="output_fluid.wav", sample_rate=44100):
#    subprocess.run([
#        "fluidsynth", "-ni", soundfont_path, midi_path,
#        "-F", out_path, "-r", str(sample_rate)
#    ], check=True)
#    return out_path