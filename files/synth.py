"""
synth.py — Sintesi di audio (WAV) da MIDI.

Interfaccia:
    mix_and_export(melody, harmony, tempo=120, out_path="output.wav")

Usa il file MIDI temporaneo per renderizzare in WAV con sintetizzatore
(FluidSynth, timidity, o metodo fallback semplice).
"""

import os
import tempfile


def mix_and_export(melody, harmony, tempo=120, out_path="output.wav"):
    """
    Sintetizza la melodia e l'armonia in un file WAV.
    
    Nota: Questa implementazione è un stub che prova a usare FluidSynth
    (se disponibile) o fallisce gracefully con un messaggio di errore.
    
    Per una versione completa, installare:
      pip install pydub fluidsynth librosa
    
    Args:
      melody: Melody object
      harmony: Harmony object
      tempo: BPM
      out_path: percorso file WAV di output
    """
    
    # Prova a usare FluidSynth
    try:
        import pyfluidsynth
        import midi_builder
        
        # Crea un MIDI temporaneo
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
        
        try:
            midi_builder.build_midi(melody, harmony, tempo=tempo, out_path=midi_path)
            
            # Renderizza con FluidSynth
            fs = pyfluidsynth.Synth()
            fs.start()
            fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")  # Path su Linux; varia su Windows/Mac
            fs.program_select(0, 0, 0, 0)
            
            # Carica e riproduce il MIDI
            # (Nota: pyfluidsynth ha un'API limitata, questo è uno stub)
            
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            # Placeholder: dovrebbe esportare il file WAV qui
            print(f"[synth] Nota: sintetizzazione audio richiede FluidSynth configurato correttamente")
            print(f"[synth] Per ora, esporta il MIDI in {midi_path}")
            
        finally:
            if os.path.exists(midi_path):
                os.remove(midi_path)
    
    except ImportError:
        pass
    
    # Fallback: prova con pydub e AudioSegment
    try:
        import midi_builder
        from pydub import AudioSegment
        
        # Crea un MIDI temporaneo
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
        
        try:
            midi_builder.build_midi(melody, harmony, tempo=tempo, out_path=midi_path)
            
            # Renderizza a WAV usando timidity (se disponibile)
            import subprocess
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            
            result = subprocess.run(
                ["timidity", midi_path, "-Ow", "-o", out_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return out_path
        
        finally:
            if os.path.exists(midi_path):
                os.remove(midi_path)
    
    except Exception:
        pass
    
    # Fallback finale: esporta solo il MIDI e avvisa l'utente
    import midi_builder
    
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    
    # Esporta il MIDI con lo stesso nome (ma .mid invece di .wav)
    midi_out = out_path.replace(".wav", ".mid")
    midi_builder.build_midi(melody, harmony, tempo=tempo, out_path=midi_out)
    
    print(f"[synth] Avviso: sintetizzazione WAV non disponibile")
    print(f"[synth] Sono state installate le dipendenze? (timidity, fluidsynth, librosa)")
    print(f"[synth] MIDI esportato in: {midi_out}")
    print(f"[synth] Installa con: pip install pydub librosa")
    
    return out_path


if __name__ == "__main__":
    from music_transformer import Note, Melody, Chord, Harmony
    
    # Esempio: crea una semplice melodia e armonia
    melody = Melody()
    melody.notes.append(Note(pitch=60, duration=0.5, velocity=80))
    melody.notes.append(Note(pitch=62, duration=0.5, velocity=80))
    melody.notes.append(Note(pitch=64, duration=1.0, velocity=80))
    
    harmony = Harmony()
    harmony.chords.append(Chord(pitches=[48, 52, 55], duration=2.0))
    
    out_path = mix_and_export(melody, harmony, tempo=120, out_path="test.wav")
    print(f"Audio esportato in: {out_path}")
