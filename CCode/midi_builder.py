"""
midi_builder.py — Costruisce file .mid e trasmette flussi MIDI in tempo reale
utilizzando il protocollo MIDI nativo (Mido) con canali dedicati, Program Change e Control Change.
"""

import time
import random
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from instruments import INSTRUMENT_PRESETS, DEFAULT_MELODY_INSTRUMENT, DEFAULT_HARMONY_INSTRUMENT, validate_instrument
import argparse

TICKS_PER_BEAT = 480  # Risoluzione MIDI standard

'''
def _write_melodic_track(mid, notes, channel, program, name, humanize=True,
                          base_velocity_jitter=(-3, 3), articulation=0.92):        #prima: articulation=0.92
    """Scrive una traccia monofonica (una nota alla volta) sul canale MIDI
    dato: usata sia per la melodia sia — dopo questa modifica — per basso
    e arpeggio, che hanno la stessa forma (lista di Note in sequenza)."""
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('track_name', name=name, time=0))
    track.append(Message('program_change', program=program, channel=channel, time=0))
    if channel == 0:
        track.append(Message('control_change', channel=channel, control=11, value=127, time=0))

    last_rest_ticks = 0
    for note in notes:
        duration_ticks = int(note.duration * TICKS_PER_BEAT)
        vel = note.velocity
        if humanize:
            vel = max(1, min(127, vel + random.randint(*base_velocity_jitter)))

        play_ticks = int(duration_ticks * articulation)
        rest_ticks = duration_ticks - play_ticks

        track.append(Message('note_on', note=note.pitch, velocity=vel, channel=channel, time=last_rest_ticks))
        track.append(Message('note_off', note=note.pitch, velocity=0, channel=channel, time=play_ticks))

        last_rest_ticks = rest_ticks
    return track
'''

def _write_melodic_track(mid, notes, channel, program, name, humanize=True,
                          base_velocity_jitter=(-3, 3), articulation=0.92,
                          legato_overlap_ticks=0):
    """Scrive una traccia monofonica sul canale MIDI dato.

    legato_overlap_ticks: se >0, il note_off di ogni nota viene ritardato
    OLTRE l'inizio della nota successiva, così le due note si sovrappongono
    per un istante invece di susseguirsi in modo staccato — è la vera
    differenza rispetto ad `articulation`, che toglie solo il silenzio TRA
    le note ma non le fa mai suonare insieme. 0 = comportamento originale.
    """
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('track_name', name=name, time=0))
    track.append(Message('program_change', program=program, channel=channel, time=0))
    if channel == 0:
        track.append(Message('control_change', channel=channel, control=11, value=127, time=0))

    # Eventi in tempo ASSOLUTO: con l'overlap il note_off di una nota può
    # cadere dopo il note_on della successiva, quindi non possiamo più
    # scriverli in sequenza stretta come prima — li accumuliamo e ordiniamo.
    events = []
    t = 0
    for note in notes:
        duration_ticks = int(note.duration * TICKS_PER_BEAT)
        vel = note.velocity
        if humanize:
            vel = max(1, min(127, vel + random.randint(*base_velocity_jitter)))

        play_ticks = int(duration_ticks * articulation)
        note_off_time = t + play_ticks + legato_overlap_ticks

        events.append((t, 1, Message('note_on', note=note.pitch, velocity=vel, channel=channel, time=0)))
        events.append((note_off_time, 0, Message('note_off', note=note.pitch, velocity=0, channel=channel, time=0)))
        t += duration_ticks

    # A parità di tick, note_off prima di note_on (evita un silenzio
    # spurio quando non c'è overlap, cioè comportamento originale)
    events.sort(key=lambda e: (e[0], e[1]))

    last_time = 0
    for abs_time, _, msg in events:
        msg.time = max(0, abs_time - last_time)
        track.append(msg)
        last_time = abs_time
    return track

def build_midi(melody, harmony, tempo=90,
               melody_instrument=DEFAULT_MELODY_INSTRUMENT,
               harmony_instrument=DEFAULT_HARMONY_INSTRUMENT,
               bass_notes=None, arpeggio_notes=None,
               bass_instrument="bass", arpeggio_instrument="guitar",
               out_path="output.mid",
               humanize=True):
    """
    Costruisce un file MIDI standard (.mid) generando la struttura binaria a basso livello
    tramite messaggi di protocollo MIDI (Program Change, Control Change, Note On/Off).

    bass_notes e arpeggio_notes sono opzionali (liste di Note, tipicamente
    ottenute da music_transformer.derive_bass_and_arpeggio): se forniti,
    aggiungono una 3a e 4a traccia (canali 2 e 3), così il .mid rispecchia
    lo stesso arrangiamento a 4 parti che synth.py già renderizza nel .wav
    — prima queste due voci esistevano solo nell'audio, non nel file MIDI.
    """
    melody_instrument = validate_instrument(melody_instrument)
    harmony_instrument = validate_instrument(harmony_instrument)
    melody_program = INSTRUMENT_PRESETS[melody_instrument]["program"]
    harmony_program = INSTRUMENT_PRESETS[harmony_instrument]["program"]

    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # --- TRACCIA 0: Meta-informazioni (Tempo e Signature) ---
    meta_track = MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(MetaMessage('track_name', name='Tempo Track', time=0))
    meta_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

    # --- TRACCIA 1: Melodia (Canale MIDI 0) ---
    '''
    _write_melodic_track(mid, melody.notes, channel=0, program=melody_program,
                          name=f"Melodia ({melody_instrument})", humanize=humanize)
    '''
    _write_melodic_track(mid, melody.notes, channel=0, program=melody_program,
                      name=f"Melodia ({melody_instrument})", humanize=humanize,
                      articulation=1.0, legato_overlap_ticks=100)               #overlap 40 è circa 1/12 di battito di sovrapposizione (legato leggero)
                                                                                #con 80-100 è più marcato- Se non inserito è zero (comportamento di prima)
    
    # --- TRACCIA 2: Armonia (Canale MIDI 1) ---
    harm_track = MidiTrack()
    mid.tracks.append(harm_track)
    harm_track.append(MetaMessage('track_name', name=f"Armonia ({harmony_instrument})", time=0))
    
    # Protocollo MIDI: Program Change (imposta strumento su Canale 1)
    harm_track.append(Message('program_change', program=harmony_program, channel=1, time=0))

    for chord in harmony.chords:
        dur_ticks = int(chord.duration * TICKS_PER_BEAT)

        # Protocollo MIDI: CC 64 (Sustain Pedal ON) all'inizio dell'accordo
        harm_track.append(Message('control_change', channel=1, control=64, value=127, time=0))

        # Attivazione Note On per tutte le voci dell'accordo
        for i, pitch in enumerate(chord.pitches):
            vel = 55 + (random.randint(-3, 3) if humanize else 0)
            harm_track.append(Message('note_on', note=pitch, velocity=vel, channel=1, time=0))

        hold_ticks = int(dur_ticks * 0.95)
        release_ticks = dur_ticks - hold_ticks

        # Disattivazione Note Off per le voci dell'accordo
        for i, pitch in enumerate(chord.pitches):
            dt = hold_ticks if i == 0 else 0
            harm_track.append(Message('note_off', note=pitch, velocity=0, channel=1, time=dt))

        # Protocollo MIDI: CC 64 (Sustain Pedal OFF) prima del cambio accordo
        harm_track.append(Message('control_change', channel=1, control=64, value=0, time=release_ticks))

    # --- TRACCIA 3: Basso (Canale MIDI 2) ---
    if bass_notes:
        bass_instrument = validate_instrument(bass_instrument)
        _write_melodic_track(mid, bass_notes, channel=2,
                              program=INSTRUMENT_PRESETS[bass_instrument]["program"],
                              name=f"Basso ({bass_instrument})", humanize=humanize,
                              base_velocity_jitter=(-3, 3), articulation=0.95)

    # --- TRACCIA 4: Arpeggio (Canale MIDI 3) ---
    if arpeggio_notes:
        arpeggio_instrument = validate_instrument(arpeggio_instrument)
        _write_melodic_track(mid, arpeggio_notes, channel=3,
                              program=INSTRUMENT_PRESETS[arpeggio_instrument]["program"],
                              name=f"Arpeggio ({arpeggio_instrument})", humanize=humanize,
                              base_velocity_jitter=(-2, 2), articulation=0.85)

    mid.save(out_path)
    return out_path


def stream_live_midi(melody, harmony, tempo=90,
                     melody_instrument=DEFAULT_MELODY_INSTRUMENT,
                     harmony_instrument=DEFAULT_HARMONY_INSTRUMENT,
                     port_name=None):
    """
    Trasmette i pacchetti di byte del protocollo MIDI in TEMPO REALE
    verso una porta MIDI del sistema (DAW, sintetizzatore hardware o virtuale).
    """
    melody_instrument = validate_instrument(melody_instrument)
    harmony_instrument = validate_instrument(harmony_instrument)
    melody_program = INSTRUMENT_PRESETS[melody_instrument]["program"]
    harmony_program = INSTRUMENT_PRESETS[harmony_instrument]["program"]

    beat_sec = 60.0 / tempo

    try:
        outport = mido.open_output(port_name) if port_name else mido.open_output()
        print(f"[MIDI Real-Time] Connesso alla porta: '{outport.name}'")
    except Exception as e:
        print(f"[MIDI Real-Time] Avviso porta fisica ({e}). Apertura porta virtuale...")
        outport = mido.open_output('Prosody2Music Output', virtual=True)

    with outport:
        # Program Change iniziale
        outport.send(Message('program_change', program=melody_program, channel=0))
        outport.send(Message('program_change', program=harmony_program, channel=1))
        outport.send(Message('control_change', channel=0, control=11, value=127))

        print("[MIDI Real-Time] Streaming in corso...")
        for note in melody.notes:
            dur_sec = note.duration * beat_sec
            play_sec = dur_sec * 0.92
            rest_sec = dur_sec - play_sec

            outport.send(Message('note_on', note=note.pitch, velocity=note.velocity, channel=0))
            time.sleep(play_sec)
            outport.send(Message('note_off', note=note.pitch, velocity=0, channel=0))
            time.sleep(rest_sec)

        print("[MIDI Real-Time] Trasmissione completata.")


#per cambiare lo strumento da terminale (quando esegui questo fil eda solo)
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Genera musica da un testo poetico")
    parser.add_argument("text_file", nargs="?", help="file di testo/poesia in input")
    parser.add_argument("--melody-instrument", choices=sorted(INSTRUMENT_PRESETS),
                         default=DEFAULT_MELODY_INSTRUMENT,
                         help="strumento per la voce principale (melodia)")
    parser.add_argument("--harmony-instrument", choices=sorted(INSTRUMENT_PRESETS),
                         default=DEFAULT_HARMONY_INSTRUMENT,
                         help="strumento per l'accompagnamento armonico")
    parser.add_argument("--out", default="output.mid", help="percorso file MIDI di output")
    return parser.parse_args()
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Genera musica da un testo poetico")
    parser.add_argument("text_file", nargs="?", help="file di testo/poesia in input")
    parser.add_argument("--melody-instrument", choices=sorted(INSTRUMENT_PRESETS),
                         default="flute",
                         help="strumento per la voce principale (melodia)")
    parser.add_argument("--harmony-instrument", choices=sorted(INSTRUMENT_PRESETS),
                         default="organ",
                         help="strumento per l'accompagnamento armonico")
    parser.add_argument("--out", default="output.mid", help="percorso file MIDI di output")
    return parser.parse_args()

'''

if __name__ == "__main__":
    from prosody import analyze_poem
    from emotion import analyze_emotion
    from music_transformer import MusicTransformer

    demo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    pa = analyze_poem(demo)
    em = analyze_emotion(demo)
    mt = MusicTransformer(seed=42)
    melody, harmony, meta = mt.generate(pa, em, text_seed=demo)

    # 1. Generazione del file MIDI
    path = build_midi(melody, harmony, tempo=meta["tempo"],
                      melody_instrument="flute", harmony_instrument="organ")
    print("File MIDI generato con protocollo Mido:", path)
'''

#!!! serve solo nel caso in cui si voglia eseguire questo script da solo
# python midi_builder.py poesia.txt --melody-instrument clarinet --harmony-instrument organ --out mia_canzone.mid 
if __name__ == "__main__":
    from prosody import analyze_poem
    from emotion import analyze_emotion
    from music_transformer import MusicTransformer

    args = parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            testo = f.read()
    else:
        testo = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"

    pa = analyze_poem(testo)
    em = analyze_emotion(testo)
    mt = MusicTransformer(seed=42)
    melody, harmony, meta = mt.generate(pa, em, text_seed=testo)

    path = build_midi(melody, harmony, tempo=meta["tempo"],
                      melody_instrument=args.melody_instrument,
                      harmony_instrument=args.harmony_instrument,
                      out_path=args.out)
    print("File MIDI generato con protocollo Mido:", path)
    



    # 2. (Opzionale) Decommenta per trasmettere il flusso MIDI dal vivo
    # stream_live_midi(melody, harmony, tempo=meta["tempo"],
    #                  melody_instrument="flute", harmony_instrument="organ")