# Poesia → Musica: pipeline completa

Implementazione della pipeline:

```
POESIA → analisi semantica (emotion embedding)
       → analisi prosodica (sillabe+accenti → RHYTHM)
       → MUSIC TRANSFORMER → MELODIA + ARMONIA
       → MIDI
       → Synth/Audio (WAV)
```

## File
- `prosody.py` — sillabazione italiana, individuazione accento tonico, ritmo
- `emotion.py` — emotion embedding (valenza/arousal/tenerezza) da lessico italiano
- `music_transformer.py` — genera melodia e armonia condizionate da ritmo + emozione
- `midi_builder.py` — costruisce il file .mid (pretty_midi)
- `synth.py` — sintesi additiva in numpy, esporta .wav senza bisogno di soundfont
- `main.py` — orchestratore della pipeline

## Uso
```bash
pip install pretty_midi numpy scipy
python3 main.py                 # usa la poesia demo inclusa (Dante)
python3 main.py mia_poesia.txt  # oppure una tua poesia, un verso per riga
```

Output: `output.mid` e `output.wav` nella cartella corrente.

## Nota sul "Music Transformer"
Il blocco `MusicTransformer` nel diagramma originale implica un vero
modello sequence-to-sequence attention-based addestrato su corpora MIDI.
Non avendo dati di training né un modello pre-addestrato disponibili qui,
`music_transformer.py` implementa una **policy generativa condizionata**
(random-walk pesato su scala musicale, scelta in base a valenza/arousal)
che occupa lo stesso ruolo architetturale nella pipeline: riceve
`(rhythm, emotion_embedding)` e produce `(melody, harmony)`.
L'interfaccia `generate()` è pensata per poter essere sostituita 1:1 da
un vero transformer addestrato (es. Music Transformer di Huang et al.,
o un modello fine-tuned) senza modificare il resto della pipeline.

## Esempio incluso
`demo_output.mid` / `demo_output.wav` — generati dai primi tre versi
della Divina Commedia ("Nel mezzo del cammin di nostra vita...").
