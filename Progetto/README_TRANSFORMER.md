# Poetry -> Rhythm -> Melody Transformer

## 1. Installazione

```bash
pip install -r requirements.txt
```

## 2. Generazione dello scheletro ritmico

```bash
python DaTestoAMIDI.py
```

Il programma chiede interattivamente testo, BPM/durata e usa PhonItaliaR/Q2Stress se presenti nelle cartelle `phonitaliaR/` e `Q2Stress/`.

Produce:

- `output/rhythmic_skeleton.mid`
- `output/semantic_profile.json`
- `output/music_condition.json`

## 3. Preparazione del dataset musicale

Scaricare Lakh MIDI Dataset e indicare la cartella che contiene i `.mid`:

```bash
python prepare_dataset.py /percorso/Lakh_MIDI --output dataset/rhythm_melody.npz --max-files 20000
```

Per una prova rapida:

```bash
python prepare_dataset.py /percorso/Lakh_MIDI --output dataset/rhythm_melody.npz --max-files 1000
```

## 4. Training

```bash
python train.py --dataset dataset/rhythm_melody.npz --epochs 20
```

Il checkpoint viene salvato in:

`models/melody_transformer.pt`

## 5. Generazione

```bash
python RythmToMelody.py
```

Scegli:

1. genere
2. livello ritmico
3. accompagnamento
4. Random Walk oppure Transformer
5. seed opzionale

Il Transformer è il primo modello `rhythm -> pitch`: il pretraining musicale usa caratteristiche ritmiche estratte dai MIDI. I valori semantici sono già previsti nell'input del modello, ma per farli apprendere realmente serve il successivo fine-tuning su coppie poesia/melodia annotate.
