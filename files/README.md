# Poesia → Musica: pipeline completa (con dati reali)

```
POESIA → analisi semantica (emotion embedding)
       → analisi prosodica: PhonItalia → Q2Stress → euristica → RHYTHM
       → MUSIC TRANSFORMER (guidato da statistiche del corpus Lakh MIDI) → MELODIA + ARMONIA
       → MIDI
       → Synth/Audio (WAV)
```

## Novità rispetto alla prima versione
Tre risorse reali sono state integrate al posto delle euristiche pure:

### 1. `phon_italia.py` — lessico PhonItalia
Da https://github.com/stefanocoretta/phonItaliaR (Goslin, Galluzzi & Romani,
2014) — 120.000 forme di parola italiane con sillabe e accento tonico
**verificati empiricamente**, non dedotti da regole. `prosody.py` lo
interroga per primo: se la parola c'è nel lessico, l'accento è quello vero
(es. "musica" → MU-si-ca, non più l'euristica sbagliata "mu-SI-ca").

**Serve**: la cartella `repo_phonItaliaR/` clonata dal repo (contiene
`data-raw/phonItalia-1.10/phonItalia-1.10-wordforms.tsv`, ~45MB, non incluso
qui per peso — clona il repo e mettilo accanto agli script).

### 2. `q2stress.py` — predittore statistico Q2Stress
Da Spinelli, Sulpizio & Burani (2017), archivio scaricato da istc.cnr.it.
Per le parole **assenti** da PhonItalia (neologismi, forme rare, poetiche),
stima la posizione dell'accento dalla desinenza di 3 lettere, usando le
percentuali reali di parole italiane accentate su ultima/penultima/
terzultima/quartultima sillaba per quella desinenza.

**Serve**: la cartella `repo_q2stress/Q2Stress/summary tables/adults/endings/endings/`
(i 4 file `types_{ccv,vcc,vcv,vvv}.txt`).

### 3. `lakh_midi.py` — statistiche dal Lakh MIDI Dataset
Da https://github.com/ryohey/lakh-midi (sottoinsieme "clean_midi", 17.256
file MIDI reali). Al posto dei pesi melodici inventati a mano, il
`MusicTransformer` ora campiona gli intervalli tra note consecutive dalla
**distribuzione empirica reale** (note ripetute e passi congiunti dominano,
poi quarte/quinte, come in ogni corpus musicale tonale), pesata da
valenza/arousal per orientare la scelta senza abbandonare i pattern
osservati.

**Cache inclusa**: `data/lakh_stats.json` (statistiche già estratte da un
campione di 250 file — funziona subito, senza bisogno del corpus completo).
Per ri-estrarre da più file: `python3 -c "from lakh_midi import *; save_stats(build_stats(max_files=1000))"`
(richiede la cartella `repo_lakh-midi/clean_midi/`).

## Catena di risoluzione dell'accento (prosody.py)
1. accento grafico esplicito nel testo (città, perché...)
2. lookup esatto in PhonItalia (120k parole verificate)
3. predizione statistica Q2Stress (desinenza → posizione più probabile)
4. euristica di default (parola piana) se nessuna delle precedenti risponde

Ogni sillaba tonica riporta la fonte usata (`stress_source`), utile per
debug e per capire quanto ci si sta affidando ai dati reali vs. al fallback.

## Uso
```bash
pip install pretty_midi numpy scipy

# clona le risorse esterne (una tantum, accanto agli script):
git clone https://github.com/stefanocoretta/phonItaliaR.git repo_phonItaliaR
git clone https://github.com/ryohey/lakh-midi.git repo_lakh-midi
# Q2Stress: scarica lo zip da istc.cnr.it, estrai in repo_q2stress/

python3 main.py                 # poesia demo (Dante)
python3 main.py mia_poesia.txt  # una tua poesia, un verso per riga
```

Se una risorsa esterna manca, ogni modulo ricade automaticamente sul
livello successivo della catena (o sull'euristica) e lo segnala a schermo
— la pipeline non si rompe mai, degrada solo la qualità dell'accento/melodia.

## File inclusi in questo pacchetto
- `prosody.py`, `emotion.py`, `music_transformer.py`, `midi_builder.py`,
  `synth.py`, `main.py` — pipeline base (vedi versione precedente)
- `phon_italia.py`, `q2stress.py`, `lakh_midi.py` — i tre nuovi moduli
- `data/lakh_stats.json` — cache delle statistiche Lakh (pronta all'uso)
- `demo_output_v2.mid` / `.wav` — esempio rigenerato con la pipeline completa
  (stessi versi della Divina Commedia, ora con accenti reali PhonItalia e
  melodia guidata da statistiche del corpus Lakh)
