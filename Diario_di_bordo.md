### Fase1.py
Sono state prodotte 6 versioni di questo file con le seguenti problematiche:
Versioni:
1. Non contava bene gli iati (caso limite che purtroppo si verifica spesso)
2. Funzionava ma veniva data troppa importanza alla recall (che prediligeva dei risultati da un gruppo con una bassa precision)
3. Aggiunta della parte MIDI, ma non risolveva l'errore della versione precedente
4. Diminuito il class_weight="balanced" (skitlearn lo dava pieno), adesso predice molto bene (da accertarsi sia vero)
5. Aggiunta stampa diagnostica finale per vedere la durata effettiva del MIDI
6. MIDI prodotto con note di batteria! Aggiunta del tag --drum (se lasciato senza, usa le note C4 e A3 del piano)

per eseguirlo in modo ok:
python3 Fase1.py frase.txt --midi-output frase.mid --rhythm-output frase.csv --drums

Sembra che produca il ritmo in modo fedele alla prosodia!!


**Prossimo passo**: passare lo schema ritmico nel Transformer.

Può tornare utile l'analisi semantica che è stata silenziata in questa fase ma che può essere riattivata
in questo modo: --semantics

```txt
Sull'analisi semantica — l'ho tenuta ma spenta di default (--semantics per attivarla), e penso sia la scelta giusta per questa fase, per un paio di motivi concreti:

Lo scheletro ritmico dipende solo da accento, sillabe e pause: nessuna di queste feature usa la semantica, quindi non ti serve per l'obiettivo che hai descritto ora.
La similarità semantica richiede it_core_news_md (vettori pieni) — con it_core_news_sm resta comunque a 0.0, quindi va usata con il modello giusto o è inutile.
Il lessico di polarità nello script è manuale e coprirà solo una piccola frazione delle parole reali dei vostri testi; usarlo ora per pesare il ritmo rischierebbe di introdurre bias arbitrari proprio nella fase in cui volete che il ritmo segua fedelmente la prosodia.
```

RT
---------------

Test importante: pyphen non è affidabile come fonte primaria per l'italiano — le sue regole sono pensate per l'a-capo tipografico (TeX), non per la fonetica, quindi sbaglia sistematicamente proprio i casi di iato che il commento del file promette di risolvere (poesia → poe-sia, dovrebbe essere po-e-si-a; verificato contro il conteggio reale PhonItalia).

Conferma sistematica: pyphen sbaglia quasi sempre proprio sui casi di iato (7 errori su 13, tutti sistematicamente sottostimati). Verifico che la mia euristica attuale (che gestisce esplicitamente iato vs dittongo) se la cavi meglio su questi stessi casi.

Quindi non uso pyphen come fonte primaria, lo integro invece come seconda opzione più un meccanismo di riconciliazione che sfrutta il conteggio reale PhonItalia (SumSylls) già disponibile per correggere i casi sbagliati (es. "poesia", "farmacia").

Benefici concreti per la pipeline
Melodia più fedele al testo: il numero di sillabe determina direttamente il numero di note per parola in music_transformer.py — un conteggio sbagliato produceva un ritmo musicale disallineato dalla prosodia reale.
Meno falsi negativi su parole con iato (paese, poesia, farmacia, via, zio...), molto comuni nel lessico poetico italiano.
Robustezza a cascata: se manca PhonItalia o pyphen, il codice degrada comunque alla sola euristica — niente si rompe.

Durante la verifica ho trovato un caso interessante: "ritrovai" nel primo verso di Dante. PhonItalia lo annota come ri-tro-va-i (4 sillabe, pronuncia di citazione della parola isolata), ma nell'endecasillabo dantese va scandito con "vai" come un'unica sillaba (sinalefe/sineresi) per tornare a 11 sillabe. È una divergenza reale tra fonetica lessicale e metrica poetica — un problema di scansione dei versi che va oltre questa integrazione (richiederebbe un modulo dedicato alle regole di sinalefe/dieresi).

EC
--------------------
