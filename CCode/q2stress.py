"""
q2stress.py — Predittore statistico dell'accento tonico per l'italiano,
basato su Q2Stress (Spinelli, Sulpizio & Burani, 2017), fornito dall'utente
come archivio scaricato da istc.cnr.it.

Q2Stress fornisce, per ogni desinenza di 3 lettere (classificata per
struttura CV: VCV/CCV/VCC/VVV), la percentuale di parole italiane accentate
su:
    ULT    = ultima sillaba      (tronca)
    PULT   = penultima sillaba   (piana)
    APULT  = terzultima sillaba  (sdrucciola)
    PAPULT = quartultima sillaba (bisdrucciola)

Questo modulo è il secondo livello della catena di assegnazione
dell'accento (dopo il lookup esatto in phon_italia.py): per parole assenti
dal lessico PhonItalia, usa questi indizi distribuzionali sulla desinenza
per stimare la posizione più probabile dell'accento, esattamente come
farebbe un lettore umano davanti a una parola sconosciuta.

Cartella sorgente attesa (dall'archivio scaricato):
    repo_q2stress/Q2Stress/summary tables/adults/endings/endings/types_{ccv,vcc,vcv,vvv}.txt

NOVITÀ — dataset Q2Stress a livello di PAROLA
-----------------------------------------------
Oltre alle tabelle per desinenza sopra (usate per il fallback statistico,
livello 4 della catena d'accento in rhythm.py), l'archivio Q2Stress
contiene anche un file a livello di singola parola (lexElem.txt o
"phonItalia 1.10.1 - word forms.txt", con colonne word/StressPattern/
SumSylls), usato invece come dato di ARRICCHIMENTO per il training del
Random Forest in rhythm.py (insieme al lessico di phon_italia.py). Le
funzioni load_wordlevel_dataframe()/find_wordlevel_dataset() qui sotto
sono indipendenti dalle tabelle per desinenza sopra: gestiscono un file
diverso, con un ruolo diverso (training set, non fallback a runtime), ma
vivono nello stesso modulo perché appartengono comunque al dataset
Q2Stress nel suo complesso — evita di sparpagliare la risoluzione dei
percorsi Q2Stress su più file del progetto.
"""

import csv
import glob
import os

DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "Progetto", "Q2Stress", "summary tables", "adults", "endings", "endings",
)

# mappa dalle etichette Q2Stress al nostro schema "distanza dalla fine"
# (coerente con phon_italia.py): 1=ultima, 2=penultima, 3=terzultima, 4=quartultima
_LABEL_TO_DISTANCE = {"ULT": 1, "PULT": 2, "APULT": 3, "PAPULT": 4}

# priori di default per l'italiano quando la desinenza non è nella tabella
# (proporzioni approssimative note in letteratura: piane >> sdrucciole > tronche > bisdrucciole)
DEFAULT_PRIOR = {1: 0.08, 2: 0.70, 3: 0.20, 4: 0.02}

_CUES_CACHE = {}


def load_cues(directory=DEFAULT_DIR, verbose=True):
    """
    Carica e unisce le 4 tabelle di desinenza (ccv, vcc, vcv, vvv) in un
    unico dict: desinenza (3 lettere) -> {1: pct, 2: pct, 3: pct, 4: pct}
    """
    global _CUES_CACHE
    if _CUES_CACHE:
        return _CUES_CACHE

    pattern = os.path.join(directory, "types_*.txt")
    files = [f for f in glob.glob(pattern) if "syllables" not in f and "grammcat" not in f]

    if not files:
        if verbose:
            print(f"[q2stress] nessuna tabella trovata in {directory}")
            print("[q2stress] uso solo i priori di default")
        return {}

    cues = {}
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ending = row.get("ending", "").strip().lower()
                if not ending:
                    continue
                dist = {}
                for label, d in _LABEL_TO_DISTANCE.items():
                    raw = row.get(f"%{label}_Ty", "NA")
                    try:
                        pct = float(raw)
                    except (ValueError, TypeError):
                        pct = 0.0
                    dist[d] = pct
                total = sum(dist.values())
                if total > 0:
                    cues[ending] = {d: v / total for d, v in dist.items()}

    if verbose:
        print(f"[q2stress] caricate {len(cues)} desinenze da {len(files)} tabelle ({directory})")

    _CUES_CACHE = cues
    return cues


def predict_stress_from_end(word: str, cues=None):
    """
    Ritorna (distanza_dalla_fine, confidenza, source) stimando l'accento
    dalla desinenza di 3 lettere. Se la desinenza non è in tabella, ricade
    sui priori di default per l'italiano.
    """
    if cues is None:
        cues = load_cues(verbose=False)

    w = word.lower()
    ending = w[-3:] if len(w) >= 3 else w
    dist = cues.get(ending)

    if dist is None:
        # prova desinenze più corte come ripiego (2 lettere) cercando match parziali
        dist = DEFAULT_PRIOR
        source = "prior_default"
    else:
        source = "q2stress"

    best_distance = max(dist, key=dist.get)
    confidence = dist[best_distance]
    return best_distance, confidence, source


def stress_index_for_syllables(word: str, syllables: list, cues=None):
    """
    Come phon_italia.stress_index_for_syllables, ma via predizione
    statistica invece che lookup esatto. Ritorna sempre un indice (mai
    None), dato che i priori di default coprono ogni caso.
    """
    n = len(syllables)
    if n == 0:
        return 0
    distance, confidence, source = predict_stress_from_end(word, cues)
    idx = n - distance
    idx = max(0, min(n - 1, idx))
    return idx, confidence, source


# ============================================================
# DATASET Q2STRESS A LIVELLO DI PAROLA (per il training del Random
# Forest — vedi nota in testa al file)
# ============================================================
# Stesso pattern di risoluzione percorsi di DEFAULT_DIR sopra e di
# phon_italia.DEFAULT_PATH: parte dalla cartella di questo file e sale
# di un livello (..) per trovare Progetto/, che è dove vivono sia
# phonItaliaR/ sia Q2Stress/ quando gli script stanno in una cartella
# sorella (es. CCode/ e Progetto/ allo stesso livello). Se in futuro la
# struttura cambia ancora, prova anche i percorsi senza risalita, così
# non serve toccare il codice per un semplice spostamento di cartelle —
# e in ultima istanza si può sempre passare un path esplicito.

WORDLEVEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "Progetto", "Q2Stress",
                 "scripts", "children", "lexElem.txt"),
    os.path.join(os.path.dirname(__file__), "..", "Progetto", "Q2Stress",
                 "scripts", "adults", "phonItalia 1.10.1 - word forms.txt"),
    os.path.join(os.path.dirname(__file__), "Q2Stress",
                 "scripts", "children", "lexElem.txt"),
    os.path.join(os.path.dirname(__file__), "Q2Stress",
                 "scripts", "adults", "phonItalia 1.10.1 - word forms.txt"),
    os.path.join(os.path.dirname(__file__), "Progetto", "Q2Stress",
                 "scripts", "children", "lexElem.txt"),
    os.path.join(os.path.dirname(__file__), "Progetto", "Q2Stress",
                 "scripts", "adults", "phonItalia 1.10.1 - word forms.txt"),
]

_WORDLEVEL_DF_CACHE = {"df": None, "loaded": False}


def find_wordlevel_dataset(explicit_path=None):
    """
    Ritorna il path del dataset Q2Stress a livello di parola, o None se
    non trovato in nessuno dei percorsi candidati. explicit_path (se
    passato) ha sempre precedenza e salta la ricerca nei candidati.
    """
    if explicit_path:
        return explicit_path if os.path.isfile(explicit_path) else None
    for path in WORDLEVEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def load_wordlevel_dataframe(explicit_path=None, force_reload=False, verbose=True):
    """
    Ritorna un pandas.DataFrame del dataset Q2Stress a livello di parola
    (colonne attese: word, StressPattern, SumSylls), o None se il file
    non è stato trovato. Cache a livello di modulo (come load_cues sopra
    e phon_italia.load_lexicon), non ricarica da disco a ogni chiamata.

    NB: richiede pandas, ma l'import è locale a questa funzione — chi usa
    solo il fallback per desinenza (load_cues/predict_stress_from_end,
    che usano solo csv/glob della standard library) non paga il costo di
    un import extra.
    """
    global _WORDLEVEL_DF_CACHE
    if not force_reload and _WORDLEVEL_DF_CACHE["loaded"]:
        return _WORDLEVEL_DF_CACHE["df"]

    try:
        import pandas as pd
    except ImportError:
        if verbose:
            print("[q2stress] pandas non installato: impossibile caricare il dataset word-level")
        _WORDLEVEL_DF_CACHE = {"df": None, "loaded": True}
        return None

    path = find_wordlevel_dataset(explicit_path)
    df = None

    if path is None:
        if verbose:
            print("[q2stress] nessun dataset word-level trovato. Percorsi controllati:")
            for candidate in WORDLEVEL_CANDIDATES:
                print(f"  - {os.path.normpath(candidate)}")
            print("[q2stress] il Random Forest verrà addestrato solo sul lessico phon_italia (se disponibile)")
    else:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(path, sep="\t", engine="python", encoding=encoding, on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                continue
        if verbose:
            if df is not None:
                print(f"[q2stress] dataset word-level caricato: {os.path.normpath(path)} ({len(df)} righe)")
            else:
                print(f"[q2stress] trovato {path} ma non è stato possibile leggerlo (encoding sconosciuto)")

    _WORDLEVEL_DF_CACHE = {"df": df, "loaded": True}
    return df


if __name__ == "__main__":
    cues = load_cues()
    tests = ["gattino", "sconosciuto", "xyzabc", "meraviglioso", "pensavano", "caffè"]
    for w in tests:
        idx_info = stress_index_for_syllables(w, list(range(3)), cues)  # placeholder sillabe
        print(w, "->", predict_stress_from_end(w, cues))

    print()
    df = load_wordlevel_dataframe()
    if df is not None:
        print(df.head())
