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
"""

import csv
import glob
import os

DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../Progetto/Q2Stress", "Q2Stress", "summary tables", "adults", "endings", "endings",
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


if __name__ == "__main__":
    cues = load_cues()
    tests = ["gattino", "sconosciuto", "xyzabc", "meraviglioso", "pensavano", "caffè"]
    for w in tests:
        idx_info = stress_index_for_syllables(w, list(range(3)), cues)  # placeholder sillabe
        print(w, "->", predict_stress_from_end(w, cues))
