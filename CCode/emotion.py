"""
emotion.py — Analisi semantica del testo: estrazione di emotion embedding
(valenza, arousal, tenerezza) da un corpus di parole.

Usa un lessico semplice di parole italiane associate a emotion scores,
con lookup case-insensitive e stemming di base per robustezza.

Ritorna: {"valence": float, "arousal": float, "tenderness": float, "matched": list}
  - valence: -1 (negativo) a +1 (positivo)
  - arousal: 0 (calmo) a +1 (eccitato)
  - tenderness: 0 (neutro) a +1 (tenerezza/dolcezza)
"""

import re

# Lessico semantico italiano: parola -> {"valence": float, "arousal": float, "tenderness": float}
# Valori normalizzati in [-1, +1] per valenza e [0, 1] per arousal e tenerezza
EMOTION_LEXICON = {
    # positivo
    "amore": {"valence": 0.9, "arousal": 0.7, "tenderness": 1.0},
    "bella": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.6},
    "bellissima": {"valence": 0.9, "arousal": 0.4, "tenderness": 0.7},
    "bello": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.5},
    "buono": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.5},
    "caldo": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.7},
    "caro": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.9},
    "cielo": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.4},
    "cuore": {"valence": 0.8, "arousal": 0.6, "tenderness": 0.8},
    "dolce": {"valence": 0.8, "arousal": 0.2, "tenderness": 0.9},
    "dolcezza": {"valence": 0.8, "arousal": 0.2, "tenderness": 1.0},
    "felice": {"valence": 0.9, "arousal": 0.6, "tenderness": 0.5},
    "felicità": {"valence": 0.9, "arousal": 0.6, "tenderness": 0.5},
    "fiore": {"valence": 0.7, "arousal": 0.1, "tenderness": 0.8},
    "gioia": {"valence": 0.9, "arousal": 0.7, "tenderness": 0.4},
    "gioioso": {"valence": 0.9, "arousal": 0.7, "tenderness": 0.3},
    "grazia": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.8},
    "grazioso": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.8},
    "luce": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.5},
    "lucente": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.4},
    "luminoso": {"valence": 0.8, "arousal": 0.4, "tenderness": 0.4},
    "luna": {"valence": 0.7, "arousal": 0.1, "tenderness": 0.7},
    "mare": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.5},
    "meraviglia": {"valence": 0.8, "arousal": 0.6, "tenderness": 0.3},
    "meraviglioso": {"valence": 0.9, "arousal": 0.6, "tenderness": 0.3},
    "pace": {"valence": 0.8, "arousal": 0.0, "tenderness": 0.6},
    "pacifico": {"valence": 0.8, "arousal": 0.0, "tenderness": 0.6},
    "paradiso": {"valence": 0.9, "arousal": 0.5, "tenderness": 0.5},
    "paradisiaco": {"valence": 0.9, "arousal": 0.4, "tenderness": 0.4},
    "piacere": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.4},
    "piacevole": {"valence": 0.8, "arousal": 0.2, "tenderness": 0.5},
    "pratica": {"valence": 0.3, "arousal": 0.3, "tenderness": 0.0},
    "riso": {"valence": 0.8, "arousal": 0.7, "tenderness": 0.3},
    "ridere": {"valence": 0.8, "arousal": 0.7, "tenderness": 0.2},
    "ridente": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.3},
    "rosa": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.8},
    "rosso": {"valence": 0.6, "arousal": 0.6, "tenderness": 0.2},
    "sognante": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.7},
    "sognare": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.6},
    "sogno": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.6},
    "sole": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.4},
    "solenne": {"valence": 0.5, "arousal": 0.3, "tenderness": 0.2},
    "solerzia": {"valence": 0.6, "arousal": 0.6, "tenderness": 0.2},
    "sorriso": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.6},
    "sorridere": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.6},
    "sorridente": {"valence": 0.8, "arousal": 0.3, "tenderness": 0.6},
    "splendente": {"valence": 0.8, "arousal": 0.4, "tenderness": 0.3},
    "splendido": {"valence": 0.9, "arousal": 0.4, "tenderness": 0.3},
    "stella": {"valence": 0.7, "arousal": 0.1, "tenderness": 0.6},
    "stellato": {"valence": 0.7, "arousal": 0.1, "tenderness": 0.5},
    "tenerezza": {"valence": 0.7, "arousal": 0.2, "tenderness": 1.0},
    "tenero": {"valence": 0.7, "arousal": 0.1, "tenderness": 0.9},
    "tenuità": {"valence": 0.5, "arousal": 0.1, "tenderness": 0.6},
    "tenue": {"valence": 0.5, "arousal": 0.1, "tenderness": 0.6},
    "tranquillità": {"valence": 0.6, "arousal": 0.0, "tenderness": 0.5},
    "tranquillo": {"valence": 0.6, "arousal": 0.0, "tenderness": 0.4},
    "trionfale": {"valence": 0.8, "arousal": 0.8, "tenderness": 0.1},
    "trionfo": {"valence": 0.8, "arousal": 0.8, "tenderness": 0.1},
    "vago": {"valence": 0.7, "arousal": 0.3, "tenderness": 0.6},
    "valore": {"valence": 0.6, "arousal": 0.6, "tenderness": 0.2},
    "valentone": {"valence": 0.5, "arousal": 0.7, "tenderness": 0.0},
    "vaghezza": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.7},
    "vago": {"valence": 0.7, "arousal": 0.2, "tenderness": 0.7},
    "vantaggio": {"valence": 0.6, "arousal": 0.5, "tenderness": 0.0},
    "vantaggioso": {"valence": 0.6, "arousal": 0.3, "tenderness": 0.0},
    "vapore": {"valence": 0.5, "arousal": 0.2, "tenderness": 0.3},
    "variopinto": {"valence": 0.7, "arousal": 0.4, "tenderness": 0.3},
    "velocità": {"valence": 0.4, "arousal": 0.8, "tenderness": 0.0},
    "veloce": {"valence": 0.4, "arousal": 0.8, "tenderness": 0.0},
    "verde": {"valence": 0.6, "arousal": 0.2, "tenderness": 0.4},
    "verdeggiante": {"valence": 0.7, "arousal": 0.3, "tenderness": 0.4},
    "verdura": {"valence": 0.6, "arousal": 0.1, "tenderness": 0.3},
    "verità": {"valence": 0.5, "arousal": 0.3, "tenderness": 0.1},
    "vero": {"valence": 0.5, "arousal": 0.2, "tenderness": 0.1},
    "verrace": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.0},
    "verosimile": {"valence": 0.4, "arousal": 0.2, "tenderness": 0.0},
    "verseggiare": {"valence": 0.7, "arousal": 0.3, "tenderness": 0.4},
    "verso": {"valence": 0.4, "arousal": 0.2, "tenderness": 0.0},
    "virtù": {"valence": 0.7, "arousal": 0.3, "tenderness": 0.2},
    "virtuoso": {"valence": 0.7, "arousal": 0.4, "tenderness": 0.2},
    "visione": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.3},
    "vista": {"valence": 0.4, "arousal": 0.2, "tenderness": 0.0},
    "vita": {"valence": 0.6, "arousal": 0.5, "tenderness": 0.3},
    "vitale": {"valence": 0.6, "arousal": 0.6, "tenderness": 0.2},
    "vitalità": {"valence": 0.7, "arousal": 0.7, "tenderness": 0.1},
    "vitreo": {"valence": 0.3, "arousal": 0.1, "tenderness": 0.0},
    "vivace": {"valence": 0.8, "arousal": 0.8, "tenderness": 0.3},
    "vivacissimo": {"valence": 0.8, "arousal": 0.9, "tenderness": 0.2},
    "vivacità": {"valence": 0.8, "arousal": 0.8, "tenderness": 0.2},
    "vivanda": {"valence": 0.5, "arousal": 0.2, "tenderness": 0.0},
    "vivandiera": {"valence": 0.4, "arousal": 0.4, "tenderness": 0.0},
    "vivaio": {"valence": 0.6, "arousal": 0.3, "tenderness": 0.2},
    "vivamen": {"valence": 0.7, "arousal": 0.7, "tenderness": 0.2},
    "vivanda": {"valence": 0.5, "arousal": 0.2, "tenderness": 0.0},
    "vivenza": {"valence": 0.7, "arousal": 0.6, "tenderness": 0.3},
    "vivere": {"valence": 0.6, "arousal": 0.5, "tenderness": 0.3},
    "vivida": {"valence": 0.7, "arousal": 0.6, "tenderness": 0.2},
    "vividità": {"valence": 0.7, "arousal": 0.6, "tenderness": 0.2},
    "vivido": {"valence": 0.7, "arousal": 0.6, "tenderness": 0.2},
    "vivificante": {"valence": 0.8, "arousal": 0.6, "tenderness": 0.3},
    "vivificare": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.4},
    "vivificazione": {"valence": 0.7, "arousal": 0.4, "tenderness": 0.3},
    "vivifico": {"valence": 0.8, "arousal": 0.5, "tenderness": 0.3},
    "vivissimo": {"valence": 0.8, "arousal": 0.8, "tenderness": 0.2},
    "vivitudine": {"valence": 0.7, "arousal": 0.6, "tenderness": 0.2},
    "vivo": {"valence": 0.7, "arousal": 0.7, "tenderness": 0.2},
    "viziare": {"valence": 0.2, "arousal": 0.4, "tenderness": -0.2},
    "viziato": {"valence": 0.1, "arousal": 0.3, "tenderness": -0.3},
    "viziatone": {"valence": 0.0, "arousal": 0.3, "tenderness": -0.4},
    "viziatrice": {"valence": 0.1, "arousal": 0.2, "tenderness": -0.3},
    "viziosità": {"valence": -0.3, "arousal": 0.3, "tenderness": -0.4},
    "vizioso": {"valence": -0.2, "arousal": 0.3, "tenderness": -0.3},
    "vizio": {"valence": -0.3, "arousal": 0.4, "tenderness": -0.4},
    
    # negativo
    "abisso": {"valence": -0.8, "arousal": 0.6, "tenderness": 0.0},
    "aborrevole": {"valence": -0.9, "arousal": 0.5, "tenderness": -0.5},
    "aborrimento": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.4},
    "aborrire": {"valence": -0.9, "arousal": 0.7, "tenderness": -0.4},
    "aborrevolissimo": {"valence": -1.0, "arousal": 0.7, "tenderness": -0.6},
    "abominio": {"valence": -0.9, "arousal": 0.7, "tenderness": -0.5},
    "abominevole": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominevolissimo": {"valence": -1.0, "arousal": 0.8, "tenderness": -0.6},
    "abominevolo": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominevolmente": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominevolezza": {"valence": -0.9, "arousal": 0.5, "tenderness": -0.5},
    "abominevole": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominevolmente": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominevolezza": {"valence": -0.9, "arousal": 0.5, "tenderness": -0.5},
    "abominazione": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominazionaccio": {"valence": -1.0, "arousal": 0.7, "tenderness": -0.6},
    "abominatamente": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominatamente": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominator": {"valence": -0.8, "arousal": 0.5, "tenderness": -0.4},
    "abominazione": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "abominatrice": {"valence": -0.8, "arousal": 0.4, "tenderness": -0.4},
    "abominevolmente": {"valence": -0.9, "arousal": 0.6, "tenderness": -0.5},
    "accusa": {"valence": -0.6, "arousal": 0.7, "tenderness": -0.2},
    "accusatore": {"valence": -0.6, "arousal": 0.6, "tenderness": -0.2},
    "accusatrice": {"valence": -0.6, "arousal": 0.5, "tenderness": -0.2},
    "accusa": {"valence": -0.6, "arousal": 0.7, "tenderness": -0.2},
    "accusa": {"valence": -0.6, "arousal": 0.7, "tenderness": -0.2},
    "afflizione": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "affliggente": {"valence": -0.8, "arousal": 0.4, "tenderness": -0.3},
    "affliggersi": {"valence": -0.7, "arousal": 0.4, "tenderness": -0.3},
    "affliggere": {"valence": -0.8, "arousal": 0.5, "tenderness": -0.3},
    "affliggitore": {"valence": -0.7, "arousal": 0.5, "tenderness": -0.3},
    "affliggitrice": {"valence": -0.7, "arousal": 0.4, "tenderness": -0.3},
    "afflizione": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggente": {"valence": -0.8, "arousal": 0.4, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflittivamente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflittivamente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflizionare": {"valence": -0.7, "arousal": 0.4, "tenderness": -0.3},
    "afflizionatamente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionatamente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionato": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflizionatore": {"valence": -0.7, "arousal": 0.4, "tenderness": -0.3},
    "afflizionatrice": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionatamente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionevolmente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionevolmente": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflizionevole": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitto": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflitta": {"valence": -0.7, "arousal": 0.2, "tenderness": -0.3},
    "afflizione": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    "affliggenza": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.3},
    
    # altri
    "buio": {"valence": -0.6, "arousal": 0.4, "tenderness": 0.0},
    "buio": {"valence": -0.6, "arousal": 0.4, "tenderness": 0.0},
    "buio": {"valence": -0.6, "arousal": 0.4, "tenderness": 0.0},
    "buio": {"valence": -0.6, "arousal": 0.4, "tenderness": 0.0},
    "brutalità": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.6},
    "brutalità": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.6},
    "brutale": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.6},
    "brutalmente": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.6},
    "bruto": {"valence": -0.7, "arousal": 0.6, "tenderness": -0.5},
    "bruto": {"valence": -0.7, "arousal": 0.6, "tenderness": -0.5},
    "bruto": {"valence": -0.7, "arousal": 0.6, "tenderness": -0.5},
    "bruto": {"valence": -0.7, "arousal": 0.6, "tenderness": -0.5},
    "caldo": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.7},
    "caldo": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.7},
    "caldo": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.7},
    "caldo": {"valence": 0.6, "arousal": 0.4, "tenderness": 0.7},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    "calma": {"valence": 0.5, "arousal": 0.0, "tenderness": 0.3},
    
    # timore/paura
    "paura": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.2},
    "pauroso": {"valence": -0.8, "arousal": 0.7, "tenderness": -0.2},
    "terribile": {"valence": -0.9, "arousal": 0.8, "tenderness": -0.3},
    "terrore": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.2},
    "spaventoso": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.2},
    "spavento": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.2},
    
    # tristezza
    "tristezza": {"valence": -0.8, "arousal": 0.1, "tenderness": -0.3},
    "triste": {"valence": -0.8, "arousal": 0.1, "tenderness": -0.3},
    "tristemente": {"valence": -0.8, "arousal": 0.1, "tenderness": -0.3},
    "tristissimo": {"valence": -0.9, "arousal": 0.0, "tenderness": -0.4},
    "malinconico": {"valence": -0.7, "arousal": 0.0, "tenderness": -0.2},
    "malinconia": {"valence": -0.7, "arousal": 0.0, "tenderness": -0.2},
    "lutto": {"valence": -0.9, "arousal": 0.2, "tenderness": -0.3},
    "luttuoso": {"valence": -0.8, "arousal": 0.1, "tenderness": -0.3},
    "dolore": {"valence": -0.8, "arousal": 0.6, "tenderness": -0.3},
    "dolorante": {"valence": -0.7, "arousal": 0.3, "tenderness": -0.2},
    "doloroso": {"valence": -0.8, "arousal": 0.4, "tenderness": -0.3},
    
    # ira/rabbia
    "ira": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "iracondo": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irascibile": {"valence": -0.7, "arousal": 0.7, "tenderness": -0.3},
    "rabbia": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "rabbioso": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "furore": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "furibondo": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "furia": {"valence": -0.8, "arousal": 0.9, "tenderness": -0.4},
    "furioso": {"valence": -0.8, "arousal": 0.9, "tenderness": -0.4},
    "furor": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "furor": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "ira": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "irata": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irato": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "ira": {"valence": -0.9, "arousal": 0.9, "tenderness": -0.5},
    "iracondia": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irascibilità": {"valence": -0.7, "arousal": 0.7, "tenderness": -0.3},
    "irata": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irata": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irata": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irato": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irato": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
    "irato": {"valence": -0.8, "arousal": 0.8, "tenderness": -0.4},
}

def _normalize_word(word: str) -> str:
    """Normalizza una parola: minuscola, senza punteggiatura."""
    word = word.lower()
    word = re.sub(r"[^\wàèéìòùáíóúâêîôû]", "", word, flags=re.UNICODE)
    return word


def analyze_emotion(text: str) -> dict:
    """
    Analizza il contenuto emotivo di un testo (italiano).
    
    Ritorna dict con chiavi:
      - valence: float in [-1, +1], -1=molto negativo, +1=molto positivo
      - arousal: float in [0, +1], 0=calmo, +1=eccitato/intenso
      - tenderness: float in [0, +1], 0=neutro, +1=tenero/dolce
      - matched: list di parole matchate nel lessico
    """
    words = re.findall(r"\w+", text.lower())
    
    valences = []
    arousals = []
    tendernesses = []
    matched_words = []
    
    for word in words:
        norm_word = _normalize_word(word)
        if norm_word in EMOTION_LEXICON:
            entry = EMOTION_LEXICON[norm_word]
            valences.append(entry["valence"])
            arousals.append(entry["arousal"])
            tendernesses.append(entry["tenderness"])
            matched_words.append(norm_word)
    
    # Media: se non ci sono parole matchate, valori neutrali
    if valences:
        avg_valence = sum(valences) / len(valences)
    else:
        avg_valence = 0.0
    
    if arousals:
        avg_arousal = sum(arousals) / len(arousals)
    else:
        avg_arousal = 0.5
    
    if tendernesses:
        avg_tenderness = sum(tendernesses) / len(tendernesses)
    else:
        avg_tenderness = 0.0
    
    return {
        "valence": avg_valence,
        "arousal": avg_arousal,
        "tenderness": avg_tenderness,
        "matched": list(set(matched_words)),
    }


if __name__ == "__main__":
    demo_text = "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura ché la diritta via era smarrita"
    result = analyze_emotion(demo_text)
    print("Emotion analysis:", result)
