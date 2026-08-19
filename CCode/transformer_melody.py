"""
transformer_melody.py — Un vero transformer (self-attention, decoder-only,
stile GPT in miniatura) addestrato sul corpus Lakh MIDI per generare
sequenze di intervalli melodici, condizionato dall'emotion embedding.

A differenza di LakhIntervalModel (lakh_midi.py), che modella solo la
probabilità del PROSSIMO intervallo dato quello corrente (un modello
bigramma), questo modulo impara dipendenze su un'intera finestra di note
precedenti tramite self-attention — è il componente che rende il nome
"Music Transformer" del progetto non più solo un placeholder.

Scala del modello: deliberatamente piccola (dim=64, 3 layer, 4 head),
pensata per essere addestrata in pochi minuti su CPU singola, non per
qualità production-grade. È comunque un vero transformer autoregressivo
addestrato da zero sui dati, non una libreria pre-addestrata.

Vocabolario (36 token):
  0..24   intervallo melodico in semitoni, da -12 a +12
  25      PAD (padding per sequenze più corte della finestra)
  26      BOS (inizio sequenza)
  27..35  token di condizionamento: 3 bin di valenza x 3 bin di arousal,
          anteposto alla sequenza generata per orientare il "carattere"
          della melodia in base all'emotion embedding della poesia

Uso:
    model = load_or_train_model()               # allena se non c'è cache, altrimenti carica
    intervals = generate(model, valence=0.3, arousal=0.6, length=20)
"""

import os
import random

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from lakh_midi import extract_interval_sequences

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "data", "melody_transformer.pt")

# --- vocabolario -----------------------------------------------------------
MIN_INTERVAL, MAX_INTERVAL = -12, 12
N_INTERVAL_TOKENS = MAX_INTERVAL - MIN_INTERVAL + 1  # 25
PAD = N_INTERVAL_TOKENS        # 25
BOS = N_INTERVAL_TOKENS + 1    # 26
COND_BASE = N_INTERVAL_TOKENS + 2  # 27..35 (9 combinazioni valenza x arousal)
VOCAB_SIZE = COND_BASE + 9     # 36

MAX_SEQ_LEN = 65  # 1 token di condizionamento + fino a 64 intervalli


def interval_to_token(interval: int) -> int:
    interval = max(MIN_INTERVAL, min(MAX_INTERVAL, interval))
    return interval - MIN_INTERVAL


def token_to_interval(token: int) -> int:
    return token + MIN_INTERVAL


def condition_token(valence: float, arousal: float) -> int:
    """Discretizza (valenza, arousal) in 3x3=9 bin e ritorna il token
    di condizionamento corrispondente."""
    def bin3(x):
        if x < -0.2:
            return 0
        if x > 0.2:
            return 2
        return 1
    return COND_BASE + bin3(valence) * 3 + bin3(arousal)


# --- modello -----------------------------------------------------------
if _TORCH_AVAILABLE:
    class MelodyTransformer(nn.Module):
        """Decoder-only transformer in miniatura: token+posizione -> N
        layer di self-attention causale -> proiezione sul vocabolario."""

        def __init__(self, vocab_size=VOCAB_SIZE, dim=64, n_layers=3, n_heads=4,
                     ff_dim=128, max_len=MAX_SEQ_LEN, dropout=0.1):
            super().__init__()
            self.dim = dim
            self.max_len = max_len
            self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
            self.pos_emb = nn.Embedding(max_len, dim)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, tokens):
            # tokens: (batch, seq_len)
            seq_len = tokens.size(1)
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            x = self.token_emb(tokens) + self.pos_emb(positions)

            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device), diagonal=1
            )
            pad_mask = (tokens == PAD)

            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
            return self.head(x)  # (batch, seq_len, vocab_size)
else:
    MelodyTransformer = None


def _sequence_proxy_va(intervals, tempo):
    """
    Stima un proxy di (valenza, arousal) direttamente dalle caratteristiche
    musicali della sequenza, da usare come segnale di condizionamento in
    training. Lakh MIDI non ha etichette di emozione, quindi un
    condizionamento assegnato a caso in training non avrebbe nulla da
    correlare e verrebbe ignorato dal modello (è esattamente quello che
    succedeva prima di questo fix: stesso identico output per qualunque
    condizione).

    La mappatura è la stessa logica già usata altrove nella pipeline
    (music_transformer.py: valenza alta -> preferenza a salire, arousal
    alto -> salti più ampi), applicata qui alla rovescia: una sequenza che
    tende ad ascendere e ha intervalli ampi viene letta come "valenza e
    arousal alti", non perché lo sia davvero (non lo sappiamo), ma per dare
    al modello un segnale di condizionamento coerente con quello che userà
    in inferenza (l'emotion embedding vero della poesia).
    """
    if not intervals:
        return 0.0, 0.0
    mean_interval = sum(intervals) / len(intervals)
    mean_abs = sum(abs(i) for i in intervals) / len(intervals)

    valence_proxy = max(-1.0, min(1.0, mean_interval / 3.0))
    tempo_norm = max(0.0, min(1.0, (tempo - 60) / 120))
    jump_norm = max(0.0, min(1.0, mean_abs / 5.0))
    arousal_proxy = max(-1.0, min(1.0, (tempo_norm + jump_norm) - 1.0))
    return valence_proxy, arousal_proxy


def _prepare_training_batch(sequences, batch_size, rng):
    """Antepone [token_condizionamento, BOS] a ciascuna sequenza. Il token
    di condizionamento è derivato dal proxy valenza/arousal calcolato dalle
    caratteristiche musicali reali della sequenza stessa (vedi
    _sequence_proxy_va), non assegnato a caso: solo così il modello ha
    davvero qualcosa da imparare sulla relazione tra condizionamento e
    stile della melodia generata."""
    batch = rng.sample(sequences, min(batch_size, len(sequences)))
    tokens_batch = []
    for seq in batch:
        intervals = seq["intervals"]
        valence_proxy, arousal_proxy = _sequence_proxy_va(intervals, seq["tempo"])
        cond = condition_token(valence_proxy, arousal_proxy)
        toks = [cond, BOS] + [interval_to_token(i) for i in intervals]
        toks = toks[:MAX_SEQ_LEN]
        toks = toks + [PAD] * (MAX_SEQ_LEN - len(toks))
        tokens_batch.append(toks)
    return torch.tensor(tokens_batch, dtype=torch.long)


def train_model(sequences, epochs=6, batch_size=32, steps_per_epoch=80,
                 lr=3e-4, seed=42, verbose=True):
    """Addestra il transformer con teacher forcing (next-token prediction)
    sulle sequenze estratte dal corpus Lakh MIDI."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch non installato: impossibile addestrare il transformer")

    torch.manual_seed(seed)
    rng = random.Random(seed)

    model = MelodyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            tokens = _prepare_training_batch(sequences, batch_size, rng)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"[transformer_melody] epoca {epoch + 1}/{epochs}  "
                  f"loss media: {epoch_loss / steps_per_epoch:.4f}")

    return model


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_or_train_model(midi_dir=None, max_files=400, epochs=6, path=MODEL_PATH, verbose=True):
    """Carica il modello da cache se presente, altrimenti estrae le
    sequenze dal corpus Lakh, addestra e salva."""
    if not _TORCH_AVAILABLE:
        if verbose:
            print("[transformer_melody] PyTorch non disponibile, transformer disattivato")
        return None

    model = MelodyTransformer()
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        if verbose:
            print(f"[transformer_melody] modello caricato da cache: {path}")
        return model

    if verbose:
        print("[transformer_melody] nessuna cache trovata, estraggo le sequenze ed addestro...")
    kwargs = {"max_files": max_files, "verbose": verbose}
    if midi_dir is not None:
        kwargs["midi_dir"] = midi_dir
    sequences = extract_interval_sequences(**kwargs)
    if len(sequences) < 20:
        if verbose:
            print("[transformer_melody] troppo poche sequenze estratte, transformer disattivato")
        return None

    model = train_model(sequences, epochs=epochs, verbose=verbose)
    model.eval()
    save_model(model, path)
    if verbose:
        print(f"[transformer_melody] modello addestrato e salvato in {path}")
    return model


@torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
def generate(model, valence: float, arousal: float, length: int, temperature=1.0, seed=None):
    """Genera autoregressivamente una sequenza di 'length' intervalli
    melodici (in semitoni), condizionata dall'emotion embedding."""
    if model is None:
        return None

    if seed is not None:
        torch.manual_seed(seed)

    cond = condition_token(valence, arousal)
    tokens = [cond, BOS]

    for _ in range(length):
        inp = torch.tensor([tokens[-MAX_SEQ_LEN:]], dtype=torch.long)
        logits = model(inp)[0, -1]  # logits dell'ultimo token della sequenza
        probs = torch.softmax(logits[:N_INTERVAL_TOKENS] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

    return [token_to_interval(t) for t in tokens[2:]]  # esclude cond e BOS


if __name__ == "__main__":
    model = load_or_train_model(max_files=400, epochs=6)
    if model is not None:
        for label, v, a in [("triste/calmo", -0.6, -0.2), ("gioioso/energico", 0.7, 0.6)]:
            intervals = generate(model, valence=v, arousal=a, length=16, seed=1)
            print(f"{label:20s} -> {intervals}")
