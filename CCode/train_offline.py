"""
versione di gemini che corregge il bug della predizione continua del token 12


train_offline.py — Script autonomo per l'addestramento offline del Transformer (Fase 1)
"""

import os
import glob
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# IMPORTANTE: Importiamo anche PAD_TOKEN e VOCAB_SIZE
from transformer_melody import MelodyTransformerModel, MODEL_PATH, PAD_TOKEN, VOCAB_SIZE
from visualizer import plot_training_loss

try:
    import pretty_midi
    _PRETTY_MIDI_AVAILABLE = True
except ImportError:
    _PRETTY_MIDI_AVAILABLE = False


def _proxy_valence_arousal(intervals, tempo):
    """Stima un proxy di (valenza, arousal) dalle caratteristiche musicali reali."""
    if not intervals:
        return 0.0, 0.0
    mean_interval = sum(intervals) / len(intervals)
    mean_abs = sum(abs(i) for i in intervals) / len(intervals)

    valence_proxy = max(-1.0, min(1.0, mean_interval / 4.0))
    tempo_norm = max(0.0, min(1.0, (tempo - 60) / 120))
    jump_norm = max(0.0, min(1.0, mean_abs / 5.0))
    arousal_proxy = max(-1.0, min(1.0, (tempo_norm + jump_norm) - 1.0))
    return valence_proxy, arousal_proxy


def extract_sequences_from_midi(corpus_dir=".", max_files=600):
    if not _PRETTY_MIDI_AVAILABLE:
        print("[train_offline] Errore: libreria 'pretty_midi' non installata.")
        return []

    midi_files = glob.glob(os.path.join(corpus_dir, "**", "*.mid"), recursive=True)
    if not midi_files:
        midi_files = glob.glob(os.path.join(corpus_dir, "**", "*.midi"), recursive=True)

    if not midi_files:
        print(f"[train_offline] Nessun file MIDI trovato in '{corpus_dir}'.")
        return []

    print(f"[train_offline] Trovati {len(midi_files)} file MIDI. Parsing in corso (max {max_files})...")
    sequences = []

    for path in midi_files[:max_files]:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size else 120.0
            tempo = tempo if 30 < tempo < 300 else 120.0

            for inst in pm.instruments:
                if inst.is_drum:
                    continue
                notes = sorted(inst.notes, key=lambda n: n.start)
                if len(notes) < 5:
                    continue

                pitches = [n.pitch for n in notes]
                intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
                intervals = [max(-12, min(12, inv)) for inv in intervals]

                valence_proxy, arousal_proxy = _proxy_valence_arousal(intervals, tempo)
                sequences.append({
                    "intervals": intervals,
                    "valence": valence_proxy,
                    "arousal": arousal_proxy,
                })
        except Exception:
            continue

    print(f"[train_offline] Estratte {len(sequences)} sequenze melodiche valide.")
    return sequences


def augment_and_prepare_data(sequences, max_len=32, conditioning_dropout=0.15):
    inputs, emotions = [], []

    for seq in sequences:
        raw_intervals = seq.get("intervals", [])
        if len(raw_intervals) < 4:
            continue

        v = seq.get("valence", 0.0)
        a = seq.get("arousal", 0.0)

        variants = [raw_intervals, [-i for i in raw_intervals]]

        for var in variants:
            tokens = [max(0, min(24, i + 12)) for i in var[:max_len]]
            
            if len(tokens) < max_len:
                # --- FIX CRITICO: USIAMO PAD_TOKEN AL POSTO DI 12 ---
                tokens = tokens + [PAD_TOKEN] * (max_len - len(tokens))

            cur_v = 0.0 if random.random() < conditioning_dropout else v
            cur_a = 0.0 if random.random() < conditioning_dropout else a

            inputs.append(tokens)
            emotions.append([cur_v, cur_a])

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(emotions, dtype=torch.float32)


def train_offline_model(epochs=15, batch_size=32, lr=1e-3):
    print("=== FASE 1: ADDESTRAMENTO OFFLINE DEL TRANSFORMER GENERALIZZABILE ===")

    sequences = extract_sequences_from_midi(corpus_dir=".", max_files=600)
    if not sequences:
        print("[train_offline] Impossibile procedere: nessuna sequenza estratta dai file MIDI.")
        return

    print(f"[2/4] Applicazione Data Augmentation ed elaborazione token...")
    X, E = augment_and_prepare_data(sequences)
    if len(X) == 0:
        print("[train_offline] Errore: Nessuna sequenza valida dopo la tokenizzazione.")
        return

    dataset = TensorDataset(X, E)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelodyTransformerModel(dropout=0.2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # --- LA MAGIA INIZIA QUI: PESI E LABEL SMOOTHING ---
    
    # Creiamo un tensore di pesi pari a 1 per tutti i token
    class_weights = torch.ones(VOCAB_SIZE, device=device)
    # Riduciamo drasticamente l'importanza del token 12 (nota ripetuta)
    # L'IA capirà che indovinare uno 0 vale molto meno rispetto a indovinare un vero intervallo
    class_weights[12] = 0.2  
    
    # Configuriamo la nuova Loss Function
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,       # Applica la penalità al token 12
        ignore_index=PAD_TOKEN,     # Ignora al 100% i token di padding (non influiscono più sulla loss)
        label_smoothing=0.1         # Incoraggia l'IA a distribuire le probabilità (più creatività)
    )

    print(f"[3/4] Avvio training su {device} ({epochs} epoche, {len(X)} sequenze aumentate)...")
    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, e_batch in train_loader:
            x_batch, e_batch = x_batch.to(device), e_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch[:, :-1], e_batch)
            targets = x_batch[:, 1:]

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  Epoca {epoch:02d}/{epochs:02d} | Loss Media: {avg_loss:.4f}")

    if epoch_losses:
        try:
            os.makedirs("output", exist_ok=True)
            plot_training_loss(epoch_losses, save_path=os.path.join("output", "training_loss.png"))
        except Exception as e:
            print(f"[train_offline] Impossibile salvare il grafico della loss: {e}")

    print("[4/4] Salvataggio del modello generalizzato...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "stats": {
            "type": "Deep Transformer Generalizzato (PyTorch)",
            "training_sequences": len(X),
            "epochs": epochs,
            "final_loss": round(epoch_losses[-1], 4) if epoch_losses else 0.0,
            "device": str(device)
        }
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"-> Modello salvato con successo in '{MODEL_PATH}'!")


if __name__ == "__main__":
    train_offline_model()