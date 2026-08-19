"""
train_offline.py — Script da eseguire UNA SOLA VOLTA per l'addestramento generale (Fase 1)
"""

import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_melody import MelodyTransformerModel, MODEL_PATH

try:
    from lakh_midi import load_stats
    _LAKH_AVAILABLE = True
except ImportError:
    _LAKH_AVAILABLE = False


def augment_and_prepare_data(sequences, max_len=32, conditioning_dropout=0.15):
    """
    Applica Data Augmentation:
    1. Trasposizione sistematica degli intervalli e inversione direzionale.
    2. Conditioning Dropout (imposta valenza/arousal a 0 nel 15% dei casi per imparare la sintassi).
    """
    inputs, emotions = [], []
    
    for seq in sequences:
        raw_intervals = seq.get("intervals", [])
        if len(raw_intervals) < 4:
            continue
            
        v = seq.get("valence", 0.0)
        a = seq.get("arousal", 0.0)
        
        # Generiamo la sequenza base e la sua versione aumentata (invertita)
        variants = [raw_intervals, [-i for i in raw_intervals]]
        
        for var in variants:
            tokens = [max(0, min(24, i + 12)) for i in var[:max_len]]
            if len(tokens) < max_len:
                tokens = tokens + [12] * (max_len - len(tokens))
                
            # Conditioning Dropout
            cur_v = 0.0 if random.random() < conditioning_dropout else v
            cur_a = 0.0 if random.random() < conditioning_dropout else a
            
            inputs.append(tokens)
            emotions.append([cur_v, cur_a])
            
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(emotions, dtype=torch.float32)


def train_offline_model(epochs=12, batch_size=32, lr=1e-3):
    if not _LAKH_AVAILABLE:
        print("[train_offline] Errore: modulo lakh_midi non disponibile.")
        return

    print("=== FASE 1: ADDESTRAMENTO OFFLINE DEL TRANSFORMER GENERALIZZABILE ===")
    print("[1/4] Estrazione dati dal corpus Lakh MIDI...")
    stats = load_stats(verbose=True)
    sequences = stats.get("sequences", [])
    
    if not sequences:
        print("[train_offline] Nessuna sequenza trovata. Verificare il corpus.")
        return

    print(f"[2/4] Applicazione Data Augmentation su {len(sequences)} sequenze...")
    X, E = augment_and_prepare_data(sequences)
    
    dataset = TensorDataset(X, E)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelodyTransformerModel(dropout=0.2).to(device)
    
    # AdamW con Weight Decay per prevenire Overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"[3/4] Avvio training su {device} ({epochs} epoche)...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, e_batch in train_loader:
            x_batch, e_batch = x_batch.to(device), e_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch[:, :-1], e_batch)  # Input fino a N-1
            targets = x_batch[:, 1:]                 # Target da N
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoca {epoch:02d}/{epochs:02d} | Loss Media: {avg_loss:.4f}")

    print("[4/4] Salvataggio del modello generalizzato...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "stats": {
            "type": "Deep Transformer Generalizzato (PyTorch)",
            "training_sequences": len(X),
            "epochs": epochs,
            "final_loss": round(avg_loss, 4),
            "device": str(device)
        }
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"-> Modello salvato con successo in '{MODEL_PATH}'!")


if __name__ == "__main__":
    train_offline_model()