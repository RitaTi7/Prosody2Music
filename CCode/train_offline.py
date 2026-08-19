"""
transformer_melody.py — Modello Deep Transformer con Data Augmentation e Top-p Sampling
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.path.join("data", "melody_transformer.pt")

# Token di padding dedicato, DIVERSO dal token 12 (che rappresenta un vero
# intervallo musicale di 0 semitoni, cioè una nota ripetuta). Prima questo
# fix, il padding riusava il token 12 e la loss non li distingueva: il
# modello veniva rinforzato a predire "nota ripetuta" anche solo per
# riempire sequenze corte, gonfiando artificialmente quella probabilità
# ben oltre la sua reale frequenza nei dati (~21-42% osservato, contro
# 76-92% appreso dal modello prima del fix — verificato empiricamente).
PAD_TOKEN = 25
VOCAB_SIZE = 26  # 25 intervalli (0..24, cioè -12..+12 semitoni) + 1 PAD

# --- SAMPLING: Top-p (Nucleus) + Temperature ---
def sample_top_p(logits, temperature=0.85, top_p=0.9):
    """Campiona il prossimo token applicando Temperature e Nucleus (Top-p) Sampling."""
    logits = logits / max(temperature, 1e-5)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Rimuove i token oltre la soglia cumulativa top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# --- ARCHITETTURA NEURALE ---
class MelodyTransformerModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.emotion_fc = nn.Linear(2, d_model)  # Proiezione di Valence e Arousal
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, emotion):
        # x: [batch, seq_len], emotion: [batch, 2]
        seq_len = x.size(1)
        emb = self.embedding(x)
        emo_emb = self.emotion_fc(emotion).unsqueeze(1)  # [batch, 1, d_model]
        
        # Concateniamo l'embedding emotivo in testa alla sequenza
        h = torch.cat([emo_emb, emb], dim=1)
        
        # Maschera causale per impedire di "guardare nel futuro"
        mask = torch.triu(torch.full((seq_len + 1, seq_len + 1), float('-inf')), diagonal=1).to(x.device)
        
        out = self.transformer(h, mask=mask)
        return self.fc_out(out[:, 1:, :])  # Scartiamo il token emotivo in output

# --- INFERENCE ONLINE (Carica solo il modello salvato) ---
class TrainedMelodyTransformer:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MelodyTransformerModel().to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.stats = checkpoint.get("stats", {})
            self.model.eval()
            self.loaded = True
        else:
            self.loaded = False

    def generate(self, valence, arousal, length=32, temperature=0.85, top_p=0.9, seed=None):
        if not self.loaded:
            raise RuntimeError("Modello non trovato! Esegui prima 'train_offline.py'.")
        
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            
        self.model.eval()
        with torch.no_grad():
            emotion = torch.tensor([[valence, arousal]], dtype=torch.float32).to(self.device)
            # Token iniziale fisso (es. 12 = intervallo 0 semitoni)
            generated = [12] 
            
            for _ in range(length - 1):
                x = torch.tensor([generated], dtype=torch.long).to(self.device)
                logits = self.model(x, emotion)[0, -1, :].clone()
                logits[PAD_TOKEN] = float('-inf')  # non è un intervallo valido, mai generabile
                next_token = sample_top_p(logits, temperature=temperature, top_p=top_p)
                generated.append(next_token)
                
        # Converte i token in intervalli reali (-12 .. +12 semitoni)
        return [t - 12 for t in generated]


def load_inference_model():
    """Carica il modello pre-addestrato se esiste, senza eseguire il training."""
    model = TrainedMelodyTransformer()
    return model if model.loaded else None


def generate(model, valence, arousal, length, seed=None):
    return model.generate(valence, arousal, length=length, temperature=0.85, top_p=0.9, seed=seed)