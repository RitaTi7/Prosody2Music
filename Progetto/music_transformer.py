from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

MIN_PITCH = 36
MAX_PITCH = 84
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1


class RhythmConditionedMelodyTransformer(nn.Module):
    """Transformer per predire una nota MIDI a ogni evento ritmico.

    Input per evento:
      duration_beats, delta_beats, velocity, stress, phrase_end,
      valence, energy, movement, tension

    Output: distribuzione sulla pitch MIDI 36..84.

    Il modello è volutamente piccolo: è pensato come primo modello
    addestrabile sul dataset Lakh e poi fine-tunabile con poesia italiana.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_projection = nn.Sequential(
            nn.Linear(9, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.pitch_head = nn.Linear(d_model, NUM_PITCHES)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # x: [batch, seq, 9]
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequenza troppo lunga: {x.size(1)} > {self.max_seq_len}")
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.input_projection(x) + self.position_embedding(pos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.final_norm(h)
        return self.pitch_head(h)


def save_checkpoint(model: nn.Module, path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = RhythmConditionedMelodyTransformer(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config
