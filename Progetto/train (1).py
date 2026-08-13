from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from music_transformer import RhythmConditionedMelodyTransformer, save_checkpoint, MIN_PITCH


class MelodyDataset(Dataset):
    def __init__(self, path: Path, max_seq_len: int):
        data = np.load(path, allow_pickle=True)
        self.examples = []
        for x, y in zip(data["X"], data["Y"]):
            x = np.asarray(x, dtype=np.float32)[:max_seq_len]
            y = np.asarray(y, dtype=np.int64)[:max_seq_len] - MIN_PITCH
            if len(x) >= 4 and len(x) == len(y):
                self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    max_len = max(len(x) for x, _ in batch)
    xs = torch.zeros(len(batch), max_len, 9, dtype=torch.float32)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        n = len(x)
        xs[i, :n] = torch.from_numpy(x)
        ys[i, :n] = torch.from_numpy(y)
        mask[i, :n] = False
    return xs, ys, mask


def run_epoch(model, loader, optimizer, criterion, device, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(x, padding_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        valid = y != -100
        total_loss += float(loss.item()) * int(valid.sum())
        total_tokens += int(valid.sum())
        correct += int((logits.argmax(-1)[valid] == y[valid]).sum())
    return total_loss / max(1, total_tokens), correct / max(1, total_tokens)


def main():
    p = argparse.ArgumentParser(description="Addestra il Transformer ritmo -> melodia")
    p.add_argument("--dataset", type=Path, default=Path("dataset/rhythm_melody.npz"))
    p.add_argument("--output", type=Path, default=Path("models/melody_transformer.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dataset = MelodyDataset(args.dataset, args.max_seq_len)
    if len(dataset) < 10:
        raise RuntimeError("Dataset troppo piccolo: servono almeno 10 esempi.")
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model_config = {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "max_seq_len": args.max_seq_len,
    }
    device = torch.device(args.device)
    model = RhythmConditionedMelodyTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, False)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(model, args.output, model_config)
            print(f"  [OK] checkpoint salvato: {args.output}")


if __name__ == "__main__":
    main()
