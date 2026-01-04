from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import (
    LyricsMelodyDataset,
    build_embedding_matrix,
    build_midi_index,
    build_vocab,
    collate_batch,
    load_lyrics_csv,
    train_word2vec,
)
from models import TextMelodyRNNStatic, TextMelodyRNNTemporal, init_with_embeddings


def make_dataloaders(
    records,
    midi_index,
    stoi,
    variant: str,
    batch_size: int,
    bucket_seconds: float,
    val_frac: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    dataset = LyricsMelodyDataset(
        records=records,
        midi_index=midi_index,
        stoi=stoi,
        variant=variant,
        bucket_seconds=bucket_seconds,
    )
    val_len = max(1, int(len(dataset) * val_frac))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    def collate_fn(batch):
        return collate_batch(batch, variant=variant)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, variant: str):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        melody_static = batch["melody_static"].to(device)
        melody_seq = batch["melody_seq"].to(device)

        optimizer.zero_grad()
        if variant == "temporal":
            logits = model(tokens, melody_seq)
        else:
            logits = model(tokens, melody_static)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tokens.numel()
        total_tokens += tokens.numel()
    return total_loss / max(total_tokens, 1)


def evaluate(model, dataloader, criterion, device, variant: str):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="val", leave=False):
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            melody_static = batch["melody_static"].to(device)
            melody_seq = batch["melody_seq"].to(device)
            if variant == "temporal":
                logits = model(tokens, melody_seq)
            else:
                logits = model(tokens, melody_static)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * tokens.numel()
            total_tokens += tokens.numel()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--midi_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["static", "temporal"], default="static")
    parser.add_argument("--bucket_seconds", type=float, default=2.0)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_lyrics_csv(args.train_csv)
    stoi, itos = build_vocab(records, min_freq=args.min_freq)
    w2v = train_word2vec(records, vector_size=args.embedding_dim, min_count=1)
    embedding_matrix = build_embedding_matrix(w2v, stoi)

    midi_index = build_midi_index(args.midi_dir)
    train_loader, val_loader = make_dataloaders(
        records, midi_index, stoi, args.variant, args.batch_size, args.bucket_seconds
    )

    melody_dim_static = embedding_matrix.shape[1]  # placeholder replaced per variant
    if args.variant == "static":
        melody_dim_static = 12 + 4
        model = TextMelodyRNNStatic(
            vocab_size=len(stoi),
            embed_dim=args.embedding_dim,
            melody_dim=melody_dim_static,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        melody_dim_temporal = 12 + 2
        model = TextMelodyRNNTemporal(
            vocab_size=len(stoi),
            embed_dim=args.embedding_dim,
            melody_dim=melody_dim_temporal,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )

    model.to(device)
    init_with_embeddings(model, embedding_matrix, device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=args.log_dir / args.variant)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = np.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.variant)
        val_loss = evaluate(model, val_loader, criterion, device, args.variant)
        writer.add_scalar("loss/train", train_loss, global_step=epoch)
        writer.add_scalar("loss/val", val_loss, global_step=epoch)
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = args.checkpoint_dir / f"{args.variant}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stoi": stoi,
                    "variant": args.variant,
                    "embedding_dim": args.embedding_dim,
                    "melody_dim": melody_dim_static if args.variant == "static" else melody_dim_temporal,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                ckpt_path,
            )
        global_step += 1
        print(f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    main()

