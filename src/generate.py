from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data import (
    UNK_TOKEN,
    align_melody_sequence,
    build_midi_index,
    load_lyrics_csv,
    match_midi_path,
    simple_tokenize,
    static_melody_features,
    time_bucket_features,
)
from models import TextMelodyRNNStatic, TextMelodyRNNTemporal


def load_checkpoint(ckpt_path: Path, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device)
    stoi = payload["stoi"]
    variant = payload["variant"]
    embedding_dim = payload["embedding_dim"]
    melody_dim = payload["melody_dim"]
    hidden_dim = payload["hidden_dim"]
    num_layers = payload["num_layers"]
    vocab_size = len(stoi)
    if variant == "static":
        model = TextMelodyRNNStatic(
            vocab_size=vocab_size,
            embed_dim=embedding_dim,
            melody_dim=melody_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    else:
        model = TextMelodyRNNTemporal(
            vocab_size=vocab_size,
            embed_dim=embedding_dim,
            melody_dim=melody_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, stoi, variant


def decode_tokens(tokens: List[int], itos: Dict[int, str]) -> str:
    words = [itos.get(t, UNK_TOKEN) for t in tokens]
    return " ".join(words)


def sample_next(probs: torch.Tensor, temperature: float = 1.0, top_k: int = 10) -> int:
    if temperature != 1.0:
        probs = torch.log(probs + 1e-8) / temperature
        probs = torch.softmax(probs, dim=-1)
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)))
        topk_probs = topk_vals / topk_vals.sum()
        choice = torch.multinomial(topk_probs, 1).item()
        return topk_idx[choice].item()
    return torch.multinomial(probs, 1).item()


def generate_for_record(
    record,
    model,
    stoi: Dict[str, int],
    variant: str,
    midi_index: Dict[str, Path],
    start_word: str,
    max_len: int,
    bucket_seconds: float,
    device: torch.device,
):
    midi_path = match_midi_path(record["artist"], record["title"], midi_index)
    if not midi_path:
        return None
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    static_feat = static_melody_features(pm)
    temporal = time_bucket_features(pm, bucket_seconds=bucket_seconds)

    itos = {i: s for s, i in stoi.items()}
    tokens = [stoi.get(start_word.lower(), stoi[UNK_TOKEN])]

    with torch.no_grad():
        for _ in range(max_len):
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            if variant == "static":
                melody = torch.tensor(static_feat, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(token_tensor, melody)
            else:
                aligned = align_melody_sequence(temporal, len(tokens))
                melody_seq = torch.tensor(aligned, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(token_tensor, melody_seq)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_id = sample_next(probs, temperature=1.0, top_k=10)
            tokens.append(next_id)
    return decode_tokens(tokens, itos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--midi_dir", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--start_word", type=str, default="hello")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--bucket_seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("generated.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stoi, variant = load_checkpoint(args.ckpt, device)
    midi_index = build_midi_index(args.midi_dir)
    records = load_lyrics_csv(args.test_csv)

    outputs = []
    for record in records:
        text = generate_for_record(
            record,
            model,
            stoi,
            variant,
            midi_index,
            start_word=args.start_word,
            max_len=args.max_len,
            bucket_seconds=args.bucket_seconds,
            device=device,
        )
        if text:
            outputs.append(
                {"artist": record["artist"], "title": record["title"], "generated": text, "start_word": args.start_word}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, indent=2))
    print(f"saved generations to {args.output}")


if __name__ == "__main__":
    main()

