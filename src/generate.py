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
    start_words: List[str],
    max_len: int,
    bucket_seconds: float,
    device: torch.device,
    temperature: float,
    top_k: int,
    target_len: int,
    max_words_per_line: int,
    max_lines: int,
):
    midi_path = match_midi_path(record["artist"], record["title"], midi_index)
    if not midi_path:
        return None
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    static_feat = static_melody_features(pm)
    temporal = time_bucket_features(pm, bucket_seconds=bucket_seconds)

    itos = {i: s for s, i in stoi.items()}
    generations = []
    for start_word in start_words:
        tokens = [stoi.get(start_word.lower(), stoi[UNK_TOKEN])]
        with torch.no_grad():
            for _ in range(max_len):
                if len(tokens) >= target_len:
                    break
                token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                if variant == "static":
                    melody = torch.tensor(static_feat, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = model(token_tensor, melody)
                else:
                    aligned = align_melody_sequence(temporal, len(tokens))
                    melody_seq = torch.tensor(aligned, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = model(token_tensor, melody_seq)
                probs = torch.softmax(logits[0, -1], dim=-1)
                next_id = sample_next(probs, temperature=temperature, top_k=top_k)
                tokens.append(next_id)

        words = decode_tokens(tokens, itos).split()
        lines = []
        for i in range(0, len(words), max_words_per_line):
            if len(lines) >= max_lines:
                break
            lines.append(" ".join(words[i : i + max_words_per_line]))
        generations.append({"start_word": start_word, "generated": "\n".join(lines)})
    return generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, help="single checkpoint (backward compatible)")
    parser.add_argument("--ckpt_static", type=Path, help="checkpoint path for static variant")
    parser.add_argument("--ckpt_temporal", type=Path, help="checkpoint path for temporal variant")
    parser.add_argument("--midi_dir", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--start_words", nargs="+", default=["love", "night", "sky"], help="list of start words to iterate")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--target_len", type=int, default=50, help="desired total words before line chunking")
    parser.add_argument("--max_words_per_line", type=int, default=8, help="guideline: max words per output line")
    parser.add_argument("--max_lines", type=int, default=6, help="guideline: max number of lines")
    parser.add_argument("--bucket_seconds", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("generated.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midi_index = build_midi_index(args.midi_dir)
    records = load_lyrics_csv(args.test_csv)

    ckpts: List[Path] = []
    if args.ckpt:
        ckpts.append(args.ckpt)
    if args.ckpt_static:
        ckpts.append(args.ckpt_static)
    if args.ckpt_temporal:
        ckpts.append(args.ckpt_temporal)
    if not ckpts:
        raise ValueError("Provide at least one checkpoint via --ckpt or --ckpt_static/--ckpt_temporal")

    outputs = []
    for ckpt_path in ckpts:
        model, stoi, variant = load_checkpoint(ckpt_path, device)
        for record in records:
            gens = generate_for_record(
                record,
                model,
                stoi,
                variant,
                midi_index,
                start_words=args.start_words,
                max_len=args.max_len,
                bucket_seconds=args.bucket_seconds,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                target_len=args.target_len,
                max_words_per_line=args.max_words_per_line,
                max_lines=args.max_lines,
            )
            for g in gens:
                outputs.append(
                    {
                        "variant": variant,
                        "artist": record["artist"],
                        "title": record["title"],
                        "start_word": g["start_word"],
                        "generated": g["generated"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, indent=2))
    print(f"saved generations to {args.output}")


if __name__ == "__main__":
    main()
