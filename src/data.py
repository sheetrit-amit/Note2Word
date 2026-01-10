from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z']+|[0-9]+", text.lower())
    return tokens


def load_lyrics_csv(csv_path: Path) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path, header=None)
    df = df.rename(columns={0: "artist", 1: "title", 2: "lyrics"})
    df = df[["artist", "title", "lyrics"]]
    df = df.dropna(subset=["artist", "title", "lyrics"])
    return df.to_dict(orient="records")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_midi_index(midi_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in midi_dir.glob("*.mid"):
        slug = normalize_name(path.stem)
        index[slug] = path
    return index


def match_midi_path(artist: str, title: str, midi_index: Dict[str, Path]) -> Optional[Path]:
    slug = normalize_name(f"{artist} - {title}")
    return midi_index.get(slug)


def build_vocab(records: Sequence[Dict[str, str]], min_freq: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter: Counter[str] = Counter()
    for row in records:
        counter.update(simple_tokenize(row["lyrics"]))

    stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            stoi.setdefault(token, len(stoi))
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def load_pretrained_embeddings(path: Path, expected_dim: int = 300) -> Dict[str, np.ndarray]:
    """
    Load GloVe-style pretrained embeddings from a text file.

    Each line: <token> <v1> <v2> ... <vN>
    """
    if not path.exists():
        raise FileNotFoundError(f"Pretrained embeddings not found at {path}")

    embeddings: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < expected_dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] != expected_dim:
                # skip rows with unexpected dimensionality
                continue
            embeddings[word] = vec
    if not embeddings:
        raise ValueError(f"No embeddings loaded from {path}")
    return embeddings


def build_embedding_matrix(embeddings: Dict[str, np.ndarray], stoi: Dict[str, int], embedding_dim: int) -> np.ndarray:
    matrix = np.random.normal(0, 0.1, size=(len(stoi), embedding_dim)).astype(np.float32)
    # PAD explicitly zeroed
    if PAD_TOKEN in stoi:
        matrix[stoi[PAD_TOKEN]] = 0.0
    for token, idx in stoi.items():
        vec = embeddings.get(token)
        if vec is not None and vec.shape[0] == embedding_dim:
            matrix[idx] = vec
    return matrix


def pitch_class_histogram(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    hist = pm.get_pitch_class_histogram(use_duration=True)
    denom = hist.sum() + 1e-8
    return hist / denom


def static_melody_features(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    pc = pitch_class_histogram(pm)
    tempo_changes = pm.get_tempo_changes()[1]
    tempo_mean = float(np.mean(tempo_changes)) if len(tempo_changes) else 0.0
    tempo_std = float(np.std(tempo_changes)) if len(tempo_changes) else 0.0
    duration = pm.get_end_time()
    note_count = sum(len(inst.notes) for inst in pm.instruments)
    note_density = note_count / max(duration, 1e-3)
    return np.concatenate([pc, np.array([tempo_mean, tempo_std, duration, note_density], dtype=np.float32)])


def time_bucket_features(
    pm: pretty_midi.PrettyMIDI, bucket_seconds: float = 2.0, max_buckets: Optional[int] = None
) -> np.ndarray:
    end_time = max(pm.get_end_time(), bucket_seconds)
    buckets = int(np.ceil(end_time / bucket_seconds))
    if max_buckets:
        buckets = min(buckets, max_buckets)
    features: List[np.ndarray] = []
    for b in range(buckets):
        start = b * bucket_seconds
        stop = start + bucket_seconds
        notes = [
            n
            for inst in pm.instruments
            for n in inst.notes
            if n.start < stop and n.end > start
        ]
        if not notes:
            pitch_hist = np.zeros(12, dtype=np.float32)
            vel = 0.0
            dur = 0.0
        else:
            pitch_hist = np.zeros(12, dtype=np.float32)
            velocities = []
            durations = []
            for n in notes:
                pitch_hist[n.pitch % 12] += n.end - n.start
                velocities.append(n.velocity)
                durations.append(n.end - n.start)
            denom = pitch_hist.sum() + 1e-8
            pitch_hist = pitch_hist / denom
            vel = float(np.mean(velocities))
            dur = float(np.mean(durations))
        features.append(np.concatenate([pitch_hist, np.array([vel, dur], dtype=np.float32)]))
    return np.stack(features, axis=0)


def align_melody_sequence(melody_seq: np.ndarray, target_len: int) -> np.ndarray:
    if len(melody_seq) == 0:
        return np.zeros((target_len, 14), dtype=np.float32)
    src_idx = np.linspace(0, len(melody_seq) - 1, num=target_len)
    aligned = np.zeros((target_len, melody_seq.shape[1]), dtype=np.float32)
    for d in range(melody_seq.shape[1]):
        aligned[:, d] = np.interp(src_idx, np.arange(len(melody_seq)), melody_seq[:, d])
    return aligned


@dataclass
class Sample:
    tokens: List[int]
    target: List[int]
    melody_static: np.ndarray
    melody_seq: np.ndarray


class LyricsMelodyDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        midi_index: Dict[str, Path],
        stoi: Dict[str, int],
        max_len: int = 256,
        bucket_seconds: float = 2.0,
        variant: str = "static",
    ):
        assert variant in {"static", "temporal"}
        self.variant = variant
        self.samples: List[Sample] = []
        for row in records:
            artist, title, lyrics = row["artist"], row["title"], row["lyrics"]
            midi_path = match_midi_path(artist, title, midi_index)
            if not midi_path:
                continue
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)
            except Exception:
                # Skip malformed MIDI files that pretty_midi/mido cannot parse
                continue
            static_feat = static_melody_features(pm).astype(np.float32)
            temporal = time_bucket_features(pm, bucket_seconds=bucket_seconds)

            tokens = [stoi.get(tok, stoi[UNK_TOKEN]) for tok in simple_tokenize(lyrics)]
            if len(tokens) < 2:
                continue
            tokens = tokens[: max_len + 1]
            input_tokens = tokens[:-1]
            targets = tokens[1:]
            melody_seq = align_melody_sequence(temporal, len(input_tokens)).astype(np.float32)
            self.samples.append(
                Sample(tokens=input_tokens, target=targets, melody_static=static_feat, melody_seq=melody_seq)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_batch(batch: Sequence[Sample], variant: str):
    max_len = max(len(s.tokens) for s in batch)
    pad_id = 0
    token_batch = []
    target_batch = []
    melody_static = []
    melody_seq = []
    for s in batch:
        pad_tokens = s.tokens + [pad_id] * (max_len - len(s.tokens))
        pad_targets = s.target + [pad_id] * (max_len - len(s.target))
        token_batch.append(pad_tokens)
        target_batch.append(pad_targets)
        melody_static.append(s.melody_static)
        melody_seq.append(
            np.vstack([s.melody_seq, np.zeros((max_len - len(s.tokens), s.melody_seq.shape[1]), dtype=np.float32)])
        )
    tokens = torch.tensor(token_batch, dtype=torch.long)
    targets = torch.tensor(target_batch, dtype=torch.long)
    static_tensor = torch.tensor(np.stack(melody_static), dtype=torch.float32)
    seq_tensor = torch.tensor(np.stack(melody_seq), dtype=torch.float32)
    return {
        "tokens": tokens,
        "targets": targets,
        "melody_static": static_tensor,
        "melody_seq": seq_tensor,
    }

