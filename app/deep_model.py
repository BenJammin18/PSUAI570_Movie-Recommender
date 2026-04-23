from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TfidfEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)


class PairDataset(Dataset):
    def __init__(self, x: np.ndarray, pairs: list[tuple[int, int, int]]):
        self.x = x.astype(np.float32)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j, y = self.pairs[idx]
        return self.x[i], self.x[j], np.float32(y)


def _safe_set(values: Iterable[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def build_pairs(df: pd.DataFrame, max_pairs_per_movie: int = 2, seed: int = 42) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[int, int, int]] = []
    n = len(df)

    genre_sets = [_safe_set(row) if isinstance(row, list) else set() for row in df['genres']]
    languages = df['language_label'].astype(str).str.lower().tolist()
    keyword_sets = [_safe_set(row) if isinstance(row, list) else set() for row in df['keywords']]

    language_to_indices: dict[str, list[int]] = {}
    genre_to_indices: dict[str, list[int]] = {}
    for idx, (language, genres) in enumerate(zip(languages, genre_sets)):
        language_to_indices.setdefault(language, []).append(idx)
        for genre in genres:
            genre_to_indices.setdefault(genre, []).append(idx)

    all_indices = list(range(n))

    for i in range(n):
        positive_candidates: set[int] = set(language_to_indices.get(languages[i], []))
        for genre in genre_sets[i]:
            positive_candidates.update(genre_to_indices.get(genre, []))
        positive_candidates.discard(i)

        positive_list = list(positive_candidates)
        rng.shuffle(positive_list)

        added_positive = 0
        for j in positive_list[: min(len(positive_list), 80)]:
            genre_overlap = len(genre_sets[i].intersection(genre_sets[j]))
            keyword_overlap = len(keyword_sets[i].intersection(keyword_sets[j]))
            same_language = languages[i] == languages[j]
            if same_language and (genre_overlap >= 2 or (genre_overlap >= 1 and keyword_overlap >= 1)):
                pairs.append((i, j, 1))
                added_positive += 1
                if added_positive >= max_pairs_per_movie:
                    break

        added_negative = 0
        attempts = 0
        max_attempts = 120
        while added_negative < max_pairs_per_movie and attempts < max_attempts:
            j = rng.choice(all_indices)
            attempts += 1
            if i == j:
                continue
            if genre_sets[i].intersection(genre_sets[j]):
                continue
            if keyword_sets[i].intersection(keyword_sets[j]):
                continue
            pairs.append((i, j, -1))
            added_negative += 1

    rng.shuffle(pairs)
    return pairs


def train_encoder(
    x_dense: np.ndarray,
    df: pd.DataFrame,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    embedding_dim: int = 96,
    device: str = 'cpu',
):
    pairs = build_pairs(df)
    if not pairs:
        raise ValueError('Could not construct training pairs for the neural encoder.')

    dataset = PairDataset(x_dense, pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TfidfEncoder(
        input_dim=x_dense.shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.2)

    model.train()
    for _ in range(epochs):
        for x1, x2, y in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def encode_all(model: nn.Module, x_dense: np.ndarray, device: str = 'cpu') -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_dense, dtype=torch.float32, device=device)
        z = model(x_tensor).cpu().numpy()
    return z
