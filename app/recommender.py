from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .data_prep import prepare_movies_dataframe


@dataclass
class RecommendationResult:
    title: str
    score: float
    genres: str
    release_year: int
    overview: str
    reason: str


class MovieRecommender:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: MinMaxScaler | None = None
        self.feature_matrix = None
        self.title_to_index: dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        self.df = prepare_movies_dataframe(df)
        self.title_to_index = {title.lower(): idx for idx, title in enumerate(self.df['title_clean'])}

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.9,
            dtype=np.float32,
        )
        text_matrix = self.vectorizer.fit_transform(self.df['profile_text'])

        numeric_cols = ['popularity', 'release_year', 'vote_average', 'vote_count']
        self.scaler = MinMaxScaler()
        numeric_matrix = self.scaler.fit_transform(self.df[numeric_cols]).astype(np.float32)
        numeric_sparse = sparse.csr_matrix(numeric_matrix)

        self.feature_matrix = sparse.hstack([text_matrix, numeric_sparse], format='csr')
        return self

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'df': self.df,
                    'vectorizer': self.vectorizer,
                    'scaler': self.scaler,
                    'feature_matrix': self.feature_matrix,
                    'title_to_index': self.title_to_index,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> 'MovieRecommender':
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        model = cls()
        model.df = payload['df']
        model.vectorizer = payload['vectorizer']
        model.scaler = payload['scaler']
        model.feature_matrix = payload['feature_matrix']
        model.title_to_index = payload['title_to_index']
        return model

    def titles(self) -> List[str]:
        if self.df is None:
            return []
        return self.df['title_clean'].tolist()

    def get_recent_popular_by_genres(self, genres: list[str], n: int = 15, min_year: int | None = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Recommender is not fitted.')

        df = self.df.copy()
        if genres:
            genre_set = {g.lower() for g in genres}
            mask = df['genres'].apply(lambda values: bool(genre_set.intersection({str(v).lower() for v in values})))
            df = df[mask]
        if min_year is not None:
            df = df[df['release_year'] >= min_year]

        df = df.sort_values(['popularity', 'vote_average', 'vote_count'], ascending=[False, False, False])
        return df.head(n)

    def recommend(self, seed_titles: list[str], k: int = 10) -> list[RecommendationResult]:
        if self.df is None or self.feature_matrix is None:
            raise ValueError('Recommender is not fitted.')
        if not seed_titles:
            return []

        indices = [self.title_to_index[t.lower()] for t in seed_titles if t.lower() in self.title_to_index]
        if not indices:
            return []

        seed_vectors = self.feature_matrix[indices]
        user_vector = sparse.csr_matrix(seed_vectors.mean(axis=0))

        similarities = cosine_similarity(user_vector, self.feature_matrix).flatten()
        ranked_indices = np.argsort(-similarities)
        seed_set = set(indices)
        results = []

        seed_genres = set()
        for idx in indices:
            row = self.df.iloc[idx]
            seed_genres.update({str(v).lower() for v in row['genres']})

        for idx in ranked_indices:
            if idx in seed_set:
                continue

            row = self.df.iloc[idx]
            overlap_genres = seed_genres.intersection({str(v).lower() for v in row['genres']})
            if overlap_genres:
                reason = 'shared genres: ' + ', '.join(sorted(overlap_genres)[:3])
            else:
                reason = 'strong metadata similarity'

            results.append(
                RecommendationResult(
                    title=row['title_clean'],
                    score=float(similarities[idx]),
                    genres=', '.join(row['genres']) if isinstance(row['genres'], list) else str(row['genres']),
                    release_year=int(row['release_year']) if not pd.isna(row['release_year']) else 0,
                    overview=str(row['overview'])[:400],
                    reason=reason,
                )
            )
            if len(results) >= k:
                break

        return results
