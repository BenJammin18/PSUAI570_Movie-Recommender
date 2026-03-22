from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
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
        self.pipeline: Pipeline | None = None
        self.feature_matrix = None
        self.title_to_index: dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        self.df = prepare_movies_dataframe(df)
        self.title_to_index = {title.lower(): idx for idx, title in enumerate(self.df['title_clean'])}

        text_cols = ['profile_text']
        numeric_cols = ['budget', 'runtime', 'popularity', 'release_year', 'vote_average', 'vote_count']

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(stop_words='english', max_features=12000, ngram_range=(1, 2)), 'profile_text'),
                ('num', MinMaxScaler(), numeric_cols),
            ],
            remainder='drop',
            sparse_threshold=0.3,
        )

        self.pipeline = Pipeline([('preprocessor', preprocessor)])
        self.feature_matrix = self.pipeline.fit_transform(self.df)
        return self

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'df': self.df,
                'pipeline': self.pipeline,
                'feature_matrix': self.feature_matrix,
                'title_to_index': self.title_to_index,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> 'MovieRecommender':
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        model = cls()
        model.df = payload['df']
        model.pipeline = payload['pipeline']
        model.feature_matrix = payload['feature_matrix']
        model.title_to_index = payload['title_to_index']
        return model

    def is_fitted(self) -> bool:
        return self.df is not None and self.feature_matrix is not None

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
        if sparse.issparse(seed_vectors):
            user_vector = seed_vectors.mean(axis=0)
        else:
            user_vector = np.mean(seed_vectors, axis=0, keepdims=True)

        similarities = cosine_similarity(user_vector, self.feature_matrix).flatten()
        ranked_indices = np.argsort(-similarities)
        seed_set = set(indices)
        results = []

        seed_genres = set()
        seed_directors = set()
        for idx in indices:
            row = self.df.iloc[idx]
            seed_genres.update({str(v).lower() for v in row['genres']})
            if row['director']:
                seed_directors.add(str(row['director']).lower())

        for idx in ranked_indices:
            if idx in seed_set:
                continue
            row = self.df.iloc[idx]
            overlap_genres = seed_genres.intersection({str(v).lower() for v in row['genres']})
            same_director = str(row['director']).lower() in seed_directors if row['director'] else False
            reason_bits = []
            if overlap_genres:
                reason_bits.append('shared genres: ' + ', '.join(sorted(overlap_genres)[:3]))
            if same_director:
                reason_bits.append('same director signal')
            if not reason_bits:
                reason_bits.append('strong metadata similarity')

            results.append(
                RecommendationResult(
                    title=row['title_clean'],
                    score=float(similarities[idx]),
                    genres=', '.join(row['genres']) if isinstance(row['genres'], list) else str(row['genres']),
                    release_year=int(row['release_year']) if not pd.isna(row['release_year']) else 0,
                    overview=str(row['overview'])[:300],
                    reason='; '.join(reason_bits),
                )
            )
            if len(results) >= k:
                break
        return results
