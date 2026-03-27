from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

try:
    from .config import MODEL_VERSION
    from .data_prep import prepare_movies_dataframe
except ImportError:
    from config import MODEL_VERSION
    from data_prep import prepare_movies_dataframe


@dataclass
class RecommendationResult:
    title: str
    score: float
    genres: str
    release_year: int
    overview: str
    reason: str
    language: str


class MovieRecommender:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: MinMaxScaler | None = None
        self.feature_matrix = None
        self.title_to_index: dict[str, int] = {}

    def fit(self, df: pd.DataFrame):
        full_df = prepare_movies_dataframe(df)
        self.df = full_df[full_df['is_eligible']].copy().reset_index(drop=True)
        self.title_to_index = {title.lower(): idx for idx, title in enumerate(self.df['title_clean'])}

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=12000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            dtype=np.float32,
        )
        text_matrix = self.vectorizer.fit_transform(self.df['profile_text'])

        numeric_frame = pd.DataFrame({
            'popularity_log': np.log1p(self.df['popularity'].clip(lower=0)),
            'vote_count_log': np.log1p(self.df['vote_count'].clip(lower=0)),
            'vote_average': self.df['vote_average'].clip(lower=0, upper=10),
            'release_year': self.df['release_year'].clip(lower=0),
        })

        self.scaler = MinMaxScaler()
        numeric_matrix = self.scaler.fit_transform(numeric_frame).astype(np.float32)
        numeric_sparse = sparse.csr_matrix(numeric_matrix)

        self.feature_matrix = sparse.hstack([text_matrix, numeric_sparse], format='csr')
        return self

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'model_version': MODEL_VERSION,
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
        if payload.get('model_version') != MODEL_VERSION:
            raise ValueError('Stale recommender artifact version.')
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

    def language_options(self) -> List[str]:
        if self.df is None:
            return []
        vals = sorted(v for v in self.df['language_label'].dropna().astype(str).unique().tolist() if v and v != 'Unknown')
        if 'Unknown' in set(self.df['language_label'].astype(str)):
            vals.append('Unknown')
        return vals

    def get_recent_popular_by_genres(self, genres: list[str], n: int = 15, min_year: int | None = None, languages: list[str] | None = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Recommender is not fitted.')

        df = self.df.copy()
        if genres:
            genre_set = {g.lower() for g in genres}
            mask = df['genres'].apply(lambda values: bool(genre_set.intersection({str(v).lower() for v in values})))
            df = df[mask]
        if languages:
            language_set = {str(v).lower() for v in languages}
            df = df[df['language_label'].astype(str).str.lower().isin(language_set)]
        if min_year is not None:
            df = df[df['release_year'] >= min_year]

        df = df.sort_values(['popularity', 'vote_count', 'vote_average'], ascending=[False, False, False])
        return df.head(n).reset_index(drop=True)

    def _indices_for_titles(self, titles: Iterable[str]) -> list[int]:
        indices = []
        for title in titles:
            idx = self.title_to_index.get(str(title).lower())
            if idx is not None:
                indices.append(idx)
        return indices

    def recommend(
        self,
        seed_titles: list[str],
        selected_genres: list[str] | None = None,
        selected_languages: list[str] | None = None,
        k: int = 10,
        excluded_titles: Iterable[str] | None = None,
        liked_titles: Iterable[str] | None = None,
        disliked_titles: Iterable[str] | None = None,
    ) -> list[RecommendationResult]:
        if self.df is None or self.feature_matrix is None:
            raise ValueError('Recommender is not fitted.')
        if not seed_titles:
            return []

        selected_genres = selected_genres or []
        selected_languages = selected_languages or []
        excluded_titles = {str(v).lower() for v in (excluded_titles or [])}
        liked_titles = list(liked_titles or [])
        disliked_titles = list(disliked_titles or [])

        seed_indices = self._indices_for_titles(seed_titles)
        if not seed_indices:
            return []

        positive_titles = list(dict.fromkeys([*seed_titles, *liked_titles]))
        positive_indices = self._indices_for_titles(positive_titles)
        negative_indices = self._indices_for_titles(disliked_titles)

        positive_matrix = self.feature_matrix[positive_indices]
        user_vector = sparse.csr_matrix(positive_matrix.mean(axis=0))
        positive_similarity = cosine_similarity(user_vector, self.feature_matrix).flatten()

        if negative_indices:
            negative_matrix = self.feature_matrix[negative_indices]
            negative_vector = sparse.csr_matrix(negative_matrix.mean(axis=0))
            negative_similarity = cosine_similarity(negative_vector, self.feature_matrix).flatten()
        else:
            negative_similarity = np.zeros(len(self.df), dtype=np.float32)

        wanted_genres = {g.lower() for g in selected_genres}
        wanted_languages = {str(v).lower() for v in selected_languages}
        seed_genres = set()
        seed_languages = set()
        for idx in seed_indices + self._indices_for_titles(liked_titles):
            row = self.df.iloc[idx]
            seed_genres.update({str(v).lower() for v in row['genres']})
            seed_languages.add(str(row['language_label']).lower())

        gate_genres = wanted_genres or seed_genres
        overlap_base = wanted_genres.union(seed_genres)
        if not overlap_base:
            overlap_base = seed_genres

        language_base = wanted_languages or seed_languages

        genre_overlap = []
        language_bonus = []
        candidate_mask = []
        for _, row in self.df.iterrows():
            row_genres = {str(v).lower() for v in row['genres']}
            row_language = str(row['language_label']).lower()
            overlap = len(overlap_base.intersection(row_genres))
            genre_overlap.append(overlap / max(len(overlap_base), 1))

            lang_match = 1.0 if (not language_base or row_language in language_base) else 0.0
            language_bonus.append(lang_match)

            genre_ok = True if not gate_genres else bool(gate_genres.intersection(row_genres))
            language_ok = True if not wanted_languages else row_language in wanted_languages
            candidate_mask.append(genre_ok and language_ok)

        genre_overlap = np.array(genre_overlap, dtype=np.float32)
        language_bonus = np.array(language_bonus, dtype=np.float32)
        candidate_mask = np.array(candidate_mask, dtype=bool)

        popularity_component = np.log1p(self.df['popularity'].clip(lower=0))
        vote_count_component = np.log1p(self.df['vote_count'].clip(lower=0))
        popularity_component = popularity_component / max(float(popularity_component.max()), 1.0)
        vote_count_component = vote_count_component / max(float(vote_count_component.max()), 1.0)
        quality_bonus = (0.40 * popularity_component) + (0.60 * vote_count_component)

        scores = (
            (0.72 * positive_similarity)
            - (0.18 * negative_similarity)
            + (0.16 * genre_overlap)
            + (0.10 * language_bonus)
            + (0.08 * quality_bonus)
        )

        scores = np.where(candidate_mask, scores, -1e9)
        scores = scores - np.where(self.df['vote_count'].to_numpy() < 20, 0.08, 0.0)
        scores = scores - np.where(self.df['release_year'].to_numpy() < 1980, 0.04, 0.0)

        ranked_indices = np.argsort(-scores)
        seed_set = set(seed_indices)
        results = []

        for idx in ranked_indices:
            row = self.df.iloc[idx]
            title = str(row['title_clean'])
            title_key = title.lower()
            if idx in seed_set:
                continue
            if title_key in excluded_titles:
                continue
            if scores[idx] <= -1e8:
                continue

            row_genres = {str(v).lower() for v in row['genres']}
            row_language = str(row['language_label'])
            selected_overlap = row_genres.intersection(wanted_genres)
            seed_overlap = row_genres.intersection(seed_genres)

            reasons = []
            if selected_overlap:
                reasons.append('matches selected genre(s): ' + ', '.join(sorted(selected_overlap)[:3]))
            if selected_languages and row_language.lower() in wanted_languages:
                reasons.append(f'matches selected language: {row_language}')
            if seed_overlap:
                reasons.append('similar to your pool picks via: ' + ', '.join(sorted(seed_overlap)[:3]))
            if row['vote_count'] >= 200:
                reasons.append('has stronger audience signal')
            if not reasons:
                reasons.append('strong metadata similarity')

            results.append(
                RecommendationResult(
                    title=title,
                    score=float(scores[idx]),
                    genres=', '.join(row['genres']) if isinstance(row['genres'], list) else str(row['genres']),
                    release_year=int(row['release_year']) if not pd.isna(row['release_year']) else 0,
                    overview=str(row['overview'])[:400],
                    reason='; '.join(reasons),
                    language=row_language,
                )
            )
            if len(results) >= k:
                break

        return results
