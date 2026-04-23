from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

try:
    from .config import MODEL_VERSION
    from .data_prep import prepare_movies_dataframe
except ImportError:
    from config import MODEL_VERSION
    from data_prep import prepare_movies_dataframe


def _load_deep_model_tools():
    try:
        from .deep_model import encode_all, train_encoder
        return encode_all, train_encoder
    except Exception:
        try:
            from deep_model import encode_all, train_encoder
            return encode_all, train_encoder
        except Exception:
            return None, None


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
        self.normalized_feature_matrix = None
        self.embedding_matrix: np.ndarray | None = None
        self.normalized_embedding_matrix: np.ndarray | None = None
        self.title_to_index: dict[str, int] = {}
        self.genre_sets: list[set[str]] = []
        self.language_labels: np.ndarray | None = None
        self.popularity_component: np.ndarray | None = None
        self.vote_count_component: np.ndarray | None = None
        self.quality_bonus: np.ndarray | None = None
        self.release_years: np.ndarray | None = None
        self.vote_counts: np.ndarray | None = None
        self.similarity_backend = 'tfidf_baseline'
        self.encoder_config = {
            'hidden_dim': 256,
            'embedding_dim': 96,
            'epochs': 4,
            'learning_rate': 1e-3,
            'train_sample_size': 4000,
            'encode_batch_size': 1024,
        }

    def fit(self, df: pd.DataFrame, enable_deep_training: bool = True):
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

        numeric_frame = pd.DataFrame(
            {
                'popularity_log': np.log1p(self.df['popularity'].clip(lower=0)),
                'vote_count_log': np.log1p(self.df['vote_count'].clip(lower=0)),
                'vote_average': self.df['vote_average'].clip(lower=0, upper=10),
                'release_year': self.df['release_year'].clip(lower=0),
            }
        )

        self.scaler = MinMaxScaler()
        numeric_matrix = self.scaler.fit_transform(numeric_frame).astype(np.float32)
        numeric_sparse = sparse.csr_matrix(numeric_matrix)

        self.feature_matrix = sparse.hstack([text_matrix, numeric_sparse], format='csr')
        self.normalized_feature_matrix = normalize(self.feature_matrix, norm='l2', axis=1, copy=False)
        self.embedding_matrix = None
        self.normalized_embedding_matrix = None
        self.similarity_backend = 'tfidf_baseline'

        if enable_deep_training:
            encode_all, train_encoder = _load_deep_model_tools()
        else:
            encode_all, train_encoder = None, None

        if enable_deep_training and train_encoder is not None and encode_all is not None:
            try:
                train_sample_size = int(self.encoder_config.get('train_sample_size', len(self.df)))
                if len(self.df) > train_sample_size:
                    rng = np.random.default_rng(42)
                    train_indices = np.sort(rng.choice(len(self.df), size=train_sample_size, replace=False))
                else:
                    train_indices = np.arange(len(self.df))

                x_dense = self.feature_matrix[train_indices].toarray().astype(np.float32)
                encoder = train_encoder(
                    x_dense=x_dense,
                    df=self.df.iloc[train_indices].reset_index(drop=True),
                    epochs=self.encoder_config['epochs'],
                    hidden_dim=self.encoder_config['hidden_dim'],
                    embedding_dim=self.encoder_config['embedding_dim'],
                    lr=self.encoder_config['learning_rate'],
                )
                self.embedding_matrix = self._encode_feature_matrix(encoder, encode_all)
                self.normalized_embedding_matrix = self._normalize_rows(self.embedding_matrix)
                self.similarity_backend = 'neural_hybrid'
            except Exception:
                self.embedding_matrix = None
                self.normalized_embedding_matrix = None
                self.similarity_backend = 'tfidf_baseline'

        self._build_runtime_cache()
        return self

    def _build_runtime_cache(self):
        if self.df is None:
            return

        self.genre_sets = [
            {str(value).lower() for value in row if str(value).strip()}
            for row in self.df['genres']
        ]
        self.language_labels = self.df['language_label'].astype(str).str.lower().to_numpy()

        popularity_component = np.log1p(self.df['popularity'].clip(lower=0).to_numpy(dtype=np.float32))
        vote_count_component = np.log1p(self.df['vote_count'].clip(lower=0).to_numpy(dtype=np.float32))
        popularity_max = max(float(popularity_component.max()) if popularity_component.size else 0.0, 1.0)
        vote_count_max = max(float(vote_count_component.max()) if vote_count_component.size else 0.0, 1.0)
        self.popularity_component = popularity_component / popularity_max
        self.vote_count_component = vote_count_component / vote_count_max
        self.quality_bonus = (0.40 * self.popularity_component) + (0.60 * self.vote_count_component)
        self.release_years = self.df['release_year'].to_numpy(dtype=np.float32)
        self.vote_counts = self.df['vote_count'].to_numpy(dtype=np.float32)

        if self.feature_matrix is not None and self.normalized_feature_matrix is None:
            self.normalized_feature_matrix = normalize(self.feature_matrix, norm='l2', axis=1, copy=False)
        if self.embedding_matrix is not None and self.normalized_embedding_matrix is None:
            self.normalized_embedding_matrix = self._normalize_rows(self.embedding_matrix)
        if self.embedding_matrix is None:
            self.similarity_backend = 'tfidf_baseline'

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
                    'embedding_matrix': self.embedding_matrix,
                    'title_to_index': self.title_to_index,
                    'encoder_config': self.encoder_config,
                    'similarity_backend': self.similarity_backend,
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
        model.embedding_matrix = payload.get('embedding_matrix')
        model.title_to_index = payload['title_to_index']
        model.encoder_config = payload.get('encoder_config', model.encoder_config)
        model.similarity_backend = payload.get(
            'similarity_backend',
            'neural_hybrid' if model.embedding_matrix is not None else 'tfidf_baseline',
        )
        model._build_runtime_cache()
        return model

    @property
    def uses_deep_model(self) -> bool:
        return self.embedding_matrix is not None and self.normalized_embedding_matrix is not None

    def titles(self) -> List[str]:
        if self.df is None:
            return []
        return self.df['title_clean'].tolist()

    def language_options(self) -> List[str]:
        if self.df is None:
            return []
        vals = sorted(
            value
            for value in self.df['language_label'].dropna().astype(str).unique().tolist()
            if value and value != 'Unknown'
        )
        if 'Unknown' in set(self.df['language_label'].astype(str)):
            vals.append('Unknown')
        return vals

    def get_recent_popular_by_genres(
        self,
        genres: list[str],
        n: int = 15,
        min_year: int | None = None,
        languages: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.df is None:
            raise ValueError('Recommender is not fitted.')

        df = self.df.copy()
        genre_set = {genre.lower() for genre in genres}
        if genres:
            mask = df['genres'].apply(lambda values: bool(genre_set.intersection({str(v).lower() for v in values})))
            df = df[mask]
        if languages:
            language_set = {str(value).lower() for value in languages}
            df = df[df['language_label'].astype(str).str.lower().isin(language_set)]
        if min_year is not None:
            df = df[df['release_year'] >= min_year]

        if df.empty:
            return df

        df = df.copy()
        df['genre_match_count'] = df['genres'].apply(
            lambda values: len(genre_set.intersection({str(v).lower() for v in values}))
        )
        if min_year is not None:
            df['year_distance'] = (df['release_year'] - min_year).clip(lower=0)
        else:
            df['year_distance'] = df['release_year']

        sort_columns = ['genre_match_count', 'year_distance', 'popularity', 'vote_count', 'vote_average']
        ascending = [False, False, False, False, False]
        df = df.sort_values(sort_columns, ascending=ascending)
        return df.head(n).reset_index(drop=True)

    def _indices_for_titles(self, titles: Iterable[str]) -> list[int]:
        indices = []
        for title in titles:
            idx = self.title_to_index.get(str(title).lower())
            if idx is not None:
                indices.append(idx)
        return indices

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return matrix / norms

    def _encode_feature_matrix(self, encoder, encode_all_fn) -> np.ndarray:
        if self.feature_matrix is None:
            raise ValueError('Feature matrix is not available.')

        batch_size = int(self.encoder_config.get('encode_batch_size', 1024))
        outputs = []
        for start in range(0, self.feature_matrix.shape[0], batch_size):
            stop = min(start + batch_size, self.feature_matrix.shape[0])
            batch_dense = self.feature_matrix[start:stop].toarray().astype(np.float32)
            outputs.append(encode_all_fn(encoder, batch_dense))
        return np.vstack(outputs)

    def _feature_similarity(self, indices: list[int]) -> np.ndarray:
        if self.normalized_feature_matrix is None:
            raise ValueError('Feature matrix is not available.')
        if not indices:
            return np.zeros(self.normalized_feature_matrix.shape[0], dtype=np.float32)

        matrix = self.normalized_feature_matrix[indices]
        user_vector = normalize(sparse.csr_matrix(matrix.mean(axis=0)), norm='l2', axis=1)
        similarities = self.normalized_feature_matrix @ user_vector.T
        return np.asarray(similarities.toarray()).ravel().astype(np.float32)

    def _embedding_similarity(self, indices: list[int]) -> np.ndarray:
        if self.normalized_embedding_matrix is None or self.embedding_matrix is None:
            raise ValueError('Embedding matrix is not available.')
        if not indices:
            return np.zeros(len(self.df), dtype=np.float32)

        matrix = self.embedding_matrix[indices]
        user_vector = self._normalize_rows(matrix.mean(axis=0, keepdims=True))
        return (user_vector @ self.normalized_embedding_matrix.T).flatten().astype(np.float32)

    def recommend(
        self,
        seed_titles: list[str],
        selected_genres: list[str] | None = None,
        selected_languages: list[str] | None = None,
        min_year: int | None = None,
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
        excluded_titles = {str(value).lower() for value in (excluded_titles or [])}
        liked_titles = list(liked_titles or [])
        disliked_titles = list(disliked_titles or [])

        seed_indices = self._indices_for_titles(seed_titles)
        if not seed_indices:
            return []

        positive_titles = list(dict.fromkeys([*seed_titles, *liked_titles]))
        positive_indices = self._indices_for_titles(positive_titles)
        negative_indices = self._indices_for_titles(disliked_titles)

        if self.uses_deep_model:
            positive_similarity = self._embedding_similarity(positive_indices)
            negative_similarity = self._embedding_similarity(negative_indices)
            base_reason = 'learned neural similarity'
        else:
            positive_similarity = self._feature_similarity(positive_indices)
            negative_similarity = self._feature_similarity(negative_indices)
            base_reason = 'strong metadata similarity'

        wanted_genres = {genre.lower() for genre in selected_genres}
        wanted_languages = {str(value).lower() for value in selected_languages}

        seed_genres = set()
        seed_languages = set()
        for idx in seed_indices + self._indices_for_titles(liked_titles):
            seed_genres.update(self.genre_sets[idx])
            seed_languages.add(self.language_labels[idx])

        gate_genres = wanted_genres or seed_genres
        overlap_base = wanted_genres.union(seed_genres) or seed_genres
        language_base = wanted_languages or seed_languages

        genre_overlap = np.array(
            [
                len(overlap_base.intersection(row_genres)) / max(len(overlap_base), 1)
                for row_genres in self.genre_sets
            ],
            dtype=np.float32,
        )
        language_bonus = np.array(
            [
                1.0 if (not language_base or row_language in language_base) else 0.0
                for row_language in self.language_labels
            ],
            dtype=np.float32,
        )
        candidate_mask = np.array(
            [
                (not gate_genres or bool(gate_genres.intersection(row_genres)))
                and (not wanted_languages or row_language in wanted_languages)
                for row_genres, row_language in zip(self.genre_sets, self.language_labels)
            ],
            dtype=bool,
        )
        if min_year is not None:
            candidate_mask = candidate_mask & (self.release_years >= float(min_year))

        scores = (
            (0.74 * positive_similarity)
            - (0.20 * negative_similarity)
            + (0.16 * genre_overlap)
            + (0.08 * language_bonus)
            + (0.08 * self.quality_bonus)
        )

        scores = np.where(candidate_mask, scores, -1e9)
        scores = scores - np.where(self.vote_counts < 20, 0.08, 0.0)
        scores = scores - np.where(self.release_years < 1980, 0.04, 0.0)

        ranked_indices = np.argsort(-scores)
        seed_set = set(seed_indices)
        results = []

        for idx in ranked_indices:
            row = self.df.iloc[idx]
            title = str(row['title_clean'])
            title_key = title.lower()
            if idx in seed_set or title_key in excluded_titles or scores[idx] <= -1e8:
                continue

            row_genres = self.genre_sets[idx]
            row_language = str(row['language_label'])
            selected_overlap = row_genres.intersection(wanted_genres)
            seed_overlap = row_genres.intersection(seed_genres)

            reasons = [base_reason]
            if selected_overlap:
                reasons.append('matches selected genres: ' + ', '.join(sorted(selected_overlap)[:3]))
            if selected_languages and row_language.lower() in wanted_languages:
                reasons.append(f'matches selected language: {row_language}')
            if seed_overlap:
                reasons.append('close to your picks via: ' + ', '.join(sorted(seed_overlap)[:3]))
            if row['vote_count'] >= 200:
                reasons.append('has stronger audience signal')

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
