from __future__ import annotations

import ast
import re
from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    'popularity', 'release_year', 'vote_average', 'vote_count'
]


def _safe_parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value

    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    name = item.get('name') or item.get('job') or item.get('character')
                    if name:
                        results.append(str(name))
                else:
                    results.append(str(item))
            return results
    except Exception:
        pass

    parts = re.split(r'[,|;/]', text)
    return [p.strip() for p in parts if p.strip()]


def _list_to_text(values: Iterable[str], limit: int | None = None) -> str:
    items = [str(v).strip().lower().replace(' ', '_') for v in values if str(v).strip()]
    if limit is not None:
        items = items[:limit]
    return ' '.join(items)


def prepare_movies_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        'movie_id': 'id',
        'name': 'title',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_defaults = {
        'title': '',
        'overview': '',
        'genres': '',
        'keywords': '',
        'popularity': np.nan,
        'vote_average': np.nan,
        'vote_count': np.nan,
        'release_date': '',
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df['title'] = df['title'].fillna('').astype(str).str.strip()
    df = df[df['title'] != ''].copy()

    for col in ['genres', 'keywords']:
        df[col] = df[col].apply(_safe_parse_list)

    df['overview'] = df['overview'].fillna('').astype(str)

    dates = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = dates.dt.year

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median() if df[col].notna().any() else 0
        df[col] = df[col].fillna(median)

    df['genres_text'] = df['genres'].apply(lambda x: _list_to_text(x, limit=8))
    df['keywords_text'] = df['keywords'].apply(lambda x: _list_to_text(x, limit=12))
    df['title_clean'] = df['title'].astype(str).str.strip()

    df['profile_text'] = (
        df['title_clean'].str.lower().str.replace(' ', '_', regex=False) + ' ' +
        df['overview'].str.lower() + ' ' +
        df['genres_text'] + ' ' +
        df['keywords_text']
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    df = df.sort_values(['popularity', 'vote_count', 'vote_average'], ascending=[False, False, False])
    df = df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)
    return df
