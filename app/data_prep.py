import ast
import re
from typing import Iterable

import numpy as np
import pandas as pd

TEXT_COLUMNS = [
    'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'director'
]
NUMERIC_COLUMNS = [
    'budget', 'runtime', 'popularity', 'release_year', 'vote_average', 'vote_count'
]


def _safe_parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
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

    parts = re.split(r'[,|]', text)
    return [p.strip() for p in parts if p.strip()]


def _extract_director(crew_value):
    if pd.isna(crew_value):
        return ''
    if isinstance(crew_value, list):
        for item in crew_value:
            if isinstance(item, dict) and str(item.get('job', '')).lower() == 'director':
                return str(item.get('name', ''))
    try:
        parsed = ast.literal_eval(str(crew_value))
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and str(item.get('job', '')).lower() == 'director':
                    return str(item.get('name', ''))
    except Exception:
        pass
    return ''


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
        'release_date': 'release_date',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if 'title' not in df.columns:
        raise ValueError("Dataset must contain a 'title' column.")

    for col in ['genres', 'keywords', 'cast']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].apply(_safe_parse_list)

    if 'crew' not in df.columns:
        df['crew'] = ''
    df['director'] = df.get('director', '').replace('', np.nan)
    df['director'] = df['director'].fillna(df['crew'].apply(_extract_director))
    df['crew'] = df['crew'].apply(_safe_parse_list)

    if 'overview' not in df.columns:
        df['overview'] = ''
    df['overview'] = df['overview'].fillna('')

    if 'release_date' in df.columns:
        dates = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = dates.dt.year
    elif 'release_year' not in df.columns:
        df['release_year'] = np.nan

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median() if df[col].notna().any() else 0
        df[col] = df[col].fillna(median)

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ''

    df['genres_text'] = df['genres'].apply(lambda x: _list_to_text(x))
    df['keywords_text'] = df['keywords'].apply(lambda x: _list_to_text(x, limit=15))
    df['cast_text'] = df['cast'].apply(lambda x: _list_to_text(x, limit=5))
    df['crew_text'] = df['crew'].apply(lambda x: _list_to_text(x, limit=4))
    df['director_text'] = df['director'].fillna('').astype(str).str.lower().str.replace(' ', '_', regex=False)
    df['title_clean'] = df['title'].fillna('').astype(str).str.strip()

    df['profile_text'] = (
        df['title_clean'].str.lower().str.replace(' ', '_', regex=False) + ' ' +
        df['overview'].fillna('').astype(str).str.lower() + ' ' +
        df['genres_text'] + ' ' +
        df['keywords_text'] + ' ' +
        df['cast_text'] + ' ' +
        df['crew_text'] + ' ' +
        df['director_text']
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    df = df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)
    return df
