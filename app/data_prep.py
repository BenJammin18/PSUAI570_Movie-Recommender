from __future__ import annotations

import ast
import re
from typing import Iterable

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ['popularity', 'release_year', 'vote_average', 'vote_count']
LOW_INFO_OVERVIEWS = {'', 'not found overviwe', 'not found overview', 'none', 'nan'}
LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'cn': 'Chinese',
    'hi': 'Hindi',
    'ru': 'Russian',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'pa': 'Punjabi',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'da': 'Danish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'ar': 'Arabic',
    'fa': 'Persian',
    'th': 'Thai',
    'id': 'Indonesian',
    'vi': 'Vietnamese',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'el': 'Greek',
    'he': 'Hebrew',
    'cs': 'Czech',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'uk': 'Ukrainian',
}


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
    items = items[:limit]
    return ' '.join(items)


def normalize_language_label(value: str) -> str:
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return 'Unknown'
    lowered = text.lower()
    if lowered in LANGUAGE_MAP:
        return LANGUAGE_MAP[lowered]
    if len(text) == 2:
        return lowered.upper()
    return text.title()


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
        'original_language': '',
        'original_title': '',
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df['title'] = df['title'].fillna('').astype(str).str.strip()
    df = df[df['title'] != ''].copy()

    for col in ['genres', 'keywords']:
        df[col] = df[col].apply(_safe_parse_list)

    df['overview'] = df['overview'].fillna('').astype(str)
    df['overview_clean'] = df['overview'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

    dates = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = dates.dt.year.fillna(0)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median() if df[col].notna().any() else 0
        df[col] = df[col].fillna(median)

    df['genres_text'] = df['genres'].apply(lambda x: _list_to_text(x, limit=8))
    df['keywords_text'] = df['keywords'].apply(lambda x: _list_to_text(x, limit=12))
    df['title_clean'] = df['title'].astype(str).str.strip()
    df['language_label'] = df['original_language'].apply(normalize_language_label)

    df['profile_text'] = (
        df['overview_clean'] + ' ' +
        df['genres_text'] + ' ' +
        df['keywords_text']
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    df['has_real_genres'] = df['genres'].apply(lambda x: len(x) > 0)
    df['has_real_text'] = ~df['overview_clean'].isin(LOW_INFO_OVERVIEWS)
    df['is_eligible'] = (
        df['has_real_genres'] &
        (
            df['has_real_text'] |
            (df['vote_count'] >= 25) |
            (df['popularity'] >= 15)
        )
    )

    df = df.sort_values(['popularity', 'vote_count', 'vote_average'], ascending=[False, False, False])
    df = df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)
    return df
