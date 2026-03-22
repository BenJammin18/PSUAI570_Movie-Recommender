from __future__ import annotations

import pandas as pd
from pathlib import Path


def read_movies_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Read the uploaded movie dataset safely.

    The user's CSV has a few malformed rows with broken quoting/newlines.
    For v1 we intentionally skip those bad lines so the app stays usable.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'Dataset not found: {csv_path}')

    last_error = None
    attempts = [
        dict(engine='python', on_bad_lines='skip', encoding='utf-8'),
        dict(engine='python', on_bad_lines='skip', encoding='utf-8-sig'),
        dict(engine='python', on_bad_lines='skip', encoding='latin-1'),
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            if df.empty:
                raise ValueError('Dataset loaded but contained zero rows.')
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f'Could not read dataset {csv_path}: {last_error}')
