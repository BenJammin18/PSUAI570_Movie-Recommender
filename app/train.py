from __future__ import annotations

import argparse
from pathlib import Path

try:
    from app.config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from app.io_utils import read_movies_csv
    from app.recommender import MovieRecommender
except ModuleNotFoundError:
    from config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from io_utils import read_movies_csv
    from recommender import MovieRecommender


def main():
    parser = argparse.ArgumentParser(description='Train the Movie Recommender v1 artifact.')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_PATH), help='Path to movies CSV file')
    parser.add_argument('--output', default=str(DEFAULT_EMBEDDINGS_PATH), help='Output path for serialized model')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    df = read_movies_csv(dataset_path)
    recommender = MovieRecommender().fit(df)
    recommender.save(args.output)
    print(f'Saved trained artifact to {args.output}')


if __name__ == '__main__':
    main()
