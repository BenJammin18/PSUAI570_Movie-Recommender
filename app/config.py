from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
DEFAULT_DATASET_PATH = DATA_DIR / 'movie_data.csv'
DEFAULT_EMBEDDINGS_PATH = MODELS_DIR / 'movie_features.pkl'
