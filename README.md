# Movie Recommender v5

This is a Dockerized Streamlit movie recommender that lets a user:

1. Pick up to 3 genres
2. Generate a candidate pool of recent popular movies
3. Pick 3 to 5 favorite movies
4. Receive 5 to 15 ranked recommendations

## Runtime mode

The app now starts in deep hybrid mode by default:

- Docker installs the PyTorch dependency during image build
- App startup expects a deep-learning artifact
- If it finds an older baseline-only artifact, it rebuilds it as a deep model before serving
- `app.train` still exists if you want to prebuild the artifact manually

### Current scoring
- TF-IDF on combined movie profile text:
  - title
  - overview
  - genres
  - keywords
  - cast
  - crew
  - director
- MinMax-scaled numeric features:
  - budget
  - runtime
  - popularity
  - release year
  - vote average
  - vote count
- Neural embedding similarity on top of the baseline feature space

## Expected dataset columns
The app is flexible, but best results come from a CSV with these columns:

- title
- overview
- genres
- keywords
- cast
- crew
- director
- budget
- runtime
- popularity
- release_date or release_year
- vote_average
- vote_count

List-like columns can be Python-list style, JSON-ish arrays of dicts, or comma-separated text.

## Run locally with Docker

```bash
docker build -t movie-recommender-v5 .
docker run --rm -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  movie-recommender-v5
```

Then open `http://localhost:8501`.

## Run with Docker Compose

```bash
docker compose up --build
```

The first startup may take longer because the app can train and save the deep artifact if it is missing.

## Train a reusable artifact ahead of time

Install deep-learning dependencies first:

```bash
pip install -r requirements.txt -r requirements-deep.txt
```

```bash
python -m app.train --dataset data/movies.csv --output models/movie_features_v5.pkl
```

## Suggested upgrades

- Add evaluation metrics such as nDCG@10, Recall@10, and Hit Rate@10
- Add richer explanation cards for cast, keywords, and genre overlap
- Add a lightweight training-progress indicator for first-run model builds
- Add optional streaming availability metadata if you later want provider-aware recommendations
