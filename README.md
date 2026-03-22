# Movie Recommender v1

This is a practical v1 implementation of the proposal: a Dockerized Streamlit app that lets a user:

1. Pick up to 3 genres
2. Generate a candidate pool of recent popular movies
3. Pick 3 to 5 favorite movies
4. Receive 5 to 15 ranked recommendations

## What v1 does

This MVP uses a strong metadata baseline instead of a full contrastive PyTorch training pipeline. That keeps v1 easy to run locally while matching the core product flow from the proposal. The architecture is intentionally split so you can swap in a learned embedding model in v2 without reworking the UI.

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
- Cosine similarity between the averaged seed vector and the full movie catalog

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
docker build -t movie-recommender-v1 .
docker run --rm -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  movie-recommender-v1
```

Then open `http://localhost:8501`.

## Run with Docker Compose

```bash
docker compose up --build
```

## Train a reusable artifact ahead of time

```bash
python -m app.train --dataset data/movies.csv --output models/movie_features.pkl
```

## Suggested v2 upgrades

- Replace TF-IDF with a PyTorch encoder and contrastive loss
- Add negative-pair generation from metadata overlap rules
- Add nDCG@10, Recall@10, and Hit Rate@10 evaluation scripts
- Add explanation cards showing top matching genres, cast, and keywords
- Add optional streaming availability metadata if you later decide to use an API
