# Movie Recommender v5

This project is a Streamlit movie recommender with a deep-learning ranking layer on top of a strong metadata baseline. The app is designed to:

1. Filter the catalog by genre, language, and year
2. Build a candidate pool of movies that actually changes with those filters
3. Let the user choose 3 to 5 seed movies from that pool
4. Return recommendations using a neural embedding model, genre/language constraints, and feedback refill

## What the app does now

The current app uses:

- TF-IDF features from movie profile text
- Scaled numeric movie features
- A small PyTorch encoder that learns an embedding space over the metadata feature matrix
- Deep embedding similarity for ranking
- Candidate pool generation that prioritizes stronger genre matches and newer titles
- Recommendation filtering by language and minimum release year
- Live feedback actions: like, dislike, and dismiss

The app expects a deep model artifact at startup. If the saved artifact is missing or stale, the app rebuilds it in deep mode before serving.

## Main user flow

In the UI, the user can:

1. Choose up to 3 genres
2. Choose one or more languages
3. Set a year floor for the candidate pool
4. Generate a candidate pool
5. Pick 3 to 5 favorites from that pool
6. Set a minimum recommendation year from the sidebar
7. Request recommendations and refine them with feedback

## Current model behavior

The recommender combines:

- Text features from:
  - title
  - overview
  - genres
  - keywords
- Numeric features from:
  - popularity
  - vote count
  - vote average
  - release year

The deep model is a compact feed-forward encoder trained with similarity pairs built from metadata overlap rules. To keep training practical on the full catalog, the encoder is trained on a representative sample and then used to batch-encode the full movie set.

## Expected dataset columns

The app is flexible, but best results come from a CSV with these columns:

- title
- overview
- genres
- keywords
- popularity
- release_date
- vote_average
- vote_count
- original_language

The loader can handle list-like fields as Python-list strings, JSON-ish arrays, or comma-separated text.

## Repository structure

- [app/main.py](/Users/benreber/grad_code/AI570/Final/PSUAI570_Movie-Recommender/app/main.py:1): Streamlit UI and app flow
- [app/recommender.py](/Users/benreber/grad_code/AI570/Final/PSUAI570_Movie-Recommender/app/recommender.py:1): feature building, training orchestration, and ranking logic
- [app/deep_model.py](/Users/benreber/grad_code/AI570/Final/PSUAI570_Movie-Recommender/app/deep_model.py:1): PyTorch encoder and pair generation
- [app/train.py](/Users/benreber/grad_code/AI570/Final/PSUAI570_Movie-Recommender/app/train.py:1): CLI training entry point
- [app/data_prep.py](/Users/benreber/grad_code/AI570/Final/PSUAI570_Movie-Recommender/app/data_prep.py:1): dataset cleanup and normalization

## Local Python setup

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Install the deep-learning dependency:

```bash
pip install -r requirements-deep.txt
```

Current deep dependency:

```text
torch==2.2.2
```

## Train the model artifact manually

```bash
python -m app.train --dataset data/movie_data.csv --output models/movie_features_v5.pkl
```

That command writes the deep artifact used by the app at startup.

## Run with Docker

Build and run directly:

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

The compose file mounts local `data/` and `models/` into the container, so the dataset and saved artifacts persist outside the image.

## Docker notes

- The Docker image installs both base dependencies and the PyTorch CPU wheel
- The project includes a `.dockerignore` so large local datasets and model files are not copied into the build context
- First startup can take longer if the deep artifact has to be rebuilt

## Default artifact paths

- Dataset: `data/movie_data.csv`
- Model artifact: `models/movie_features_v5.pkl`

## Known tradeoffs

- First-run startup is slower than a pure baseline app because a deep artifact may need to be trained
- The encoder is intentionally compact to keep training time manageable on local hardware
- Recommendations are only as good as the metadata quality in the source CSV

## Future improvements

- Add evaluation metrics such as nDCG@10, Recall@10, and Hit Rate@10
- Add richer explanation cards for genres, keywords, and similarity reasons
- Add visible training progress for first-run artifact generation
- Add separate train and serve Docker targets if you want lighter production containers
