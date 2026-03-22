from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from app.config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from app.io_utils import read_movies_csv
    from app.recommender import MovieRecommender
except ModuleNotFoundError:
    from config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from io_utils import read_movies_csv
    from recommender import MovieRecommender

st.set_page_config(page_title='Movie Recommender v1', page_icon='🎬', layout='wide')


@st.cache_resource(show_spinner=False)
def load_or_train_recommender(dataset_path: str, model_path: str):
    dataset = Path(dataset_path)
    artifact = Path(model_path)

    if artifact.exists():
        try:
            return MovieRecommender.load(artifact), None
        except Exception as exc:
            artifact.unlink(missing_ok=True)
            load_error = f'Existing model artifact was invalid and got rebuilt: {exc}'
        else:
            load_error = None
    else:
        load_error = None

    if not dataset.exists():
        return None, f'Dataset not found: {dataset}'

    df = read_movies_csv(dataset)
    recommender = MovieRecommender().fit(df)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(artifact)
    return recommender, load_error


def render_header():
    st.title('🎬 Cross-Platform Movie Recommender v1')
    st.caption('Streamlit MVP tuned to safely load your uploaded dataset even when a few CSV rows are malformed.')


render_header()

with st.sidebar:
    st.header('Settings')
    dataset_path = st.text_input('Dataset CSV path', str(DEFAULT_DATASET_PATH))
    model_path = st.text_input('Serialized model path', str(DEFAULT_EMBEDDINGS_PATH))
    recommendation_count = st.slider('Number of recommendations', min_value=5, max_value=20, value=10)
    recent_year_floor = st.slider('Only show candidate movies from year', min_value=1950, max_value=2026, value=2016)

recommender, status_message = load_or_train_recommender(dataset_path, model_path)

if recommender is None:
    st.error(status_message or 'Could not load the dataset.')
    st.stop()

if status_message:
    st.warning(status_message)

st.success(f'Loaded {len(recommender.titles())} movies.')
st.info('This build reads the CSV with a tolerant parser and skips malformed lines so the app can still run against your uploaded dataset.')

st.subheader('Step 1: choose up to 3 favorite genres')
all_genres = sorted({g for row in recommender.df['genres'] for g in row if str(g).strip()})
selected_genres = st.multiselect('Genres', options=all_genres, max_selections=3)

col1, col2 = st.columns([1, 2])
with col1:
    if st.button('Generate candidate pool'):
        st.session_state['candidate_pool'] = recommender.get_recent_popular_by_genres(
            genres=selected_genres,
            n=15,
            min_year=recent_year_floor,
        )

candidate_pool = st.session_state.get('candidate_pool')
if candidate_pool is not None and not candidate_pool.empty:
    with col2:
        st.subheader('Candidate pool')
        st.dataframe(
            candidate_pool[['title_clean', 'release_year', 'popularity', 'vote_average']].rename(columns={'title_clean': 'title'}),
            use_container_width=True,
        )

    st.subheader('Step 2: pick 3 to 5 favorites')
else:
    st.info('Pick genres and generate a candidate pool, or skip straight to manual movie selection.')

all_titles = recommender.titles()
seed_titles = st.multiselect(
    'Favorite movies',
    options=all_titles,
    default=[],
    max_selections=5,
    placeholder='Select 3 to 5 movies',
)

if 0 < len(seed_titles) < 3:
    st.warning('Choose at least 3 favorites for better results.')

if st.button('Get recommendations', type='primary', disabled=len(seed_titles) < 3):
    st.session_state['results'] = recommender.recommend(seed_titles=seed_titles, k=recommendation_count)

results = st.session_state.get('results', [])
if results:
    st.subheader('Recommendations')
    for i, item in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f'### {i}. {item.title}')
            st.write(f'**Score:** {item.score:.4f}')
            st.write(f'**Genres:** {item.genres}')
            st.write(f'**Release year:** {item.release_year}')
            st.write(f'**Why it matched:** {item.reason}')
            if item.overview:
                st.write(item.overview)

st.divider()
with st.expander('What changed in this build'):
    st.markdown(
        '''
- safer CSV loading for malformed rows
- import handling that works better in Docker and local runs
- preprocessing aligned to your uploaded movie dataset columns
- duplicate-title cleanup and more resilient metadata parsing
        '''
    )
