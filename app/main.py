from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
from app.recommender import MovieRecommender

st.set_page_config(page_title='Movie Recommender v1', page_icon='🎬', layout='wide')


@st.cache_resource
def load_or_train_recommender(dataset_path: str, model_path: str):
    dataset = Path(dataset_path)
    artifact = Path(model_path)

    if artifact.exists():
        return MovieRecommender.load(artifact)

    if not dataset.exists():
        return None

    df = pd.read_csv(dataset)
    recommender = MovieRecommender().fit(df)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(artifact)
    return recommender


def render_header():
    st.title('🎬 Cross-Platform Movie Recommender v1')
    st.caption('Metadata-driven MVP with a Streamlit UI, Docker packaging, and a clean path to a deeper neural ranking model later.')


render_header()

with st.sidebar:
    st.header('Settings')
    dataset_path = st.text_input('Dataset CSV path', str(DEFAULT_DATASET_PATH))
    model_path = st.text_input('Serialized model path', str(DEFAULT_EMBEDDINGS_PATH))
    recommendation_count = st.slider('Number of recommendations', min_value=5, max_value=15, value=10)
    recent_year_floor = st.slider('Only show candidate movies from year', min_value=1980, max_value=2026, value=2016)

recommender = load_or_train_recommender(dataset_path, model_path)

if recommender is None:
    st.error('No dataset was found yet. Add your CSV to data/movies.csv or point the sidebar to your file path.')
    st.code('docker run --rm -p 8501:8501 -v $(pwd)/data:/app/data movie-recommender-v1')
    st.stop()

st.success(f'Loaded {len(recommender.titles())} movies.')

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
        st.dataframe(candidate_pool[['title_clean', 'release_year', 'popularity', 'vote_average']].rename(columns={'title_clean': 'title'}), use_container_width=True)

    st.subheader('Step 2: pick 3 to 5 favorites from the pool, or choose from the full catalog')
    pool_titles = candidate_pool['title_clean'].tolist()
else:
    st.info('Pick genres and generate a candidate pool, or skip straight to manual movie selection.')
    pool_titles = []

all_titles = recommender.titles()
default_options = pool_titles if pool_titles else all_titles[:100]
seed_titles = st.multiselect('Favorite movies', options=all_titles, default=[], max_selections=5, placeholder='Select 3 to 5 movies')

if len(seed_titles) and len(seed_titles) < 3:
    st.warning('For the proposal flow, choose at least 3 favorites.')

if st.button('Get recommendations', type='primary', disabled=len(seed_titles) < 3):
    results = recommender.recommend(seed_titles=seed_titles, k=recommendation_count)
    st.session_state['results'] = results

results = st.session_state.get('results', [])
if results:
    st.subheader('Recommendations')
    for i, item in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {item.title}")
            st.write(f"**Score:** {item.score:.4f}")
            st.write(f"**Genres:** {item.genres}")
            st.write(f"**Release year:** {item.release_year}")
            st.write(f"**Why it matched:** {item.reason}")
            if item.overview:
                st.write(item.overview)

st.divider()
with st.expander('What is in v1'):
    st.markdown('''
- Streamlit interface for genre filtering, seed selection, and ranked recommendations
- Dockerized deployment
- Metadata-driven retrieval using combined text + numeric movie features
- Serialized local artifact for fast reuse
- Clean module split so a PyTorch embedding model can replace the TF-IDF step in v2
    ''')
