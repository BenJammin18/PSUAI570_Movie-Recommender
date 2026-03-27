from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from app.config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from app.io_utils import read_movies_csv
    from app.recommender import MovieRecommender, RecommendationResult
except ModuleNotFoundError:
    from config import DEFAULT_DATASET_PATH, DEFAULT_EMBEDDINGS_PATH
    from io_utils import read_movies_csv
    from recommender import MovieRecommender, RecommendationResult

st.set_page_config(page_title='Movie Recommender v4', page_icon='🎬', layout='wide')


@st.cache_resource(show_spinner=False)
def load_or_train_recommender(dataset_path: str, model_path: str):
    dataset = Path(dataset_path)
    artifact = Path(model_path)

    if artifact.exists():
        try:
            return MovieRecommender.load(artifact), None
        except Exception as exc:
            artifact.unlink(missing_ok=True)
            load_error = f'Existing model artifact was rebuilt: {exc}'
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


def init_state():
    st.session_state.setdefault('candidate_pool_df', None)
    st.session_state.setdefault('candidate_pool_titles', [])
    st.session_state.setdefault('selected_seed_titles', [])
    st.session_state.setdefault('active_recommendations', [])
    st.session_state.setdefault('feedback', {})
    st.session_state.setdefault('selected_genres_state', [])
    st.session_state.setdefault('selected_languages_state', ['English'])
    st.session_state.setdefault('recent_year_floor_state', 2016)


def reset_recommendation_state():
    st.session_state['active_recommendations'] = []
    st.session_state['feedback'] = {}


def build_feedback_lists():
    liked = [title for title, label in st.session_state['feedback'].items() if label == 'liked']
    disliked = [title for title, label in st.session_state['feedback'].items() if label == 'disliked']
    dismissed = [title for title, label in st.session_state['feedback'].items() if label == 'dismissed']
    return liked, disliked, dismissed


def fill_recommendations(recommender: MovieRecommender, threshold: int):
    seeds = list(st.session_state.get('selected_seed_titles', []))
    genres = list(st.session_state.get('selected_genres_state', []))
    languages = list(st.session_state.get('selected_languages_state', []))
    active: list[RecommendationResult] = list(st.session_state.get('active_recommendations', []))
    if not seeds:
        st.session_state['active_recommendations'] = []
        return

    liked, disliked, dismissed = build_feedback_lists()
    excluded = set(seeds)
    excluded.update(liked)
    excluded.update(disliked)
    excluded.update(dismissed)
    excluded.update([item.title for item in active])

    needed = max(threshold - len(active), 0)
    if needed <= 0:
        st.session_state['active_recommendations'] = active[:threshold]
        return

    new_items = recommender.recommend(
        seed_titles=seeds,
        selected_genres=genres,
        selected_languages=languages,
        k=threshold * 8,
        excluded_titles=excluded,
        liked_titles=liked,
        disliked_titles=disliked,
    )

    deduped = []
    seen = {item.title.lower() for item in active}
    for item in new_items:
        if item.title.lower() in seen:
            continue
        deduped.append(item)
        seen.add(item.title.lower())
        if len(deduped) >= needed:
            break

    st.session_state['active_recommendations'] = active + deduped


def handle_feedback(title: str, label: str, recommender: MovieRecommender, threshold: int):
    st.session_state['feedback'][title] = label
    st.session_state['active_recommendations'] = [
        item for item in st.session_state.get('active_recommendations', []) if item.title != title
    ]
    fill_recommendations(recommender, threshold)


def regenerate_pool(recommender: MovieRecommender, selected_genres: list[str], selected_languages: list[str], recent_year_floor: int):
    candidate_pool = recommender.get_recent_popular_by_genres(
        genres=selected_genres,
        languages=selected_languages,
        n=18,
        min_year=recent_year_floor,
    )
    st.session_state['candidate_pool_df'] = candidate_pool
    st.session_state['candidate_pool_titles'] = candidate_pool['title_clean'].tolist()
    st.session_state['selected_seed_titles'] = []
    reset_recommendation_state()


def render_header():
    st.title('🎬 Cross-Platform Movie Recommender v4')
    st.caption('Candidate-pool-driven recommendations with genre gating, language filtering, quality filtering, and live feedback refill.')


init_state()
render_header()

with st.sidebar:
    st.header('Settings')
    dataset_path = st.text_input('Dataset CSV path', str(DEFAULT_DATASET_PATH))
    model_path = st.text_input('Serialized model path', str(DEFAULT_EMBEDDINGS_PATH))
    recommendation_count = st.slider('Number of recommendations', min_value=5, max_value=20, value=10)
    recent_year_floor = st.slider('Only show candidate movies from year', min_value=1950, max_value=2026, value=2016)
    if st.button('Clear candidate pool and feedback'):
        st.session_state['candidate_pool_df'] = None
        st.session_state['candidate_pool_titles'] = []
        st.session_state['selected_seed_titles'] = []
        reset_recommendation_state()
        st.rerun()

recommender, status_message = load_or_train_recommender(dataset_path, model_path)

if recommender is None:
    st.error(status_message or 'Could not load the dataset.')
    st.stop()

if status_message:
    st.warning(status_message)

st.success(f'Loaded {len(recommender.titles())} eligible movies.')
all_genres = sorted({g for row in recommender.df['genres'] for g in row if str(g).strip()})
all_languages = recommender.language_options()

st.subheader('Step 1: choose up to 3 genres and your language filter')
selected_genres = st.multiselect(
    'Genres',
    options=all_genres,
    max_selections=3,
    default=st.session_state.get('selected_genres_state', []),
)
selected_languages = st.multiselect(
    'Movie language',
    options=all_languages,
    default=st.session_state.get('selected_languages_state', ['English']),
    placeholder='Choose one or more languages, like English only',
)
st.session_state['selected_genres_state'] = selected_genres
st.session_state['selected_languages_state'] = selected_languages
st.session_state['recent_year_floor_state'] = recent_year_floor

if st.button('Generate candidate pool'):
    regenerate_pool(recommender, selected_genres, selected_languages, recent_year_floor)

candidate_pool = st.session_state.get('candidate_pool_df')
if candidate_pool is None or candidate_pool.empty:
    st.info('Pick your genres and optional language filter first, then generate the candidate pool. Users can only choose seed movies from that pool.')
    st.stop()

st.subheader('Candidate pool')
display_columns = ['title_clean', 'language_label', 'release_year', 'popularity', 'vote_average']
st.dataframe(
    candidate_pool[display_columns].rename(columns={'title_clean': 'title', 'language_label': 'language'}),
    use_container_width=True,
)

candidate_titles = st.session_state.get('candidate_pool_titles', [])

st.subheader('Step 2: pick 3 to 5 favorites from the candidate pool only')
selected_seed_titles = st.multiselect(
    'Favorite movies from candidate pool',
    options=candidate_titles,
    default=st.session_state.get('selected_seed_titles', []),
    max_selections=5,
    placeholder='Select 3 to 5 titles from the candidate pool',
)
st.session_state['selected_seed_titles'] = selected_seed_titles

if 0 < len(selected_seed_titles) < 3:
    st.warning('Choose at least 3 favorites for better results.')

if st.button('Get recommendations', type='primary', disabled=len(selected_seed_titles) < 3):
    reset_recommendation_state()
    fill_recommendations(recommender, recommendation_count)

active_results: list[RecommendationResult] = st.session_state.get('active_recommendations', [])
if active_results:
    st.subheader('Recommendations')
    st.caption('Use thumbs up, thumbs down, or dismiss. The list refills automatically and stays within your slider limit.')

    for i, item in enumerate(active_results[:recommendation_count], start=1):
        with st.container(border=True):
            st.markdown(f'### {i}. {item.title}')
            st.write(f'**Score:** {item.score:.4f}')
            st.write(f'**Genres:** {item.genres}')
            st.write(f'**Language:** {item.language}')
            st.write(f'**Release year:** {item.release_year}')
            st.write(f'**Why it matched:** {item.reason}')
            if item.overview:
                st.write(item.overview)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button('👍 Good', key=f'like_{item.title}'):
                    handle_feedback(item.title, 'liked', recommender, recommendation_count)
                    st.rerun()
            with c2:
                if st.button('👎 Bad', key=f'dislike_{item.title}'):
                    handle_feedback(item.title, 'disliked', recommender, recommendation_count)
                    st.rerun()
            with c3:
                if st.button('Dismiss', key=f'dismiss_{item.title}'):
                    handle_feedback(item.title, 'dismissed', recommender, recommendation_count)
                    st.rerun()
else:
    st.info('After you choose 3 to 5 movies from the pool, click Get recommendations.')

liked, disliked, dismissed = build_feedback_lists()
with st.expander('Feedback summary'):
    st.write(f'Liked: {len(liked)}')
    if liked:
        st.write(', '.join(liked))
    st.write(f'Disliked: {len(disliked)}')
    if disliked:
        st.write(', '.join(disliked))
    st.write(f'Dismissed: {len(dismissed)}')
    if dismissed:
        st.write(', '.join(dismissed))
