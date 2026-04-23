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

st.set_page_config(page_title='Movie Recommender v5', page_icon='🎬', layout='wide')


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 20px;
            background:
                radial-gradient(circle at top right, rgba(255, 196, 0, 0.22), transparent 28%),
                linear-gradient(135deg, rgba(12, 23, 42, 0.96), rgba(20, 45, 77, 0.92));
            color: white;
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0;
            max-width: 52rem;
            color: rgba(255, 255, 255, 0.86);
        }
        .pill {
            display: inline-block;
            margin-right: 0.55rem;
            margin-top: 0.9rem;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            font-size: 0.9rem;
        }
        .section-card {
            padding: 1rem 1.1rem 0.8rem 1.1rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.82);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_or_train_recommender(dataset_path: str, model_path: str):
    dataset = Path(dataset_path)
    artifact = Path(model_path)

    if artifact.exists():
        try:
            recommender = MovieRecommender.load(artifact)
            if not recommender.uses_deep_model:
                raise ValueError('Existing artifact did not contain deep embeddings.')
            return recommender, None
        except Exception as exc:
            artifact.unlink(missing_ok=True)
            load_error = f'Existing model artifact was rebuilt in deep mode: {exc}'
    else:
        load_error = None

    if not dataset.exists():
        return None, f'Dataset not found: {dataset}'

    df = read_movies_csv(dataset)
    recommender = MovieRecommender().fit(df, enable_deep_training=True)
    if not recommender.uses_deep_model:
        return None, (
            'Deep model could not be built. Ensure PyTorch is installed in the runtime and rebuild the model artifact.'
        )
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
    st.session_state.setdefault('recommendation_year_floor_state', 2010)
    st.session_state.setdefault('dataset_path_state', str(DEFAULT_DATASET_PATH))
    st.session_state.setdefault('model_path_state', str(DEFAULT_EMBEDDINGS_PATH))
    st.session_state.setdefault('recommendation_count_state', 10)


def reset_recommendation_state():
    st.session_state['active_recommendations'] = []
    st.session_state['feedback'] = {}


def clear_experience_state():
    st.session_state['candidate_pool_df'] = None
    st.session_state['candidate_pool_titles'] = []
    st.session_state['selected_seed_titles'] = []
    reset_recommendation_state()


def build_feedback_lists():
    liked = [title for title, label in st.session_state['feedback'].items() if label == 'liked']
    disliked = [title for title, label in st.session_state['feedback'].items() if label == 'disliked']
    dismissed = [title for title, label in st.session_state['feedback'].items() if label == 'dismissed']
    return liked, disliked, dismissed


def fill_recommendations(recommender: MovieRecommender, threshold: int):
    seeds = list(st.session_state.get('selected_seed_titles', []))
    genres = list(st.session_state.get('selected_genres_state', []))
    languages = list(st.session_state.get('selected_languages_state', []))
    recommendation_year_floor = st.session_state.get('recommendation_year_floor_state')
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
        min_year=recommendation_year_floor,
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


def regenerate_pool(
    recommender: MovieRecommender,
    selected_genres: list[str],
    selected_languages: list[str],
    recent_year_floor: int,
):
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


def render_header(recommender: MovieRecommender):
    st.markdown(
        f"""
        <div class="hero">
            <h1>Movie Recommender v5</h1>
            <p>
                Deep-learning recommendations powered by a neural embedding model layered on top of your
                movie metadata. Startup now expects a deep model and rebuilds old non-deep artifacts automatically.
            </p>
            <span class="pill">{len(recommender.titles())} eligible movies</span>
            <span class="pill">Neural hybrid similarity</span>
            <span class="pill">{len(recommender.language_options())} language options</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.header('Settings')
        with st.form('settings_form', border=False):
            dataset_path = st.text_input('Dataset CSV path', value=st.session_state['dataset_path_state'])
            model_path = st.text_input('Serialized model path', value=st.session_state['model_path_state'])
            recommendation_count = st.slider(
                'Number of recommendations',
                min_value=5,
                max_value=20,
                value=st.session_state['recommendation_count_state'],
            )
            recommendation_year_floor = st.slider(
                'Only recommend movies from year',
                min_value=1950,
                max_value=2026,
                value=st.session_state['recommendation_year_floor_state'],
            )
            submitted = st.form_submit_button('Apply settings', use_container_width=True)

        if submitted:
            settings_changed = (
                dataset_path != st.session_state['dataset_path_state']
                or model_path != st.session_state['model_path_state']
                or recommendation_count != st.session_state['recommendation_count_state']
                or recommendation_year_floor != st.session_state['recommendation_year_floor_state']
            )
            st.session_state['dataset_path_state'] = dataset_path
            st.session_state['model_path_state'] = model_path
            st.session_state['recommendation_count_state'] = recommendation_count
            st.session_state['recommendation_year_floor_state'] = recommendation_year_floor
            if settings_changed:
                clear_experience_state()
                st.cache_resource.clear()
                st.rerun()

        if st.button('Clear candidate pool and feedback', use_container_width=True):
            clear_experience_state()
            st.rerun()

        st.caption('This app now starts in deep-model mode and rebuilds older baseline artifacts automatically.')


def render_feedback_summary():
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


init_state()
inject_styles()
render_sidebar()

with st.spinner('Loading recommender...'):
    recommender, status_message = load_or_train_recommender(
        st.session_state['dataset_path_state'],
        st.session_state['model_path_state'],
    )

if recommender is None:
    st.error(status_message or 'Could not load the dataset.')
    st.stop()

render_header(recommender)

if status_message:
    st.warning(status_message)

left_metric, mid_metric, right_metric = st.columns(3)
left_metric.metric('Recommendation mode', 'Neural hybrid')
mid_metric.metric('Catalog size', f'{len(recommender.titles()):,}')
right_metric.metric('Default list size', st.session_state['recommendation_count_state'])

all_genres = sorted({genre for row in recommender.df['genres'] for genre in row if str(genre).strip()})
all_languages = recommender.language_options()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader('1. Build your candidate pool')
st.caption('Lock in genre, language, and recency filters first. Using a form keeps the app fast while you adjust options.')
with st.form('candidate_pool_form'):
    selected_genres = st.multiselect(
        'Genres',
        options=all_genres,
        default=st.session_state.get('selected_genres_state', []),
        max_selections=3,
        placeholder='Pick up to three genres',
    )
    selected_languages = st.multiselect(
        'Movie language',
        options=all_languages,
        default=st.session_state.get('selected_languages_state', ['English']),
        placeholder='Choose one or more languages',
    )
    recent_year_floor = st.slider(
        'Only show candidate movies from year',
        min_value=1950,
        max_value=2026,
        value=st.session_state.get('recent_year_floor_state', 2016),
    )
    pool_submitted = st.form_submit_button('Generate candidate pool', type='primary')

if pool_submitted:
    st.session_state['selected_genres_state'] = selected_genres
    st.session_state['selected_languages_state'] = selected_languages
    st.session_state['recent_year_floor_state'] = recent_year_floor
    regenerate_pool(recommender, selected_genres, selected_languages, recent_year_floor)
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

candidate_pool = st.session_state.get('candidate_pool_df')
if candidate_pool is None or candidate_pool.empty:
    st.info('Choose your filters and generate a candidate pool to start. This keeps the seed choices focused and the results sharper.')
    st.stop()

st.markdown('<div class="section-card">', unsafe_allow_html=True)
pool_col, summary_col = st.columns([2.1, 1])
with pool_col:
    st.subheader('Candidate pool')
    st.dataframe(
        candidate_pool[['title_clean', 'language_label', 'release_year', 'popularity', 'vote_average']].rename(
            columns={'title_clean': 'Title', 'language_label': 'Language', 'release_year': 'Year', 'popularity': 'Popularity', 'vote_average': 'Rating'}
        ),
        use_container_width=True,
        hide_index=True,
    )
with summary_col:
    st.subheader('Pool summary')
    st.write(f'Genres: {", ".join(st.session_state["selected_genres_state"]) if st.session_state["selected_genres_state"] else "Any"}')
    st.write(f'Languages: {", ".join(st.session_state["selected_languages_state"]) if st.session_state["selected_languages_state"] else "Any"}')
    st.write(f'Recent from: {st.session_state["recent_year_floor_state"]}')
    st.write(f'Candidates: {len(candidate_pool)}')
st.markdown('</div>', unsafe_allow_html=True)

candidate_titles = st.session_state.get('candidate_pool_titles', [])

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader('2. Choose your seed movies')
st.caption('Pick 3 to 5 favorites from the candidate pool only. Submitting once avoids unnecessary reruns while you choose.')
with st.form('seed_form'):
    selected_seed_titles = st.multiselect(
        'Favorite movies from candidate pool',
        options=candidate_titles,
        default=st.session_state.get('selected_seed_titles', []),
        max_selections=5,
        placeholder='Select 3 to 5 titles from the candidate pool',
    )
    recommendation_requested = st.form_submit_button('Get recommendations', type='primary')

st.session_state['selected_seed_titles'] = selected_seed_titles

if 0 < len(selected_seed_titles) < 3:
    st.warning('Choose at least 3 favorites for better results.')

if recommendation_requested:
    if len(selected_seed_titles) < 3:
        st.warning('Choose at least 3 favorites before requesting recommendations.')
    else:
        reset_recommendation_state()
        fill_recommendations(recommender, st.session_state['recommendation_count_state'])
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

active_results: list[RecommendationResult] = st.session_state.get('active_recommendations', [])
if active_results:
    st.subheader('Recommendations')
    st.caption(
        f'Use the feedback buttons to refine the list. Recommendations are limited to titles from '
        f'{st.session_state["recommendation_year_floor_state"]} or newer.'
    )

    for index, item in enumerate(active_results[: st.session_state['recommendation_count_state']], start=1):
        with st.container(border=True):
            title_col, meta_col = st.columns([2.2, 1])
            with title_col:
                st.markdown(f'### {index}. {item.title}')
                if item.overview:
                    st.write(item.overview)
            with meta_col:
                st.metric('Match score', f'{item.score:.3f}')
                st.write(f'**Year:** {item.release_year}')
                st.write(f'**Language:** {item.language}')
                st.write(f'**Genres:** {item.genres}')

            st.write(f'**Why it matched:** {item.reason}')

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button('Good pick', key=f'like_{item.title}', use_container_width=True):
                    handle_feedback(item.title, 'liked', recommender, st.session_state['recommendation_count_state'])
                    st.rerun()
            with c2:
                if st.button('Not for me', key=f'dislike_{item.title}', use_container_width=True):
                    handle_feedback(item.title, 'disliked', recommender, st.session_state['recommendation_count_state'])
                    st.rerun()
            with c3:
                if st.button('Skip', key=f'dismiss_{item.title}', use_container_width=True):
                    handle_feedback(item.title, 'dismissed', recommender, st.session_state['recommendation_count_state'])
                    st.rerun()
else:
    st.info('After you choose 3 to 5 movies, submit the form to generate recommendations.')

render_feedback_summary()
