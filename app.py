import pickle
import random
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA_PATH = "data"

WEATHER_GENRE_MAP = {
    "rainy": ["Drama", "Romance", "Thriller", "Mystery", "Film-Noir"],
    "snowy": ["Animation", "Children", "Fantasy", "Musical", "Comedy"],
    "sunny": ["Action", "Adventure", "Comedy"],
}

DROP_COLS = [
    "userId", "movieId", "watched", "user_peak_season",
    "movie_peak_season", "movie_rating_count", "user_rating_count",
]

RAINY_CODES  = {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}
SNOWY_CODES  = {71, 73, 75, 77, 85, 86}


@st.cache_data(ttl=1800)
def fetch_weather(lat=43.1566, lon=-77.6088):
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=5,
        )
        data = resp.json()["current_weather"]
        code = int(data["weathercode"])
        temp = float(data["temperature"])
        if code in RAINY_CODES:
            return "rainy"
        if code in SNOWY_CODES:
            return "snowy"
        if code in {0, 1} and temp >= 28:
            return "sunny"
    except Exception:
        pass
    return "none"


@st.cache_resource
def load_models():
    with open(f"{DATA_PATH}/xgb_model_v2.pkl", "rb") as f:
        propensity_model = pickle.load(f)
    with open(f"{DATA_PATH}/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    return propensity_model, svd_model


@st.cache_data
def load_data():
    full = pd.read_parquet(f"{DATA_PATH}/full_dataset.parquet")
    movies = pd.read_csv(f"{DATA_PATH}/movies.csv")
    movies["movieId"] = movies["movieId"].astype(str)
    links = pd.read_csv(f"{DATA_PATH}/links.csv", dtype={"movieId": str, "tmdbId": str})
    popular = (
        full[full["movie_rating_count"] >= 500]
        .drop_duplicates("movieId")
        .copy()
    )
    return full, movies, links, popular


@st.cache_data
def get_poster_url(tmdb_id, api_key):
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": api_key},
            timeout=5,
        )
        path = resp.json().get("poster_path")
        if path:
            return f"https://image.tmdb.org/t/p/w300{path}"
    except Exception:
        pass
    return None


def render_movie_row(movies_df, links, api_key):
    merged = movies_df.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    cards = ""
    for _, row in merged.iterrows():
        poster = None
        if pd.notna(row.get("tmdbId")):
            poster = get_poster_url(str(row["tmdbId"]), api_key)
        if not poster:
            poster = "https://placehold.co/150x225/1a1a1a/666?text=No+Image"

        genres = "  ·  ".join(row["genres"].split("|")[:3]) if pd.notna(row.get("genres")) else ""
        title = str(row["title"]).replace('"', "&quot;").replace("<", "&lt;")

        cards += f"""
        <div style="flex:0 0 150px;text-align:left">
            <img src="{poster}" style="width:150px;height:225px;object-fit:cover;border-radius:6px;display:block">
            <p style="font-size:0.72rem;color:#ccc;margin:6px 0 2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:150px" title="{title}">{title}</p>
            <p style="font-size:0.65rem;color:#888;margin:0">{genres}</p>
        </div>
        """

    components.html(
        f'''
        <div style="background:#0e1117;padding:4px 0 16px 0">
            <div style="display:flex;gap:14px;overflow-x:auto;padding:4px 0;scrollbar-width:thin">
                {cards}
            </div>
        </div>
        ''',
        height=300,
    )


def time_relevance_score(u_hour, u_season, m_hour, m_season, rating_count):
    strength = min(rating_count / 200, 1.0)
    hour_match = 1.0 if abs(u_hour - m_hour) <= 2 else 0.8
    season_match = 1.0 if u_season == m_season else 0.9
    raw = hour_match * season_match
    return 1.0 + (raw - 1.0) * strength


def get_recommendations(user_id, propensity_model, svd_model, full, popular, movies, n=15):
    user_rows = full[full["userId"] == user_id]
    if user_rows.empty:
        return None, None

    u = user_rows.iloc[0]
    watched = set(user_rows[user_rows["watched"] == 1]["movieId"].tolist())
    candidates = popular[~popular["movieId"].isin(watched)].copy()
    if candidates.empty:
        return None, None

    # Signal 1: Propensity
    X = candidates.drop(columns=DROP_COLS, errors="ignore").fillna(0)
    candidates["propensity_score"] = propensity_model.predict_proba(X)[:, 1]

    # Signal 2: SVD
    user_idx = svd_model.train_set.uid_map.get(str(user_id))
    if user_idx is not None:
        svd_scores = svd_model.score(user_idx)
        idx_to_item = {v: k for k, v in svd_model.train_set.iid_map.items()}
        score_map = {idx_to_item[i]: svd_scores[i] for i in range(len(svd_scores))}
        candidates["svd_score"] = (
            candidates["movieId"].astype(str).map(score_map).fillna(svd_scores.mean())
        )
        smin, smax = svd_scores.min(), svd_scores.max()
        candidates["svd_score"] = (candidates["svd_score"] - smin) / (smax - smin)
    else:
        candidates["svd_score"] = 0.5

    # Signal 3: Time
    candidates["time_score"] = candidates.apply(
        lambda r: time_relevance_score(
            u["user_peak_hour"], u["user_peak_season"],
            r["movie_peak_hour"], r["movie_peak_season"],
            u["user_rating_count"],
        ),
        axis=1,
    )

    candidates["final_score"] = (
        candidates["propensity_score"]
        * candidates["svd_score"]
        * candidates["time_score"]
    )

    top = candidates.nlargest(n, "final_score")[["movieId", "final_score"]].copy()
    top["movieId"] = top["movieId"].astype(str)
    top = top.merge(movies, on="movieId")[["movieId", "title", "genres"]]

    # User profile
    genre_cols = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    watched_rows = user_rows[user_rows["watched"] == 1]
    top_genres = []
    if not watched_rows.empty:
        existing = [c for c in genre_cols if c in watched_rows.columns]
        genre_sums = watched_rows[existing].sum().sort_values(ascending=False)
        top_genres = genre_sums[genre_sums > 0].head(3).index.tolist()

    profile = {
        "total_ratings": int(u["user_rating_count"]),
        "avg_rating": round(float(u["user_avg_rating"]), 2),
        "peak_hour": int(u["user_peak_hour"]),
        "peak_season": u["user_peak_season"].capitalize(),
        "top_genres": top_genres,
    }

    return top, profile


def get_weather_picks(condition, popular, movies, n=10):
    genres = WEATHER_GENRE_MAP[condition]
    pool = popular.copy()
    mask = (pool[genres].sum(axis=1) > 0) & (pool["movie_avg_rating"] >= 3.5)
    pool = pool[mask]
    if pool.empty:
        return None
    sample = pool.sample(min(n, len(pool)))[["movieId"]].copy()
    sample["movieId"] = sample["movieId"].astype(str)
    return sample.merge(movies, on="movieId")[["movieId", "title", "genres"]]


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NextUp", layout="wide")

api_key = st.secrets["TMDB_API_KEY"]

st.markdown(
    '<p style="color:red;font-size:2rem;font-weight:bold;margin:0">NextUp</p>',
    unsafe_allow_html=True,
)
st.divider()

with st.spinner("Loading models and data..."):
    propensity_model, svd_model = load_models()
    full, movies, links, popular = load_data()

# Controls + stats row
col1, col2, col3 = st.columns([1, 1, 3])

if "user_id" not in st.session_state:
    st.session_state.user_id = random.randint(1, 162541)

with col1:
    uid_col, btn_col = st.columns([3, 1])
    with uid_col:
        typed = st.text_input("User ID", value=str(st.session_state.user_id))
        try:
            st.session_state.user_id = int(typed)
        except ValueError:
            pass
    with btn_col:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        if st.button("↺"):
            st.session_state.user_id = random.randint(1, 162541)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
user_id = st.session_state.user_id

with col2:
    weather_options = ["none", "rainy", "snowy", "sunny"]
    detected = fetch_weather()
    weather = st.selectbox("Weather", weather_options, index=weather_options.index(detected))
    if detected != "none":
        st.markdown("<p style='font-size:0.6rem;color:#666;margin-top:-10px'>Auto detected. Change manually to explore.</p>", unsafe_allow_html=True)

recs, profile = get_recommendations(user_id, propensity_model, svd_model, full, popular, movies)

with col3:
    if profile:
        st.markdown("**User Stats**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Ratings", profile["total_ratings"])
        s2.metric("Avg Rating", profile["avg_rating"])
        s3.metric("Peak Hour", f"{profile['peak_hour']}:00")
        s4.metric("Season", profile["peak_season"])
        if profile["top_genres"]:
            st.caption("Top genres: " + "  ·  ".join(profile["top_genres"]))
    else:
        st.warning("User not found.")

st.divider()

# Top picks
st.subheader("Top picks for you")
if recs is not None:
    render_movie_row(recs, links, api_key)
else:
    st.write("No recommendations available for this user.")

# Weather picks
if weather != "none":
    st.divider()
    st.subheader(f"It's {weather}, watch these")
    weather_recs = get_weather_picks(weather, popular, movies)
    if weather_recs is not None:
        render_movie_row(weather_recs, links, api_key)
