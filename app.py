# ---------------------------
# 0. IMPORT STREAMLIT
# ---------------------------
import streamlit as st
import sys
import os

# ---------------------------
# 1. IMPORT LIBRARIES
# ---------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import requests

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:  # Older Streamlit versions
    get_script_run_ctx = None


def _is_running_with_streamlit() -> bool:
    if get_script_run_ctx is None:
        return bool(os.environ.get("STREAMLIT_SERVER_PORT"))
    try:
        return get_script_run_ctx() is not None
    except RuntimeError:
        return False


if __name__ == "__main__" and not _is_running_with_streamlit():
    sys.stderr.write(
        "\nThis project is a Streamlit app. Run it with 'streamlit run app.py' instead of 'python app.py'.\n"
    )
    raise SystemExit(1)

# ---------------------------
# 2. LOAD CSV FILE
# ---------------------------
movies_file = "movies.csv"  # Your movies CSV
keywords_file = "keywords.csv"  # New keywords CSV

movie_read_kwargs = {
    "dtype": {"popularity": "string"},
    "low_memory": False,
}

try:
    movies = pd.read_csv(movies_file, encoding='utf-8', **movie_read_kwargs)
except UnicodeDecodeError:
    movies = pd.read_csv(movies_file, encoding='latin1', **movie_read_kwargs)

try:
    keywords_df = pd.read_csv(keywords_file, encoding='utf-8')
except:
    keywords_df = pd.read_csv(keywords_file, encoding='latin1')

# Keep movies with enough votes
movies = movies[movies['vote_count'] >= 1000].copy()

# Clean ID
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
movies = movies.reset_index(drop=True)

keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
keywords_df = keywords_df.dropna(subset=['id'])
keywords_df['id'] = keywords_df['id'].astype(int)
keywords_df['keywords'] = keywords_df['keywords'].fillna('[]')

# ---------------------------
# 3. CREATE COMBINED FEATURE
# ---------------------------
def clean_genres(genres):
    if pd.isna(genres) or genres == '[]':
        return ''
    if isinstance(genres, str):
        try:
            genre_list = ast.literal_eval(genres)
            return ' '.join([g['name'] for g in genre_list])
        except:
            return ''
    return ''

def clean_keywords(kw):
    if pd.isna(kw) or kw == '[]':
        return ''
    if isinstance(kw, str):
        try:
            kw_list = ast.literal_eval(kw)
            return ' '.join([k['name'] for k in kw_list])
        except:
            return ''
    return ''

movies['genres_str'] = movies['genres'].apply(clean_genres)
keywords_df['keywords_str'] = keywords_df['keywords'].apply(clean_keywords)

# Merge keywords into movies
movies = movies.merge(keywords_df[['id', 'keywords_str']], on='id', how='left')
movies['keywords_str'] = movies['keywords_str'].fillna('')

# Combine overview, genres, and keywords for TF-IDF
movies['combined'] = (
    #movies['overview'].fillna('')  + ' ' + 
    movies['genres_str'].fillna('')  + ' '  + 
    movies['keywords_str'] 
)

# ---------------------------
# 4. BUILD TF-IDF MATRIX
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
id_to_index = pd.Series(movies.index, index=movies['id']).to_dict()

# ---------------------------
# 5. TMDB POSTER FUNCTIONS
# ---------------------------
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3ZTljZmNkZTc2MTQ3NmU4Y2FmM2NmNDQ3MDM4Zjc5MSIsIm5iZiI6MTc2MzQ5Mjc0MC4xNjksInN1YiI6IjY5MWNjMzg0ZTAyZGY1NDk1OWY2MjlkNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.mxzQQXYCIOQP1b9WXVO2JUn2LJPn1P0SKhJrnFDzLx8"
headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}

def get_tmdb_config():
    url = "https://api.themoviedb.org/3/configuration"
    response = requests.get(url, headers=headers)
    data = response.json()
    base_url = data["images"]["secure_base_url"]
    poster_sizes = data["images"]["poster_sizes"]
    return base_url, poster_sizes

BASE_URL, POSTER_SIZES = get_tmdb_config()
def get_movie_poster(tmdb_id, size="w500"):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images?language=en"
    response = requests.get(url, headers=headers)
    data = response.json()
    if data.get("posters"):
        poster_path = data["posters"][0]["file_path"]
        return f"{BASE_URL}{size}{poster_path}"
    else:
        return None
# ---------------------------
# 6. RECOMMENDATION FUNCTION (Updated)
# ---------------------------
import numpy as np

def recommend_movies(movie_id1, movie_id2, top_n=3):
    if movie_id1 not in id_to_index or movie_id2 not in id_to_index:
        return "One of the movie IDs not found."

    idx1 = id_to_index[movie_id1]
    idx2 = id_to_index[movie_id2]

    # Get TF-IDF vectors for both movies
    vec1 = tfidf_matrix[idx1].toarray()
    vec2 = tfidf_matrix[idx2].toarray()

    # Element-wise minimum (soft intersection)
    hybrid_vector = np.minimum(vec1, vec2)

    # Compute similarity with all movies
    similarities = cosine_similarity(tfidf_matrix, hybrid_vector).flatten()

    # Remove the original movies
    similarities[idx1] = -1
    similarities[idx2] = -1

    # Get top N recommendations
    top_indices = similarities.argsort()[-top_n:][::-1]
    recs = movies.iloc[top_indices][['id', 'title', 'overview', 'vote_average', 'poster_path', 'genres_str']]

    return recs


# ---------------------------
# 7. STREAMLIT UI
# ---------------------------
st.markdown("""
<style>
.centered-text {text-align: center;}
.movie-title {text-align: center; font-size: 22px; font-weight: bold; margin-bottom: 6px;}
.movie-overview {text-align: center; font-size: 14px; margin-top: 6px;}
.movie-genres {text-align: center; font-size: 13px; font-style: italic; color: #888; margin-top: 4px;}
</style>
""", unsafe_allow_html=True)

st.title("Movie Recommender")
st.write("Type movie names and select the correct ones:")

movie_titles = movies['title'].tolist()
selected_title1 = st.selectbox("First Movie", options=movie_titles)
selected_title2 = st.selectbox("Second Movie", options=movie_titles)

movie_id1 = movies[movies['title'] == selected_title1]['id'].values[0]
movie_id2 = movies[movies['title'] == selected_title2]['id'].values[0]

st.subheader("Selected Movies")
cols = st.columns(2)

for col, movie_id, title in zip(cols, [movie_id1, movie_id2], [selected_title1, selected_title2]):
    movie_row = movies[movies['id'] == movie_id].iloc[0]
    poster_url = get_movie_poster(movie_id)
    genres = movie_row['genres_str']

    with col:
        st.markdown(f"### {movie_row['title']} ({movie_row['vote_average']})")
        st.markdown(f"**Genres:** {genres}")
        if poster_url:
            st.image(poster_url, width=220)
        st.write(movie_row['overview'])

if st.button("Get Recommendations"):
    recs = recommend_movies(movie_id1, movie_id2)
    if isinstance(recs, str):
        st.warning(recs)
    else:
        st.subheader("Top Recommendations")
        rec_cols = st.columns(len(recs))
        for col, (_, row) in zip(rec_cols, recs.iterrows()):
            poster_url = get_movie_poster(row['id'])
            genres = row['genres_str']

            with col:
                st.markdown(f"### {row['title']} ({row['vote_average']})")
                st.markdown(f"**Genres:** {genres}")
                if poster_url:
                    st.image(poster_url, width=180)
                st.write(row['overview'])
