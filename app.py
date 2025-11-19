"""
Movie Mixer - A hybrid movie recommendation system
Main Streamlit application entry point
"""

import streamlit as st
import sys
import os

# Streamlit runtime check
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None


def _is_running_with_streamlit() -> bool:
    """Check if the script is being run with Streamlit."""
    if get_script_run_ctx is None:
        return bool(os.environ.get("STREAMLIT_SERVER_PORT"))
    try:
        return get_script_run_ctx() is not None
    except RuntimeError:
        return False


# Ensure the script is run with Streamlit
if __name__ == "__main__" and not _is_running_with_streamlit():
    sys.stderr.write(
        "\nThis project is a Streamlit app. Run it with 'streamlit run app.py' instead of 'python app.py'.\n"
    )
    raise SystemExit(1)

# Import application modules
from src.data_loader import load_movies_data
from src.preprocess import preprocess_movies
from src.model import MovieRecommender
from src.interface import (
    apply_custom_css,
    render_header,
    select_movies,
    display_selected_movies,
    render_recommendation_button,
    display_recommendations
)


@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess movie data (cached for performance).
    
    Returns:
        Preprocessed movies DataFrame
    """
    movies = load_movies_data()
    movies = preprocess_movies(movies)
    return movies


@st.cache_resource
def initialize_recommender(_movies):
    """
    Initialize and fit the movie recommender (cached for performance).
    
    Args:
        _movies: Preprocessed movies DataFrame (underscore prefix prevents hashing)
        
    Returns:
        Fitted MovieRecommender instance
    """
    recommender = MovieRecommender()
    recommender.fit(_movies)
    return recommender


def main():
    """Main application logic."""
    # Apply custom styling
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Load and prepare data
    movies = load_and_preprocess_data()
    recommender = initialize_recommender(movies)
    
    # Movie selection interface
    movie_titles = movies['title'].tolist()
    selected_title1, selected_title2 = select_movies(movie_titles)
    
    # Get movie IDs from selected titles
    movie_id1 = movies[movies['title'] == selected_title1]['id'].values[0]
    movie_id2 = movies[movies['title'] == selected_title2]['id'].values[0]
    
    # Display selected movies
    display_selected_movies(movies, movie_id1, movie_id2)
    
    # Recommendation button and results
    if render_recommendation_button():
        recommendations = recommender.recommend(movie_id1, movie_id2, top_n=3)
        
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            display_recommendations(recommendations)


if __name__ == "__main__":
    main()
