"""
Streamlit UI module for Movie Mixer.
Handles all user interface components and layout.
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple
from src.tmdb_api import get_movie_poster


def apply_custom_css() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    .centered-text {text-align: center;}
    .movie-title {text-align: center; font-size: 22px; font-weight: bold; margin-bottom: 6px;}
    .movie-overview {text-align: center; font-size: 14px; margin-top: 6px;}
    .movie-genres {text-align: center; font-size: 13px; font-style: italic; color: #888; margin-top: 4px;}
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    """Render the main header and introduction."""
    st.title("Movie Mixer")
    st.write("Type movie names and select the correct ones:")


def select_movies(movie_titles: List[str]) -> Tuple[str, str]:
    """
    Render movie selection dropdowns.
    
    Args:
        movie_titles: List of all available movie titles
        
    Returns:
        Tuple of (first_selected_title, second_selected_title)
    """
    selected_title1 = st.selectbox("First Movie", options=movie_titles)
    selected_title2 = st.selectbox("Second Movie", options=movie_titles)
    
    return selected_title1, selected_title2


def display_movie_card(movie_row: pd.Series, col) -> None:
    """
    Display a single movie card with poster, title, and details.
    
    Args:
        movie_row: DataFrame row containing movie information
        col: Streamlit column object to render in
    """
    poster_url = get_movie_poster(movie_row['id'])
    genres = movie_row['genres_str']
    
    with col:
        st.markdown(f"### {movie_row['title']} ({movie_row['vote_average']})")
        st.markdown(f"**Genres:** {genres}")
        if poster_url:
            st.image(poster_url, width=220)
        st.write(movie_row['overview'])


def display_selected_movies(movies: pd.DataFrame, movie_id1: int, movie_id2: int) -> None:
    """
    Display the two selected movies side by side.
    
    Args:
        movies: Complete movies DataFrame
        movie_id1: ID of the first selected movie
        movie_id2: ID of the second selected movie
    """
    st.subheader("Selected Movies")
    cols = st.columns(2)
    
    for col, movie_id in zip(cols, [movie_id1, movie_id2]):
        movie_row = movies[movies['id'] == movie_id].iloc[0]
        display_movie_card(movie_row, col)


def display_recommendations(recommendations: pd.DataFrame) -> None:
    """
    Display recommended movies in a grid layout.
    
    Args:
        recommendations: DataFrame containing recommended movies
    """
    st.subheader("Top Recommendations")
    rec_cols = st.columns(len(recommendations))
    
    for col, (_, row) in zip(rec_cols, recommendations.iterrows()):
        poster_url = get_movie_poster(row['id'])
        genres = row['genres_str']
        
        with col:
            st.markdown(f"### {row['title']} ({row['vote_average']})")
            st.markdown(f"**Genres:** {genres}")
            if poster_url:
                st.image(poster_url, width=180)
            st.write(row['overview'])


def render_recommendation_button() -> bool:
    """
    Render the 'Get Recommendations' button.
    
    Returns:
        True if button was clicked, False otherwise
    """
    return st.button("Get Recommendations")
