"""
TMDB API module for Movie Mixer.
Handles TMDB API configuration and movie poster fetching.
"""

import requests
import os
from typing import Tuple, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TMDB API Bearer Token from environment
TMDB_TOKEN = os.getenv("TMDB_TOKEN")

if not TMDB_TOKEN:
    raise ValueError(
        "TMDB_TOKEN not found in environment variables. "
        "Please create a .env file with your TMDB API token."
    )

# Request headers for TMDB API
HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json"
}


def get_tmdb_config() -> Tuple[str, List[str]]:
    """
    Fetch TMDB API configuration for image URLs.
    
    Returns:
        Tuple of (base_url, poster_sizes) where:
        - base_url: Base URL for image requests
        - poster_sizes: List of available poster size options
    """
    url = "https://api.themoviedb.org/3/configuration"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    
    base_url = data["images"]["secure_base_url"]
    poster_sizes = data["images"]["poster_sizes"]
    
    return base_url, poster_sizes


def get_movie_poster(tmdb_id: int, size: str = "w500") -> Optional[str]:
    """
    Fetch the poster URL for a movie from TMDB.
    
    Args:
        tmdb_id: The TMDB movie ID
        size: Poster size (e.g., 'w500', 'w780', 'original')
        
    Returns:
        Full URL to the movie poster image, or None if not available
    """
    # Get TMDB configuration (cached in production)
    base_url, _ = get_tmdb_config()
    
    # Fetch movie images
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images?language=en"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    
    # Return first poster if available
    if data.get("posters"):
        poster_path = data["posters"][0]["file_path"]
        return f"{base_url}{size}{poster_path}"
    
    return None
