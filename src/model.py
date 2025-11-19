"""
Machine learning model module for Movie Mixer.
Handles TF-IDF vectorization, cosine similarity computation, and movie recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, Union


class MovieRecommender:
    """
    Movie recommendation system based on TF-IDF and cosine similarity.
    Uses a hybrid vector approach to find movies similar to two input movies.
    """
    
    def __init__(self):
        """Initialize the recommender with empty state."""
        self.tfidf: TfidfVectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.id_to_index: Dict[int, int] = {}
        self.movies: pd.DataFrame = None
    
    def fit(self, movies: pd.DataFrame) -> None:
        """
        Fit the TF-IDF model and compute cosine similarity matrix.
        
        Args:
            movies: Preprocessed DataFrame with 'combined' feature column
        """
        self.movies = movies
        
        # Build TF-IDF matrix from combined features
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(movies['combined'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create mapping from movie ID to DataFrame index
        self.id_to_index = pd.Series(movies.index, index=movies['id']).to_dict()
    
    def recommend(self, movie_id1: int, movie_id2: int, top_n: int = 3) -> Union[pd.DataFrame, str]:
        """
        Recommend movies based on two input movies using hybrid vector approach.
        
        The algorithm:
        1. Gets TF-IDF vectors for both input movies
        2. Computes element-wise minimum (soft intersection) to create a hybrid vector
        3. Finds movies most similar to this hybrid vector
        4. Excludes the original input movies from results
        
        Args:
            movie_id1: ID of the first movie
            movie_id2: ID of the second movie
            top_n: Number of recommendations to return (default: 3)
            
        Returns:
            DataFrame with top N recommended movies and their details,
            or error message string if movie IDs not found
        """
        # Validate movie IDs exist
        if movie_id1 not in self.id_to_index or movie_id2 not in self.id_to_index:
            return "One of the movie IDs not found."
        
        # Get DataFrame indices for the movies
        idx1 = self.id_to_index[movie_id1]
        idx2 = self.id_to_index[movie_id2]
        
        # Get TF-IDF vectors for both movies
        vec1 = self.tfidf_matrix[idx1].toarray()
        vec2 = self.tfidf_matrix[idx2].toarray()
        
        # Element-wise minimum (soft intersection)
        # This captures features common to both movies
        hybrid_vector = np.minimum(vec1, vec2)
        
        # Compute similarity between hybrid vector and all movies
        similarities = cosine_similarity(self.tfidf_matrix, hybrid_vector).flatten()
        
        # Remove the original movies from consideration
        similarities[idx1] = -1
        similarities[idx2] = -1
        
        # Get indices of top N most similar movies
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return DataFrame with movie details
        recommendations = self.movies.iloc[top_indices][
            ['id', 'title', 'overview', 'vote_average', 'poster_path', 'genres_str']
        ]
        
        return recommendations
