# ============================================================================
# RECOMMENDATION UTILITY FUNCTIONS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    """Load movie data and similarity matrix with caching"""
    try:
        movies = pickle.load(open("models/movies.pkl", "rb"))
        similarity = pickle.load(open("models/similarity.pkl", "rb"))
        
        # Check if there are duplicates
        duplicate_count = movies.duplicated(subset=['Title']).sum()
        
        # Remove duplicates and keep first occurrence
        movies = movies.drop_duplicates(subset=['Title'], keep='first')
        movies = movies.reset_index(drop=True)
        
        # Match similarity matrix size
        if len(movies) > similarity.shape[0]:
            movies = movies.iloc[:similarity.shape[0]]
            movies = movies.reset_index(drop=True)
        elif len(movies) < similarity.shape[0]:
            similarity = similarity[:len(movies), :len(movies)]
        
        # Ensure all required columns exist
        required_cols = ['Title', 'Genre', 'Vote_Average', 'Popularity', 'Year', 'Poster_Url']
        for col in required_cols:
            if col not in movies.columns:
                movies[col] = 'N/A' if col == 'Poster_Url' else 0
        
        # Convert Year to int if needed
        if movies['Year'].dtype != 'int64':
            movies['Year'] = movies['Year'].astype(int)
        
        return movies, similarity, duplicate_count
        
    except FileNotFoundError:
        st.error("❌ Model files not found! Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

def recommend(movies, similarity, movie_name, num_recommendations=10, 
              filter_genre=None, min_rating=0, year_range=None):
    """
    Get movie recommendations with optional filters
    """
    # Get movie index
    try:
        movie_index = movies[movies["Title"] == movie_name].index[0]
    except IndexError:
        st.warning("Movie not found in database!")
        return []
    
    # Check if index is within similarity matrix bounds
    if movie_index >= similarity.shape[0]:
        st.error(f"Movie index {movie_index} is out of bounds")
        return []
    
    # Get similarity scores
    distances = similarity[movie_index]
    
    # Get movie list sorted by similarity
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:50]
    
    recommended_movies = []
    seen_titles = set()
    
    for idx, (index, similarity_score) in enumerate(movie_list):
        if len(recommended_movies) >= num_recommendations:
            break
        
        if index >= len(movies):
            continue
            
        movie = movies.iloc[index]
        
        # Skip if already seen
        if movie["Title"] in seen_titles:
            continue
        seen_titles.add(movie["Title"])
        
        # Apply genre filter
        if filter_genre and filter_genre != "All Genres":
            if filter_genre not in str(movie.get("Genre", "")):
                continue
                
        # Apply rating filter
        if min_rating > 0:
            if movie.get("Vote_Average", 0) < min_rating:
                continue
        
        # Apply year range filter
        if year_range:
            movie_year = movie.get("Year", 0)
            if movie_year < year_range[0] or movie_year > year_range[1]:
                continue
        
        recommended_movies.append({
            "Title": movie.get("Title", "Unknown"),
            "Genre": movie.get("Genre", "N/A"),
            "Rating": movie.get("Vote_Average", 0),
            "Popularity": movie.get("Popularity", 0),
            "Year": movie.get("Year", 0),
            "Poster": movie.get("Poster_Url", ""),
            "Similarity": round(similarity_score, 3)
        })
    
    return recommended_movies

def get_genre_list(movies):
    """Extract all unique genres from the dataset"""
    all_genres = set()
    for genre in movies['Genre'].dropna():
        if isinstance(genre, str):
            all_genres.update([g.strip() for g in genre.split(',')])
    return ["All Genres"] + sorted(list(all_genres))

def get_movie_stats(movies):
    """Get statistics about the movie dataset"""
    return {
        'total_movies': len(movies),
        'avg_rating': movies['Vote_Average'].mean(),
        'max_rating': movies['Vote_Average'].max(),
        'min_rating': movies['Vote_Average'].min(),
        'most_common_genre': movies['Genre'].mode()[0] if not movies['Genre'].mode().empty else 'N/A'
    }

def format_rating_stars(rating):
    """Convert rating to star representation"""
    full_stars = int(rating // 2)
    half_star = 1 if rating % 2 >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "⭐" * full_stars + "½" * half_star + "☆" * empty_stars