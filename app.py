# ============================================================================
# FIXED APP.PY - Handles Similarity Matrix Mismatch
# ============================================================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="🎬 Netflix Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------
# Custom CSS for Better Styling
# ------------------------------------------

def load_css():
    """Load external CSS file"""
    with open("style.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()


# ------------------------------------------
# Load Saved Files with Error Handling
# ------------------------------------------

@st.cache_resource
def load_data():
    """Load movie data and similarity matrix with caching"""
    try:
        movies = pickle.load(open("models/movies.pkl", "rb"))
        similarity = pickle.load(open("models/similarity.pkl", "rb"))
        
        # Check if there are duplicates
        duplicate_count = movies.duplicated(subset=['Title']).sum()
        st.sidebar.info(f"📊 Found {duplicate_count} duplicate rows")
        
        # OPTION 1: Remove duplicates and keep first occurrence
        movies = movies.drop_duplicates(subset=['Title'], keep='first')
        movies = movies.reset_index(drop=True)
        
        # OPTION 2: If similarity matrix is smaller than movies, truncate movies
        if len(movies) > similarity.shape[0]:
            st.sidebar.warning(f"⚠️ Truncating movies from {len(movies)} to {similarity.shape[0]} to match similarity matrix")
            movies = movies.iloc[:similarity.shape[0]]
            movies = movies.reset_index(drop=True)
        
        # OPTION 3: If similarity matrix is larger, truncate similarity
        elif len(movies) < similarity.shape[0]:
            st.sidebar.warning(f"⚠️ Truncating similarity matrix from {similarity.shape[0]} to {len(movies)}")
            similarity = similarity[:len(movies), :len(movies)]
        
        # Ensure all required columns exist
        required_cols = ['Title', 'Genre', 'Vote_Average', 'Popularity', 'Year', 'Poster_Url']
        for col in required_cols:
            if col not in movies.columns:
                movies[col] = 'N/A' if col == 'Poster_Url' else 0
        
        # Convert Year to int if needed
        if movies['Year'].dtype != 'int64':
            movies['Year'] = movies['Year'].astype(int)
        
        return movies, similarity
        
    except FileNotFoundError:
        st.error("❌ Model files not found! Please check the path.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()

movies, similarity = load_data()

# ------------------------------------------
# Recommendation Function with Error Handling
# ------------------------------------------

def recommend(movie_name, num_recommendations=10, filter_genre=None, min_rating=0):
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
        st.error(f"Movie index {movie_index} is out of bounds for similarity matrix of size {similarity.shape[0]}")
        return []
    
    # Get similarity scores
    distances = similarity[movie_index]
    
    # Get movie list sorted by similarity
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:50]  # Get more than needed for filtering
    
    recommended_movies = []
    seen_titles = set()
    
    for idx, (index, similarity_score) in enumerate(movie_list):
        if len(recommended_movies) >= num_recommendations:
            break
        
        # Check if index is within movies dataframe
        if index >= len(movies):
            continue
            
        movie = movies.iloc[index]
        
        # Skip if already seen
        if movie["Title"] in seen_titles:
            continue
        seen_titles.add(movie["Title"])
        
        # Apply filters
        if filter_genre and filter_genre != "All Genres":
            if filter_genre not in str(movie.get("Genre", "")):
                continue
                
        if min_rating > 0:
            if movie.get("Vote_Average", 0) < min_rating:
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

# ------------------------------------------
# Sidebar - Filters & Info
# ------------------------------------------

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", 
             use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Filter Options")
    
    # Genre filter
    all_genres = set()
    for genre in movies['Genre'].dropna():
        if isinstance(genre, str):
            all_genres.update([g.strip() for g in genre.split(',')])
    genre_options = ["All Genres"] + sorted(list(all_genres))
    
    filter_genre = st.selectbox("🎭 Filter by Genre", genre_options)
    
    # Rating filter
    min_rating = st.slider("⭐ Minimum Rating", 0.0, 10.0, 0.0, 0.5)
    
    # Number of recommendations
    num_recs = st.slider("📊 Number of Recommendations", 5, 20, 10, 5)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Database Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Movies", f"{len(movies):,}")
    with col2:
        st.metric("Avg Rating", f"{movies['Vote_Average'].mean():.2f}")
    
    # Similarity matrix info
    st.info(f"📐 Similarity Matrix: {similarity.shape[0]}x{similarity.shape[1]}")
    
    # Top movies by rating
    st.subheader("🏆 Top Rated Movies")
    top_movies = movies.nlargest(5, 'Vote_Average')[['Title', 'Vote_Average']]
    for _, row in top_movies.iterrows():
        st.write(f"⭐ {row['Title']}: {row['Vote_Average']:.1f}")

# ------------------------------------------
# Main Content
# ------------------------------------------

st.markdown('<h1 class="main-header">🎬 Netflix Movie Recommendation System</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #aaa; margin-bottom: 2rem;'>
    Discover movies similar to your favorites using <strong>Machine Learning</strong> 
    and <strong>Natural Language Processing</strong>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ------------------------------------------
# Movie Selection with Search
# ------------------------------------------

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    movie_titles = movies["Title"].values
    selected_movie = st.selectbox(
        "🎥 Choose a Movie You Like",
        movie_titles,
        help="Start typing to search for a movie"
    )

with col2:
    show_random = st.button("🎲 Random Movie")

with col3:
    show_filters = st.button("📌 Show Selected Movie")

# Random movie selection
if show_random:
    selected_movie = np.random.choice(movie_titles)
    st.success(f"🎲 Selected: {selected_movie}")

# ------------------------------------------
# Display Selected Movie Info
# ------------------------------------------

if show_filters or selected_movie:
    try:
        movie_data = movies[movies["Title"] == selected_movie].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if movie_data.get("Poster_Url") and movie_data["Poster_Url"] != "N/A":
                try:
                    st.image(movie_data["Poster_Url"], use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300x450/1a1a1a/ffffff?text=No+Poster", 
                            use_container_width=True)
        
        with col2:
            st.markdown(f"## {movie_data['Title']}")
            st.markdown(f"**📅 Year:** {int(movie_data['Year'])}")
            st.markdown(f"**🎭 Genres:** {movie_data['Genre']}")
            
            # Rating stars
            rating = movie_data['Vote_Average']
            stars = "⭐" * int(rating // 2) + "☆" * (5 - int(rating // 2))
            st.markdown(f"**Rating:** {stars} ({rating:.1f}/10)")
            
            st.markdown(f"**🔥 Popularity:** {movie_data['Popularity']:.0f}")
            st.markdown(f"**📊 Votes:** {movie_data.get('Vote_Count', 'N/A')}")
    except:
        st.error("Error loading movie data")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ------------------------------------------
# Recommendation Button
# ------------------------------------------

if st.button("🔍 Find Similar Movies", use_container_width=True):
    
    with st.spinner("🎯 Finding the best recommendations for you..."):
        time.sleep(0.5)  # Small delay for UX
        
        # Get recommendations with filters
        recommendations = recommend(
            selected_movie, 
            num_recommendations=num_recs,
            filter_genre=filter_genre if filter_genre != "All Genres" else None,
            min_rating=min_rating
        )
    
    if not recommendations:
        st.warning("No recommendations found with the current filters. Try adjusting your filters!")
    else:
        st.success(f"🎉 Found {len(recommendations)} recommendations for '{selected_movie}'!")
        
        # Display recommendations in a grid
        st.subheader("🎯 Recommended Movies")
        
        # Create rows of 4 movies
        cols = st.columns(4)
        
        for i, movie in enumerate(recommendations):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="text-align: center;">
                        <img src="{movie['Poster'] if movie['Poster'] else 'https://via.placeholder.com/300x450/1a1a1a/ffffff?text=No+Poster'}" 
                             style="width: 100%; border-radius: 10px; margin-bottom: 0.5rem;">
                        <h4 style="color: #fff; margin: 0.3rem 0;">{movie['Title']}</h4>
                        <div style="font-size: 0.9rem; color: #aaa;">
                            {movie['Genre']}
                        </div>
                        <div style="margin: 0.3rem 0;">
                            <span class="rating-stars">{"⭐" * min(5, int(movie['Rating'] // 2))}</span>
                            <span style="color: #FFD700;">{movie['Rating']:.1f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #888;">
                            <span>📅 {int(movie['Year'])}</span>
                            <span>🔥 {movie['Popularity']:.0f}</span>
                        </div>
                        <div style="margin-top: 0.3rem;">
                            <span class="similarity-badge">🎯 {movie['Similarity']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a button to get recommendations for this movie
                if st.button(f"🎬 More like this", key=f"rec_{i}_{movie['Title']}"):
                    selected_movie = movie['Title']
                    st.rerun()

# ------------------------------------------
# Export Functionality
# ------------------------------------------

if 'recommendations' in locals() and recommendations:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📥 Export Recommendations"):
            df_recs = pd.DataFrame(recommendations)
            csv = df_recs.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"recommendations_{selected_movie.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    with col2:
        if st.button("📋 Copy to Clipboard"):
            movie_titles = "\n".join([f"{i+1}. {m['Title']} ({m['Rating']:.1f}⭐)" 
                                     for i, m in enumerate(recommendations)])
            st.code(movie_titles, language=None)

# ------------------------------------------
# Footer
# ------------------------------------------

st.markdown("""
<hr class="divider">
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
    <p style="font-size: 0.8rem;">Data Source: Netflix Movies Dataset</p>
</div>
""", unsafe_allow_html=True)