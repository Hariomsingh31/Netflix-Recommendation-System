# =============================================================================
# recommendation.py
# Netflix Movie Recommendation System Backend
# Part 1 - Imports, Data Loading & Fuzzy Search
# =============================================================================

import pickle
import pandas as pd
import numpy as np
import streamlit as st

from rapidfuzz import process, fuzz

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_resource
def load_data():
    """
    Load trained model files.

    Returns
    -------
    movies : DataFrame
    similarity : numpy.ndarray
    """

    try:

        movies = pickle.load(open("models/movies.pkl", "rb"))

        similarity = pickle.load(open("models/similarity.pkl", "rb"))

        return movies, similarity

    except FileNotFoundError as e:

        st.error(f"Model file not found:\n{e}")

        st.stop()

    except Exception as e:

        st.error(f"Error loading model:\n{e}")

        st.stop()

    # NOTE: st.stop() halts execution above, so this point is never reached.
    # Removed the unreachable `return movies, similarity` that referenced
    # undefined variables.


# =============================================================================
# BASIC INFORMATION
# =============================================================================

def total_movies(movies):
    """Return total number of movies."""
    return len(movies)


def total_languages(movies):
    """Return number of languages."""
    return movies["Original_Language"].nunique()


def average_rating(movies):
    """Return average movie rating."""
    return round(movies["Vote_Average"].mean(), 2)


def latest_year(movies):
    """Return latest movie year."""
    return int(movies["Year"].max())


# =============================================================================
# FUZZY SEARCH
# =============================================================================

def fuzzy_search(movie_name, movies):
    """
    Find closest matching movie title.

    Parameters
    ----------
    movie_name : str

    Returns
    -------
    Best matching movie title
    """

    titles = movies["Title"].tolist()

    result = process.extractOne(

        movie_name,

        titles,

        scorer=fuzz.WRatio

    )

    if result is None:

        return None

    matched_title, score, _ = result

    if score >= 70:

        return matched_title

    return None


# =============================================================================
# SEARCH SUGGESTIONS
# =============================================================================

def search_suggestions(movie_name, movies, limit=5):
    """
    Return top movie suggestions.

    Example

    Input:

    spder man

    Output:

    Spider-Man
    Spider-Man 2
    Spider-Man 3
    """

    titles = movies["Title"].tolist()

    results = process.extract(

        movie_name,

        titles,

        scorer=fuzz.WRatio,

        limit=limit

    )

    suggestions = []

    for movie in results:

        suggestions.append(movie[0])

    return suggestions


# =============================================================================
# GET MOVIE DETAILS
# =============================================================================

def get_movie(movie_name, movies):
    """
    Return complete movie row.

    Used by app.py
    """

    movie = movies[

        movies["Title"] == movie_name

    ]

    if movie.empty:

        return None

    return movie.iloc[0]


# =============================================================================
# CHECK MOVIE EXISTS
# =============================================================================

def movie_exists(movie_name, movies):
    """
    Check whether movie exists.
    """

    return movie_name in movies["Title"].values


# =============================================================================
# HYBRID RECOMMENDATION ENGINE
# =============================================================================

# -----------------------------------------------------------------------------
# Normalize Popularity
# -----------------------------------------------------------------------------

def normalize_popularity(popularity, max_popularity):
    """
    Normalize popularity between 0 and 1.
    """

    if max_popularity == 0:
        return 0

    return popularity / max_popularity


# -----------------------------------------------------------------------------
# Normalize Rating
# -----------------------------------------------------------------------------

def normalize_rating(rating):
    """
    Convert rating (0-10) into (0-1)
    """

    return rating / 10


# -----------------------------------------------------------------------------
# Hybrid Score
# -----------------------------------------------------------------------------

def calculate_hybrid_score(similarity_score,
                           rating,
                           popularity,
                           max_popularity):
    """
    Final Score

    70% -> Similarity
    20% -> Rating
    10% -> Popularity
    """

    rating_score = normalize_rating(rating)

    popularity_score = normalize_popularity(
        popularity,
        max_popularity
    )

    final_score = (

        0.70 * similarity_score +

        0.20 * rating_score +

        0.10 * popularity_score

    )

    return round(final_score, 4)


# =============================================================================
# RECOMMEND MOVIES
# =============================================================================

def recommend(
    movie_name,
    movies,
    similarity,
    top_n=10
):
    """
    Hybrid Movie Recommendation

    Returns

    List of recommended movies
    """

    # ----------------------------------------------------
    # Check movie exists
    # ----------------------------------------------------

    if not movie_exists(movie_name, movies):

        corrected = fuzzy_search(movie_name, movies)

        if corrected is None:

            return []

        movie_name = corrected

    # ----------------------------------------------------
    # Movie Index
    # ----------------------------------------------------

    movie_index = movies[
        movies["Title"] == movie_name
    ].index[0]

    # ----------------------------------------------------
    # Similarity Scores
    # ----------------------------------------------------

    distances = similarity[movie_index]

    # ----------------------------------------------------
    # Maximum Popularity
    # ----------------------------------------------------

    max_popularity = movies["Popularity"].max()

    recommendations = []

    # ----------------------------------------------------
    # Loop through similarity matrix
    # ----------------------------------------------------

    for index, sim_score in enumerate(distances):

        if index == movie_index:
            continue

        movie = movies.iloc[index]

        final_score = calculate_hybrid_score(

            similarity_score=sim_score,

            rating=movie["Vote_Average"],

            popularity=movie["Popularity"],

            max_popularity=max_popularity

        )

        recommendations.append({

            "Title": movie["Title"],

            "Genre": movie["Genre"],

            "Rating": movie["Vote_Average"],

            "Popularity": movie["Popularity"],

            "Year": movie["Year"],

            "Poster_Url": get_poster(movie),

            "Similarity": round(sim_score, 4),

            "HybridScore": final_score

        })

    # ----------------------------------------------------
    # Convert into DataFrame
    # ----------------------------------------------------

    recommendations = pd.DataFrame(recommendations)

    # ----------------------------------------------------
    # Remove duplicate titles
    # ----------------------------------------------------

    recommendations = recommendations.drop_duplicates(
        subset="Title"
    )

    # ----------------------------------------------------
    # Sort by Hybrid Score
    # ----------------------------------------------------

    recommendations = recommendations.sort_values(

        by="HybridScore",

        ascending=False

    )

    # ----------------------------------------------------
    # Return Top N
    # ----------------------------------------------------

    return recommendations.head(top_n)


# =============================================================================
# WHY THIS MOVIE?
# =============================================================================

def recommendation_reason(movie):
    """
    Explain recommendation.
    """

    reasons = []

    if movie["Rating"] >= 8:

        reasons.append("⭐ Highly Rated")

    if movie["Popularity"] >= 500:

        reasons.append("🔥 Popular")

    if movie["Similarity"] >= 0.60:

        reasons.append("🎯 Highly Similar")

    elif movie["Similarity"] >= 0.40:

        reasons.append("🎬 Similar Story")

    else:

        reasons.append("📖 Related Content")

    return ", ".join(reasons)


# =============================================================================
# DISPLAY RECOMMENDATIONS
# =============================================================================

def recommendation_table(
    movie_name,
    movies,
    similarity,
    top_n=10
):
    """
    Display recommendation table.
    """

    recommendations = recommend(

        movie_name,

        movies,

        similarity,

        top_n

    )

    if len(recommendations) == 0:

        return pd.DataFrame()

    recommendations["Reason"] = recommendations.apply(

        recommendation_reason,

        axis=1

    )

    return recommendations

# =============================================================================
# TRENDING MOVIES
# =============================================================================

def get_trending_movies(
    movies,
    limit=10
):
    """
    Return Top Trending Movies
    based on Popularity.
    """

    trending = movies.sort_values(

        by="Popularity",

        ascending=False

    )

    return trending.head(limit)


# =============================================================================
# TOP RATED MOVIES
# =============================================================================

def get_top_rated(
    movies,
    limit=10
):
    """
    Return Highest Rated Movies
    """

    top_rated = movies.sort_values(

        by=["Vote_Average", "Vote_Count"],

        ascending=False

    )

    return top_rated.head(limit)


# =============================================================================
# AVAILABLE LANGUAGES
# =============================================================================

def get_languages(movies):
    """
    Return available languages.
    """

    languages = movies["Original_Language"] \
                    .dropna() \
                    .unique() \
                    .tolist()

    languages.sort()

    languages.insert(0, "All")

    return languages


# =============================================================================
# AVAILABLE YEARS
# =============================================================================

def get_years(movies):
    """
    Return available release years.
    """

    years = movies["Year"] \
                .dropna() \
                .unique() \
                .tolist()

    years = sorted(years)

    return years


# =============================================================================
# AVAILABLE GENRES
# =============================================================================

def get_genres(movies):
    """
    Return all movie genres.
    """

    genres = set()

    for item in movies["Genre"].dropna():

        if isinstance(item, str):

            values = item.split(",")

            for genre in values:

                genres.add(
                    genre.strip()
                )

    genres = sorted(list(genres))

    genres.insert(0, "All")

    return genres


# =============================================================================
# FILTER MOVIES
# =============================================================================

def filter_movies(

    movies,

    genre="All",

    language="All",

    year=None,

    min_rating=0

):
    """
    Filter movies according
    to user selection.
    """

    df = movies.copy()

    # ------------------------------
    # Genre
    # ------------------------------

    if genre != "All":

        df = df[

            df["Genre"]

            .str.contains(

                genre,

                case=False,

                na=False,

                regex=False

            )

        ]

    # ------------------------------
    # Language
    # ------------------------------

    if language != "All":

        df = df[

            df["Original_Language"]

            == language

        ]

    # ------------------------------
    # Year
    # ------------------------------

    if year is not None:

        df = df[

            df["Year"] == year

        ]

    # ------------------------------
    # Rating
    # ------------------------------

    if min_rating > 0:

        df = df[

            df["Vote_Average"]

            >= min_rating

        ]

    return df


# =============================================================================
# SEARCH MOVIES
# =============================================================================

def search_movies(

    movies,

    keyword

):
    """
    Search movie titles.
    """

    keyword = keyword.lower()

    results = movies[

        movies["Title"]

        .str.lower()

        .str.contains(

            keyword,

            na=False,

            regex=False

        )

    ]

    return results


# =============================================================================
# MOVIE DETAILS
# =============================================================================

def movie_details(

    movies,

    movie_name

):
    """
    Return complete movie details.
    """

    movie = movies[

        movies["Title"]

        == movie_name

    ]

    if movie.empty:

        return None

    return movie.iloc[0]


# =============================================================================
# DATASET STATISTICS
# =============================================================================

def dataset_statistics(movies):
    """
    Dataset Summary
    """

    stats = {

        "Total Movies":

            len(movies),

        "Languages":

            movies["Original_Language"]

            .nunique(),

        "Genres":

            len(

                get_genres(movies)

            ) - 1,

        "Average Rating":

            round(

                movies["Vote_Average"]

                .mean(),

                2

            ),

        "Most Popular":

            movies.sort_values(

                "Popularity",

                ascending=False

            )

            .iloc[0]["Title"]

    }

    return stats

# =============================================================================
# PART 4 : STATISTICS, HELPER FUNCTIONS & FINAL CLEANUP
# =============================================================================

# NOTE: pandas/numpy already imported at the top of the file — no need to
# reimport them here.

# =============================================================================
# DATASET SUMMARY
# =============================================================================

def get_dataset_summary(movies):
    """
    Returns summary statistics of the dataset.
    """

    summary = {

        "Total Movies": len(movies),

        "Total Genres": len(get_genres(movies)) - 1,

        "Total Languages": movies["Original_Language"].nunique(),

        "Average Rating": round(
            movies["Vote_Average"].mean(), 2
        ),

        "Average Popularity": round(
            movies["Popularity"].mean(), 2
        ),

        "Latest Movie Year": int(
            movies["Year"].max()
        ),

        "Oldest Movie Year": int(
            movies["Year"].min()
        )

    }

    return summary


# =============================================================================
# RECOMMENDATION CONFIDENCE
# =============================================================================

def confidence_level(score):
    """
    Convert similarity score into confidence label.
    """

    if score >= 0.80:
        return "★★★★★ Excellent Match"

    elif score >= 0.60:
        return "★★★★☆ Very Good Match"

    elif score >= 0.40:
        return "★★★☆☆ Good Match"

    elif score >= 0.20:
        return "★★☆☆☆ Fair Match"

    else:
        return "★☆☆☆☆ Low Match"


# =============================================================================
# RATING BADGE
# =============================================================================

def rating_badge(rating):

    if rating >= 8:
        return "🟢 Excellent"

    elif rating >= 7:
        return "🟡 Good"

    elif rating >= 6:
        return "🟠 Average"

    return "🔴 Low"


# =============================================================================
# POPULARITY BADGE
# =============================================================================

def popularity_badge(popularity):

    if popularity >= 1000:
        return "🔥 Trending"

    elif popularity >= 500:
        return "📈 Popular"

    elif popularity >= 100:
        return "👍 Moderate"

    return "🌱 Hidden Gem"


# =============================================================================
# FORMAT RECOMMENDATION CARD
# =============================================================================

def build_movie_card(movie):
    """
    Convert one movie row into
    a dictionary suitable for Streamlit.
    """

    return {

        "Title": movie["Title"],

        "Genre": movie["Genre"],

        "Year": movie["Year"],

        "Rating": movie["Vote_Average"],

        "Popularity": movie["Popularity"],

        "Poster": get_poster(movie),

        "RatingBadge": rating_badge(
            movie["Vote_Average"]
        ),

        "PopularityBadge": popularity_badge(
            movie["Popularity"]
        )

    }


# =============================================================================
# EXPORT RECOMMENDATIONS
# =============================================================================

def export_recommendations(df, filename="recommendations.csv"):
    """
    Save recommendations as CSV.
    """

    df.to_csv(
        filename,
        index=False
    )

    return filename


# =============================================================================
# REMOVE DUPLICATES
# =============================================================================

def remove_duplicate_movies(df):

    return df.drop_duplicates(
        subset="Title"
    ).reset_index(drop=True)


# =============================================================================
# SAFE POSTER
# =============================================================================

def get_poster(movie):

    poster = movie.get("Poster_Url", "")

    if pd.isna(poster):
        return ""

    return poster


# =============================================================================
# GET MOVIE INDEX
# =============================================================================

def movie_index(movie_name, movies):

    try:

        return movies[
            movies["Title"] == movie_name
        ].index[0]

    except Exception:

        return None


# =============================================================================
# GET MOVIE BY INDEX
# =============================================================================

def movie_by_index(index, movies):

    if index >= len(movies):

        return None

    return movies.iloc[index]


# =============================================================================
# GET RANDOM MOVIES
# =============================================================================

def random_movies(
    movies,
    n=10
):
    # Guard against n exceeding the number of available rows, which would
    # otherwise raise a ValueError from .sample().
    n = min(n, len(movies))

    return movies.sample(n)


# =============================================================================
# MOST POPULAR MOVIE
# =============================================================================

def most_popular_movie(movies):

    return movies.sort_values(

        "Popularity",

        ascending=False

    ).iloc[0]


# =============================================================================
# HIGHEST RATED MOVIE
# =============================================================================

def highest_rated_movie(movies):

    return movies.sort_values(

        ["Vote_Average", "Vote_Count"],

        ascending=False

    ).iloc[0]


# =============================================================================
# MOVIES BY LANGUAGE
# =============================================================================

def movies_by_language(
    movies,
    language
):

    return movies[

        movies["Original_Language"]

        == language

    ]


# =============================================================================
# MOVIES BY GENRE
# =============================================================================

def movies_by_genre(
    movies,
    genre
):

    return movies[

        movies["Genre"]

        .str.contains(

            genre,

            case=False,

            na=False,

            regex=False

        )

    ]


# =============================================================================
# MOVIES BY YEAR
# =============================================================================

def movies_by_year(
    movies,
    year
):

    return movies[

        movies["Year"] == year

    ]


# =============================================================================
# FINAL CLEANUP
# =============================================================================

def clean_dataframe(df):
    """
    Standard dataframe cleanup.
    """

    df = df.copy()

    df = df.fillna("")

    df = df.drop_duplicates()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# =============================================================================
# END OF FILE
# =============================================================================



