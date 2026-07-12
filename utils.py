# ==========================================
# utils.py
# Netflix Movie Recommendation System
# Utility Functions
# ==========================================

import pandas as pd
import numpy as np

# ------------------------------------------
# Normalize Column
# ------------------------------------------
def normalize(series):
    """
    Normalize values between 0 and 1

    Formula:
    (x-min)/(max-min)

    Used for:
    - Popularity
    - Ratings
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return series

    return (series - minimum) / (maximum - minimum)


# ------------------------------------------
# Add Hybrid Score
# ------------------------------------------
def add_hybrid_score(df):
    """
    Create normalized popularity and rating.

    Hybrid Score

    70% Rating
    30% Popularity

    Returns updated dataframe.
    """

    df = df.copy()

    df["Normalized_Rating"] = normalize(df["Vote_Average"])

    df["Normalized_Popularity"] = normalize(df["Popularity"])

    df["Hybrid_Score"] = (
        0.70 * df["Normalized_Rating"]
        +
        0.30 * df["Normalized_Popularity"]
    )

    return df


# ------------------------------------------
# Apply Filters
# ------------------------------------------
def apply_filters(
    df,
    genre=None,
    language=None,
    year=None,
    min_rating=0
):
    """
    Filter movies.

    Parameters
    ----------
    genre : string

    language : string

    year : int

    min_rating : float
    """

    filtered = df.copy()

    if genre and genre != "All":

        filtered = filtered[
            filtered["Genre"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

    if language and language != "All":

        filtered = filtered[
            filtered["Original_Language"] == language
        ]

    if year and year != "All":

        filtered = filtered[
            filtered["Year"] == year
        ]

    filtered = filtered[
        filtered["Vote_Average"] >= min_rating
    ]

    return filtered


# ------------------------------------------
# Trending Movies
# ------------------------------------------
def get_trending_movies(df, top_n=10):
    """
    Return most popular movies.
    """

    return (
        df
        .sort_values(
            by="Popularity",
            ascending=False
        )
        .head(top_n)
    )


# ------------------------------------------
# Top Rated Movies
# ------------------------------------------
def get_top_rated_movies(df, top_n=10):
    """
    Return highest rated movies.
    """

    return (
        df
        .sort_values(
            by="Vote_Average",
            ascending=False
        )
        .head(top_n)
    )


# ------------------------------------------
# Latest Movies
# ------------------------------------------
def get_latest_movies(df, top_n=10):
    """
    Return latest released movies.
    """

    return (
        df
        .sort_values(
            by="Year",
            ascending=False
        )
        .head(top_n)
    )


# ------------------------------------------
# Get Movie Details
# ------------------------------------------
def get_movie_details(df, movie_name):
    """
    Return movie details as dictionary.
    """

    movie = df[
        df["Title"] == movie_name
    ]

    if movie.empty:
        return None

    movie = movie.iloc[0]

    return {

        "Title": movie["Title"],

        "Genre": movie["Genre"],

        "Rating": movie["Vote_Average"],

        "Popularity": movie["Popularity"],

        "Language": movie["Original_Language"],

        "Year": movie["Year"],

        "Poster": movie["Poster_Url"]

    }


# ------------------------------------------
# Search Suggestions
# ------------------------------------------
def search_movies(df, keyword):
    """
    Return movie suggestions.

    Example

    Input:
    bat

    Output:
    Batman
    Batman Begins
    The Batman
    """

    keyword = keyword.lower()

    return (
        df[
            df["Title"]
            .str.lower()
            .str.contains(
                keyword,
                na=False
            )
        ]["Title"]
        .tolist()
    )


# ------------------------------------------
# Dataset Statistics
# ------------------------------------------
def dataset_statistics(df):
    """
    Return dataset statistics.
    """

    return {

        "Total Movies": len(df),

        "Languages":
        df["Original_Language"].nunique(),

        "Genres":
        df["Genre"].nunique(),

        "Average Rating":
        round(
            df["Vote_Average"].mean(),
            2
        ),

        "Average Popularity":
        round(
            df["Popularity"].mean(),
            2
        )

    }


# ------------------------------------------
# Genre List
# ------------------------------------------
def get_all_genres(df):
    """
    Return sorted unique genres.
    """

    genres = set()

    for row in df["Genre"].dropna():

        for genre in row.split(","):

            genres.add(
                genre.strip()
            )

    return sorted(genres)


# ------------------------------------------
# Languages List
# ------------------------------------------
def get_all_languages(df):
    """
    Return sorted languages.
    """

    return sorted(
        df["Original_Language"]
        .dropna()
        .unique()
    )


# ------------------------------------------
# Years List
# ------------------------------------------
def get_all_years(df):
    """
    Return sorted years.
    """

    return sorted(

        df["Year"]

        .dropna()

        .astype(int)

        .unique(),

        reverse=True

    )