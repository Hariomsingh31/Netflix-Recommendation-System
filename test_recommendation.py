import pickle
import pandas as pd
from recommendation import (
    recommend, fuzzy_search, get_genres,
    get_languages, dataset_statistics
)

# Load your actual model files
movies = pickle.load(open("models/movies.pkl", "rb"))
similarity = pickle.load(open("models/similarity.pkl", "rb"))

print("Total movies:", len(movies))
print("Genres:", get_genres(movies)[:5])
print("Languages:", get_languages(movies)[:5])

# Test fuzzy search
print("Fuzzy match 'spder man':", fuzzy_search("spder man", movies))

# Test recommendations
recs = recommend("Spider-Man", movies, similarity, top_n=5)
print(recs[["Title", "HybridScore"]])

print("Stats:", dataset_statistics(movies))