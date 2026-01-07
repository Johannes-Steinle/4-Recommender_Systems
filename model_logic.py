import pandas as pd
import numpy as np
import logging
import time
import os
from functools import wraps

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def my_logger(orig_func):
    """Loggt den Funktionsnamen und die übergebenen Argumente."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran with args: {args}, and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """Loggt die Ausführungszeit der Funktion."""
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(f'{orig_func.__name__} ran in: {t2:.4f} sec')
        return result
    return wrapper

@my_logger
@my_timer
def load_data(ratings_file, titles_file):
    """Lädt die Filmdaten und führt ein Merge durch."""
    try:
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(ratings_file, sep='\t', names=column_names)
        movie_titles = pd.read_csv(titles_file)
        data = pd.merge(df, movie_titles, on='item_id')
        return data
    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten: {e}")
        raise

@my_logger
@my_timer
def fit_model(df):
    """Erstellt die User-Item-Matrix."""
    moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
    return moviemat

@my_logger
@my_timer
def predict_model(moviemat, movie_name, df_ratings_summary):
    """Gibt ähnliche Filme per Korrelation zurück."""
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    # Filtern nach Anzahl der Bewertungen (z.B. > 100)
    corr_movie = corr_movie.join(df_ratings_summary['num of ratings'])
    recommendations = corr_movie[corr_movie['num of ratings'] > 100].sort_values('Correlation', ascending=False)
    
    return recommendations

if __name__ == "__main__":
    df = load_data('U.data', 'Movie_Id_Titles')
    moviemat = fit_model(df)
    
    # Summary erstellen für Filterung
    ratings_summary = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings_summary['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    
    movie = 'Star Wars (1977)'
    recs = predict_model(moviemat, movie, ratings_summary)
    print(f"Top 5 Empfehlungen für {movie}:")
    print(recs.head())
