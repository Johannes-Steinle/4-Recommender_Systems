import pandas as pd
import numpy as np
import logging
import time
import os

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_data(ratings_file, titles_file):
    """Lädt die Filmdaten und führt ein Merge durch."""
    logger = logging.getLogger()
    try:
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(ratings_file, sep='\t', names=column_names)
        movie_titles = pd.read_csv(titles_file)
        
        data = pd.merge(df, movie_titles, on='item_id')
        logger.info(f"Daten erfolgreich geladen und gemergt.")
        return data
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {e}")
        raise

def get_movie_matrix(df):
    """Erstellt die User-Item-Matrix."""
    logger = logging.getLogger()
    start_time = time.time()
    
    logger.info("Erstelle Movie-Matrix (Pivot-Table)...")
    moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Matrix erstellt in {duration:.4f} Sekunden.")
    
    return moviemat, duration

def get_recommendations(moviemat, movie_name, df_ratings_summary):
    """Gibt ähnliche Filme basierend auf Korrelation zurück."""
    logger = logging.getLogger()
    logger.info(f"Suche Empfehlungen für: {movie_name}")
    
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
    moviemat, duration = get_movie_matrix(df)
    
    # Summary erstellen für Filterung
    ratings_summary = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings_summary['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
    
    movie = 'Star Wars (1977)'
    recs = get_recommendations(moviemat, movie, ratings_summary)
    print(f"Top 5 Empfehlungen für {movie}:")
    print(recs.head())
    logging.info(f"Testlauf erfolgreich beendet.")
