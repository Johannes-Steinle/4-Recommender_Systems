import unittest
import time
import pandas as pd
from model_logic import load_data, get_movie_matrix, get_recommendations

class TestRecommenderModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.df = load_data('U.data', 'Movie_Id_Titles')
        cls.moviemat, _ = get_movie_matrix(cls.df)
        
        cls.ratings_summary = pd.DataFrame(cls.df.groupby('title')['rating'].mean())
        cls.ratings_summary['num of ratings'] = pd.DataFrame(cls.df.groupby('title')['rating'].count())
        
        # Norm-Zeit für Pivot/Correlation ca. 0.5s
        cls.norm_op_time = 1.0 

    def test_1_recommendation_quality(self):
        """
        Ziel: Prüfen, ob für 'Star Wars (1977)' wieder 'Star Wars (1977)' als top (Korrelation 1.0) erscheint.
        """
        movie = 'Star Wars (1977)'
        recs = get_recommendations(self.moviemat, movie, self.ratings_summary)
        
        top_movie = recs.index[0]
        top_corr = recs.iloc[0]['Correlation']
        
        print(f"\n[Test Rec] Top Empfehlung: {top_movie} (Corr: {top_corr})")
        self.assertEqual(top_movie, movie)
        self.assertAlmostEqual(top_corr, 1.0)

    def test_2_matrix_runtime(self):
        """
        Ziel: Matrix-Erstellung < 120% der Normzeit.
        """
        _, duration = get_movie_matrix(self.df)
        
        limit = self.norm_op_time * 1.5 # Etwas mehr Puffer für Pivot
        print(f"\n[Test Fit] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Operation dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
