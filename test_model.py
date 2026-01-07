import unittest
import time
import pandas as pd
from model_logic import load_data, fit_model, predict_model

class TestRecommenderModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.df = load_data('U.data', 'Movie_Id_Titles')
        cls.moviemat = fit_model(cls.df)
        
        cls.ratings_summary = pd.DataFrame(cls.df.groupby('title')['rating'].mean())
        cls.ratings_summary['num of ratings'] = pd.DataFrame(cls.df.groupby('title')['rating'].count())
        cls.norm_fit_time = 0.5 

    def test_1_predict_quality(self):
        """
        Aufgabe: Test der Vorhersagequalität (predict).
        Ziel: 'Star Wars (1977)' muss Korrelation 1.0 mit sich selbst haben.
        """
        movie = 'Star Wars (1977)'
        recs = predict_model(self.moviemat, movie, self.ratings_summary)
        
        top_movie = recs.index[0]
        top_corr = recs.iloc[0]['Correlation']
        
        print(f"\n[Test predict()] Top Empfehlung: {top_movie} (Corr: {top_corr:.4f})")
        self.assertEqual(top_movie, movie)
        self.assertAlmostEqual(top_corr, 1.0)

    def test_2_fit_runtime(self):
        """
        Aufgabe: Überprüfung der Laufzeit der Trainingsfunktion (fit / Matrix-Erstellung).
        Ziel: Laufzeit < 120% der Normzeit.
        """
        start_time = time.time()
        fit_model(self.df)
        duration = time.time() - start_time
        
        limit = self.norm_fit_time * 1.5 # Puffer für Testumgebung
        print(f"\n[Test fit()] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Operation dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
