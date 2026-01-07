import unittest
import time
import pandas as pd
from model_logic import load_data, fit_model, predict_model

class TestRecommenderModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten und Testfälle."""
        cls.df = load_data('U.data', 'Movie_Id_Titles')
        cls.moviemat = fit_model(cls.df)

        cls.ratings_summary = pd.DataFrame(cls.df.groupby('title')['rating'].mean())
        cls.ratings_summary['num of ratings'] = pd.DataFrame(cls.df.groupby('title')['rating'].count())

        # Testfälle aus separatem Testdatenfile laden
        cls.test_cases = pd.read_csv('test_data.csv')
        cls.norm_fit_time = 0.5

    def test_1_predict_quality(self):
        """
        Testfall 1: Test der Vorhersagequalität (predict).
        Indikator: Top-Empfehlung und Korrelation, geprüft gegen Testfälle aus test_data.csv.
        """
        for _, row in self.test_cases.iterrows():
            movie = row['movie_name']
            expected_top = row['expected_top_movie']

            recs = predict_model(self.moviemat, movie, self.ratings_summary)
            top_movie = recs.index[0]
            top_corr = recs.iloc[0]['Correlation']

            print(f"\n[Test predict()] Film: {movie} -> Top Empfehlung: {top_movie} (Corr: {top_corr:.4f})")
            self.assertEqual(top_movie, expected_top)
            self.assertAlmostEqual(top_corr, 1.0)

    def test_2_fit_runtime(self):
        """
        Testfall 2: Überprüfung der Laufzeit der Trainingsfunktion (fit / Matrix-Erstellung).
        Ziel: Laufzeit < 120% der repräsentativen Normzeit.
        """
        start_time = time.time()
        fit_model(self.df)
        duration = time.time() - start_time

        limit = self.norm_fit_time * 1.2 # 120% der Normzeit
        print(f"\n[Test fit()] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")

        self.assertLess(duration, limit, f"Operation dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
