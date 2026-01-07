"""
Generiert die Testdaten für den Unit-Test.
Ausführung: python generate_test_data.py

Beim Recommender System wird die gesamte User-Item-Matrix für Korrelationen benötigt.
Daher werden die Originaldaten (U.data, Movie_Id_Titles) als Trainingsdaten genutzt.
Als Testdaten wird eine CSV-Datei mit Testfällen erzeugt (Film + erwartetes Ergebnis).
"""
import pandas as pd

# Lade Originaldaten
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('U.data', sep='\t', names=column_names)
movie_titles = pd.read_csv('Movie_Id_Titles')
data = pd.merge(df, movie_titles, on='item_id')

# Testfälle: Bekannte Filme sollten sich selbst als Top-Empfehlung haben
test_cases = pd.DataFrame({
    'movie_name': ['Star Wars (1977)', 'Liar Liar (1997)', 'Fargo (1996)'],
    'expected_top_movie': ['Star Wars (1977)', 'Liar Liar (1997)', 'Fargo (1996)'],
    'expected_min_correlation': [1.0, 1.0, 1.0]
})
test_cases.to_csv('test_data.csv', index=False)

print(f"Testfälle gespeichert: {len(test_cases)} Testfälle -> test_data.csv")
