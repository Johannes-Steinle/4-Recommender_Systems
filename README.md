# Recommender Systems Projekt

Meine Umsetzung der Recommender Systems Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/4-Recommender_Systems/main?filepath=Recommender_Systems_Solution.ipynb)

## Überblick
Ein einfaches Film-Empfehlungssystem mit Collaborative Filtering. Anhand der Korrelation zwischen Nutzerbewertungen werden für einen gegebenen Film ähnliche Filme empfohlen.

## Inhalt
* `Recommender_Systems_Solution.ipynb` - Haupt-Notebook mit der Empfehlungslogik
* `Movie_Id_Titles` - Film-ID zu Titel Mapping
* `U.data` - MovieLens Bewertungsdaten (100.000 Bewertungen)

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `Recommender_Systems_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Erstellung einer User-Item-Bewertungsmatrix
   - Berechnung der Korrelation zwischen Filmen
   - Empfehlungen für Star Wars (1977) und ähnliche Filme
   - Top-Empfehlungen mit Korrelationswerten (gefiltert nach Mindestanzahl Bewertungen)

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | Collaborative Filtering Logik mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Empfehlungsqualität) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testfälle |
| `test_data.csv` | Testfälle (Filmnamen mit erwarteten Empfehlungen und Korrelationen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** Die Testfälle aus `test_data.csv` werden geladen und für jeden Film geprüft, ob er sich selbst als Top-Empfehlung hat (Korrelation = 1.0).

**Testfall 2 - fit():** Die Laufzeit der Matrix-Erstellung wird gemessen und geprüft, ob sie unter 120% der Normzeit (0.5s) bleibt.

### Testergebnisse
```text
[Test predict()] Film: Star Wars (1977) -> Top Empfehlung: Star Wars (1977) (Corr: 1.0000)
[Test predict()] Film: Liar Liar (1997) -> Top Empfehlung: Liar Liar (1997) (Corr: 1.0000)
[Test predict()] Film: Fargo (1996) -> Top Empfehlung: Fargo (1996) (Corr: 1.0000)
.
[Test fit()] Gemessene Dauer: 0.0291s (Limit: 0.6000s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.405s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Testfälle aus `test_data.csv` und die Bewertungsdaten aus `U.data` und `Movie_Id_Titles`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testfälle neu zu generieren: `python generate_test_data.py`
