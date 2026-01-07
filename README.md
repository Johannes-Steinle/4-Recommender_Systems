# Recommender Systems Projekt

Dieses Repository enthält ein Recommender Systems Projekt als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist der Aufbau eines einfachen Film-Empfehlungssystems.

## Inhalt
* `Recommender_Systems_Solution.ipynb`: Das Haupt-Notebook mit der Empfehlungslogik.
* `Movie_Id_Titles`, `U.data`, `U.item`: Die Datensatz-Dateien.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (Collaborative Filtering) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Empfehlungsqualität und der Verarbeitungszeit durch.
- `training.log`: Protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test predict()] Top Empfehlung: Star Wars (1977) (Corr: 1.0000)
.
[Test fit()] Gemessene Dauer: 0.0296s (Limit: 0.7500s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.212s

OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/4-Recommender_Systems/main?filepath=Recommender_Systems_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/4-Recommender_Systems/main?filepath=Recommender_Systems_Solution.ipynb)
