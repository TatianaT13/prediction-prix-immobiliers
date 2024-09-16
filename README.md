# Prédiction de Prix Immobiliers

Ce projet vise à prédire les prix des maisons en utilisant un modèle de régression linéaire. Il utilise un ensemble de données provenant de [Kaggle - Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Table des Matières

- [Installation](#installation)
- [Données](#données)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Utilisation](#utilisation)
- [Licence](#licence)

## Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/prediction-prix-immobiliers.git
    ```
2. Naviguez dans le dossier du projet :
    ```bash
    cd prediction-prix-immobiliers
    ```
3. Créez un environnement virtuel et activez-le :
    ```bash
    python -m venv env
    source env/bin/activate  # Sur Windows : env\Scripts\activate
    ```
4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Données

Les données utilisées dans ce projet proviennent de [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Placez le fichier `housing.csv` dans le dossier `data/`.

## Méthodologie

1. **Exploration des Données** : Analyse des caractéristiques et visualisation.
2. **Prétraitement** : Nettoyage des données, gestion des valeurs manquantes, encodage des variables catégorielles.
3. **Modélisation** : Entraînement d'un modèle de régression linéaire.
4. **Évaluation** : Évaluation des performances du modèle avec des métriques appropriées.

## Résultats

Le modèle de régression linéaire a atteint un score R² de **0.85**, indiquant une bonne capacité de prédiction sur les données test.

## Utilisation

Pour exécuter le notebook Jupyter :
```bash
jupyter notebook notebooks/Prédiction_Prix_Immobiliers.ipynb