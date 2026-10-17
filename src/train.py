from pathlib import Path

import yaml
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION DU PROJET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="config/config.yaml"):
    """Charge la configuration YAML du projet."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Le fichier de configuration YAML est invalide.")

    return config


# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

def load_dataset():
    """Charge le dataset Breast Cancer Wisconsin."""

    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y


# ============================================================
# 3. VALIDATION DU DATASET
# ============================================================

def validate_dataset(X, y):
    """Vérifie la qualité et la cohérence des données."""

    if X.empty:
        raise ValueError(
            "Le jeu de données ne contient aucune observation."
        )

    if len(X) != len(y):
        raise ValueError(
            "Le nombre d'observations et de cibles diffère."
        )

    if X.isnull().any().any():
        raise ValueError(
            "Des valeurs manquantes ont été détectées dans X."
        )

    if y.isnull().any():
        raise ValueError(
            "Des valeurs manquantes ont été détectées dans y."
        )

    if y.nunique() < 2:
        raise ValueError(
            "La cible doit contenir au moins deux classes."
        )

    if not all(dtype.kind in "if" for dtype in X.dtypes):
        raise TypeError(
            "Toutes les variables doivent être numériques."
        )

    print("Validation du dataset réussie.")


# ============================================================
# 4. CONSTRUCTION DU PIPELINE DE MACHINE LEARNING
# ============================================================

def build_pipeline(config):
    """Construit le pipeline de classification."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config["model"]["C"],
                    max_iter=config["model"]["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


# ============================================================
# 5. SÉPARATION DES DONNÉES
# ============================================================

def split_data(X, y, config):
    """Sépare les données en ensembles train et test."""

    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=y,
    )


# ============================================================
# 6. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================

def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    config,
):
    """Entraîne le modèle et calcule les métriques."""

    pipeline = build_pipeline(config)

    # Entraîner le pipeline
    pipeline.fit(X_train, y_train)

    # Prédire les classes
    predictions = pipeline.predict(X_test)

    # Calculer les probabilités de la classe 1
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Calculer les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return pipeline, metrics


# ============================================================
# 7. ENTRAÎNEMENT ET SUIVI AVEC MLFLOW
# ============================================================

def run_training(config_path="config/config.yaml"):
    """Exécute le pipeline complet de Machine Learning."""

    # --------------------------------------------------------
    # Étape 1 : Charger la configuration
    # --------------------------------------------------------

    config = load_config(config_path)

    print("\nConfiguration chargée avec succès.")

    # --------------------------------------------------------
    # Étape 2 : Configurer MLflow
    # --------------------------------------------------------

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"Expérience MLflow : {experiment_name}")

    # --------------------------------------------------------
    # Étape 3 : Charger et valider les données
    # --------------------------------------------------------

    X, y = load_dataset()

    validate_dataset(X, y)

    print(f"Nombre d'observations : {len(X)}")
    print(f"Dimensions de X : {X.shape}")
    print(f"Classes présentes : {sorted(y.unique())}")

    # --------------------------------------------------------
    # Étape 4 : Séparer les données
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        config,
    )

    print(f"Taille de l'entraînement : {len(X_train)}")
    print(f"Taille du test : {len(X_test)}")

    # --------------------------------------------------------
    # Étape 5 : Démarrer une exécution MLflow
    # --------------------------------------------------------

    with mlflow.start_run() as active_run:

        # Enregistrer les paramètres
        mlflow.log_params(
            {
                "seed": config["seed"],
                "test_size": config["data"]["test_size"],
                "C": config["model"]["C"],
                "max_iter": config["model"]["max_iter"],
                "dataset": "Breast Cancer Wisconsin",
                "classifier": "LogisticRegression",
            }
        )

        # ----------------------------------------------------
        # Étape 6 : Entraîner et évaluer le modèle
        # ----------------------------------------------------

        pipeline, metrics = train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
        )

        # ----------------------------------------------------
        # Étape 7 : Enregistrer les métriques
        # ----------------------------------------------------

        mlflow.log_metrics(metrics)

        print("\nRésultats de l'évaluation :")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        # ----------------------------------------------------
        # Étape 8 : Enregistrer le modèle dans MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        # ----------------------------------------------------
        # Étape 9 : Afficher les informations du run
        # ----------------------------------------------------

        print("\nModèle enregistré avec succès.")
        print(f"Run ID : {active_run.info.run_id}")
        print(f"Experiment ID : {active_run.info.experiment_id}")


# ============================================================
# 8. POINT D'ENTRÉE DU PROGRAMME
# ============================================================

if __name__ == "__main__":
    run_training()