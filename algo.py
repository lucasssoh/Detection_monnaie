"""
algo.py
-------
Script principal : entraînement, réglage et évaluation du système.

Workflow (respecte strictement la règle du cours) :

    1. Charger le split train/val/test (déjà calculé par split_data.py)
    2. Pour chaque combinaison d'hyperparamètres :
         a. Prétraiter les images (train + val)
         b. Entraîner le modèle sur train
         c. Évaluer sur val
    3. Sélectionner les meilleurs hyperparamètres (argmin MAE sur val)
    4. Réentraîner avec les meilleurs hyperparamètres sur train
    5. Calculer les métriques finales UNE SEULE FOIS sur test

Règle absolue du cours :
    On ne touche au jeu de test qu'à l'étape 5.
    On ne refait pas l'étape 5 après avoir vu le résultat.
"""

import json
import os
import numpy as np

from split_data import charger_verite_terrain, charger_split, sauvegarder_split, split_stratifie
from preprocessing import preprocess_dataset
from regression import RegressionLineaire, RegressionPolynomiale, mae, mse, afficher_metriques


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

IMAGE_DIR      = "base_images"
ANNOTATION_DIR = "base_annotations"
SPLIT_FILE     = "split.json"

# Grille d'hyperparamètres à explorer sur la base de validation
# (pas sur le test !)
GRILLE_HP = {
    "sigma"  : [1, 2, 3],          # Lissage gaussien
    "noyau"  : [3, 5, 7],          # Opening morphologique
    "degre"  : [1, 2],             # Degré de la régression polynomiale
}


# ──────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────

def construire_chemin_image(nom: str, image_dir: str) -> str:
    """
    Cherche le fichier image correspondant à 'nom' dans image_dir.
    Essaie les extensions courantes.
    """
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        path = os.path.join(image_dir, nom + ext)
        if os.path.exists(path):
            return path
    # Cas où le nom contient déjà l'extension
    path = os.path.join(image_dir, nom)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Image introuvable pour : {nom}")


def preparer_set(noms: list, vt: dict, image_dir: str,
                 sigma: float, noyau: int) -> tuple:
    """
    Pour une liste de noms d'images, retourne (X, y) :
        X : matrice (N, 5) de features
        y : vecteur (N,) de vérités terrain
    """
    paths = [construire_chemin_image(n, image_dir) for n in noms]
    y = np.array([vt[n] for n in noms], dtype=np.float64)
    X = preprocess_dataset(paths, sigma=sigma, noyau=noyau)
    return X, y


# ──────────────────────────────────────────────────────────────
# ÉTAPE 1 : SPLIT
# ──────────────────────────────────────────────────────────────

def etape_split(annotation_dir: str, split_file: str):
    """Crée ou charge le split train/val/test."""
    if os.path.exists(split_file):
        print(f"[INFO] Split existant chargé depuis {split_file}")
        return charger_split(split_file)
    else:
        print("[INFO] Création du split...")
        vt = charger_verite_terrain(annotation_dir)
        train, val, test = split_stratifie(vt)
        sauvegarder_split(train, val, test, split_file)
        return {"train": train, "val": val, "test": test}


# ──────────────────────────────────────────────────────────────
# ÉTAPE 2 : RÉGLAGE DES HYPERPARAMÈTRES (sur val uniquement)
# ──────────────────────────────────────────────────────────────

def etape_validation(split: dict, vt: dict, image_dir: str, grille: dict):
    """
    Parcourt la grille d'hyperparamètres et évalue chaque combinaison sur val.

    Retourne :
        meilleurs_hp : dict des hyperparamètres avec la meilleure MAE sur val
        resultats    : liste complète des résultats (pour analyse)
    """
    print("\n=== Réglage des hyperparamètres (base de validation) ===")
    meilleure_mae_val = float("inf")
    meilleurs_hp = {}
    resultats = []

    for sigma in grille["sigma"]:
        for noyau in grille["noyau"]:
            print(f"\n  Prétraitement : sigma={sigma}, noyau={noyau}")

            # Prétraitement du train et val avec ces hyperparamètres
            X_train, y_train = preparer_set(
                split["train"], vt, image_dir, sigma, noyau)
            X_val, y_val = preparer_set(
                split["val"], vt, image_dir, sigma, noyau)

            for degre in grille["degre"]:
                # Entraînement sur train
                if degre == 1:
                    modele = RegressionLineaire()
                else:
                    modele = RegressionPolynomiale(degre=degre)

                modele.fit(X_train, y_train)

                # Évaluation sur TRAIN (pour détecter l'overfitting)
                score_train = modele.score(X_train, y_train)
                # Évaluation sur VALIDATION
                score_val = modele.score(X_val, y_val)

                hp_label = f"sigma={sigma}, noyau={noyau}, degre={degre}"
                afficher_metriques(y_train, modele.predict(X_train),
                                   f"    Train (deg={degre})")
                afficher_metriques(y_val,   modele.predict(X_val),
                                   f"    Val   (deg={degre})")

                resultats.append({
                    "sigma": sigma, "noyau": noyau, "degre": degre,
                    "mae_train": score_train["MAE"],
                    "mae_val":   score_val["MAE"],
                })

                # On sélectionne sur la MAE de validation
                if score_val["MAE"] < meilleure_mae_val:
                    meilleure_mae_val = score_val["MAE"]
                    meilleurs_hp = {"sigma": sigma, "noyau": noyau, "degre": degre}

    print(f"\n→ Meilleurs hyperparamètres : {meilleurs_hp}")
    print(f"  MAE validation associée   : {meilleure_mae_val:.3f}")
    return meilleurs_hp, resultats


# ──────────────────────────────────────────────────────────────
# ÉTAPE 3 : ÉVALUATION FINALE (une seule fois sur test)
# ──────────────────────────────────────────────────────────────

def etape_test(split: dict, vt: dict, image_dir: str, meilleurs_hp: dict):
    """
    Réentraîne avec les meilleurs hyperparamètres sur train,
    puis évalue UNE SEULE FOIS sur test.

    ⚠ Cette fonction ne doit être appelée qu'une seule fois
      après avoir fixé définitivement les hyperparamètres.
    """
    print("\n=== Évaluation finale (base de test) ===")
    sigma = meilleurs_hp["sigma"]
    noyau = meilleurs_hp["noyau"]
    degre = meilleurs_hp["degre"]

    X_train, y_train = preparer_set(
        split["train"], vt, image_dir, sigma, noyau)
    X_test, y_test = preparer_set(
        split["test"],  vt, image_dir, sigma, noyau)

    if degre == 1:
        modele = RegressionLineaire()
    else:
        modele = RegressionPolynomiale(degre=degre)

    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)

    print(f"\nHyperparamètres finaux : {meilleurs_hp}")
    afficher_metriques(y_test, y_pred, "Test final")

    # Affichage image par image pour l'analyse d'erreurs
    print("\nDétail par image (test) :")
    print(f"  {'Image':<40} {'Vrai':>6} {'Prédit':>8} {'Erreur':>8}")
    print("  " + "-" * 64)
    for nom, vrai, pred in zip(split["test"], y_test.astype(int), y_pred):
        err = pred - vrai
        print(f"  {nom:<40} {vrai:>6} {pred:>8} {err:>+8}")

    return modele, y_pred


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Split
    split = etape_split(ANNOTATION_DIR, SPLIT_FILE)
    vt = charger_verite_terrain(ANNOTATION_DIR)

    # 2. Réglage sur validation
    meilleurs_hp, resultats_val = etape_validation(
        split, vt, IMAGE_DIR, GRILLE_HP)

    # 3. Évaluation finale sur test (UNE SEULE FOIS)
    modele_final, predictions = etape_test(split, vt, IMAGE_DIR, meilleurs_hp)

    print("\n=== Terminé ===")
