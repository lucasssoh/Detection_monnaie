"""
regression.py
-------------
Modèle de régression pour prédire le nombre de pièces.

Pourquoi la régression et pas un simple seuil sur nb_objets_valides ?
→ La feature nb_objets_valides est une bonne approximation,
  mais elle se trompe quand des pièces se touchent (fusion de blobs)
  ou quand le fond crée des artefacts.
  La régression combine toutes les features pour corriger ces cas.

Modèles disponibles 
    - RegressionLineaire    : y = w·x + b   
    - RegressionPolynomiale : ajoute des termes x², x³… (capture les non-linéarités)

La sélection du meilleur modèle et de ses hyperparamètres se fait
sur la base de validation (pas le test !).
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────

def normaliser(
    X_train: np.ndarray, 
    X: np.ndarray):
    """
    Normalisation min-max calculée sur le train, appliquée partout.

    Pourquoi normaliser ?
    Les features ont des échelles très différentes :
        nb_pixels_utiles ≈ 100 000, circularité ≈ 0.8
    Sans normalisation, les features à grande échelle dominent
    le calcul des moindres carrés.

    IMPORTANT : on calcule min/max sur X_train uniquement,
    puis on applique la même transformation à val et test.
    (Sinon on "fuite" de l'information du test vers le train.)

    Retourne :
        X_norm : X normalisé
        min_   : vecteur des minima (à stocker pour le déploiement)
        range_ : vecteur des étendues (à stocker pour le déploiement)
    """

    min_ = X_train.min(axis=0)
    range_ = X_train.max(axis=0) - X_train.min(axis=0)
    # Évite la division par zéro si une feature est constante
    range_[range_ == 0] = 1.0
    X_norm = (X - min_) / range_
    return X_norm, min_, range_


def ajouter_biais(X: np.ndarray) -> np.ndarray:
    """Ajoute une colonne de 1 pour le terme de biais (intercept)."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def etendre_polynomial(
    X: np.ndarray, 
    degre: int) -> np.ndarray:
    """
    Ajoute des termes polynomiaux jusqu'au degré indiqué.

    Pour degre=2 et features [x1, x2] :
        → [x1, x2, x1², x1·x2, x2²]

    Cela permet de capturer des relations non-linéaires entre
    les features et le nombre de pièces.

    Attention : degré trop élevé → sur-apprentissage (overfitting).
    On règle le degré sur la base de validation.
    """

    n, p = X.shape
    colonnes = [X]
    for d in range(2, degre + 1):
        # Termes croisés et purs d'ordre d
        for i in range(p):
            for j in range(i, p):
                if d == 2:
                    colonnes.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(colonnes)


# ──────────────────────────────────────────────────────────────
# MÉTRIQUES D'ÉVALUATION
# ──────────────────────────────────────────────────────────────

def mae(
    y_vrai: np.ndarray, 
    y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error : MAE = (1/N) Σ |y - ŷ|

    Interprétation : en moyenne, notre modèle se trompe de X pièces.
    C'est la métrique principale pour ce problème car elle est
    directement interprétable en "nombre de pièces".
    """

    return float(np.mean(np.abs(y_vrai - y_pred)))


def mse(
    y_vrai: np.ndarray, 
    y_pred: np.ndarray) -> float:
    """
    Mean Squared Error : MSE = (1/N) Σ (y - ŷ)²

    Pénalise plus fortement les grandes erreurs (effet carré).
    Utile pour détecter les "catastrophes" (prédire 10 pièces quand il y en a 2).
    """

    return float(np.mean((y_vrai - y_pred) ** 2))


def rmse(
    y_vrai: np.ndarray, 
    y_pred: np.ndarray) -> float:
    """Root MSE — même unité que y (nb pièces)."""

    return float(np.sqrt(mse(y_vrai, y_pred)))


def afficher_metriques(
    y_vrai: np.ndarray, 
    y_pred: np.ndarray, 
    label: str = ""):
    """Affiche MAE, MSE et RMSE pour un set donné."""

    print(f"  {label} → MAE={mae(y_vrai, y_pred):.3f}, "
          f"MSE={mse(y_vrai, y_pred):.3f}, "
          f"RMSE={rmse(y_vrai, y_pred):.3f}")


# ──────────────────────────────────────────────────────────────
# MODÈLE 1 : RÉGRESSION LINÉAIRE (moindres carrés)
# ──────────────────────────────────────────────────────────────

class RegressionLineaire:
    """
    Régression linéaire : ŷ = Xw

    Apprentissage par la solution analytique des moindres carrés :
        w* = (XᵀX)⁻¹ Xᵀy


    Hyperparamètre exposé : aucun (le modèle le plus simple possible).
    """

    def __init__(self):
        self.poids = None       # Vecteur w (features + biais)
        self.min_ = None        # Pour la normalisation
        self.range_ = None      # Pour la normalisation

    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray):
        """
        Apprentissage sur la base d'entraînement.

        Étapes :
          1. Normaliser X_train (et stocker min/range pour val/test)
          2. Ajouter le terme de biais
          3. Résoudre w* = (XᵀX)⁻¹ Xᵀy via np.linalg.lstsq
             (lstsq est plus stable numériquement que l'inverse explicite)
        """

        X_norm, self.min_, self.range_ = normaliser(X_train, X_train)
        X_b = ajouter_biais(X_norm)
        # lstsq : résout le système surdéterminé Xw ≈ y au sens des MCO
        self.poids, _, _, _ = np.linalg.lstsq(X_b, y_train, rcond=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction sur un nouveau set.

        On applique la même normalisation que celle calculée sur le train,
        puis on arrondit à l'entier le plus proche (on prédit un nombre de pièces).
        """

        X_norm = (X - self.min_) / self.range_
        X_b = ajouter_biais(X_norm)
        y_pred = X_b @ self.poids
        # On arrondit et on force à ≥ 0 (un nb de pièces est positif)
        return np.maximum(0, np.round(y_pred)).astype(int)

    def score(
        self, 
        X: np.ndarray, 
        y: np.ndarray) -> dict:
        """Retourne un dict de métriques pour ce set."""

        y_pred = self.predict(X)
        return {
            "MAE": mae(y, y_pred),
            "MSE": mse(y, y_pred),
            "RMSE": rmse(y, y_pred),
        }


# ──────────────────────────────────────────────────────────────
# MODÈLE 2 : RÉGRESSION POLYNOMIALE
# ──────────────────────────────────────────────────────────────

class RegressionPolynomiale:
    """
    Régression polynomiale de degré d.

    On étend les features (x1, x2, …) en ajoutant x1², x1·x2, x2², …
    puis on applique une régression linéaire sur cet espace étendu.

    Avantage : capture des non-linéarités (ex : quand les pièces se
               chevauchent, la relation entre aire et nb_pièces n'est
               plus linéaire).
    Risque    : sur-apprentissage si degré trop élevé.

    Hyperparamètre : degré (réglé sur validation, typiquement 1, 2 ou 3).
    """

    def __init__(self, degre: int = 2):
        self.degre = degre
        self.modele_lin = RegressionLineaire()

    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray):
        """
        Apprentissage :
          1. Étendre les features au degré demandé
          2. Appliquer une régression linéaire sur l'espace étendu
        """

        X_poly = etendre_polynomial(X_train, self.degre)
        self.modele_lin.fit(X_poly, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = etendre_polynomial(X, self.degre)
        return self.modele_lin.predict(X_poly)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "MAE": mae(y, y_pred),
            "MSE": mse(y, y_pred),
            "RMSE": rmse(y, y_pred),
        }
