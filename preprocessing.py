"""
preprocessing.py
----------------
Pipeline de prétraitement et d'extraction de features.

Chaîne de traitement pour une image :
    Chargement (PIL → numpy)
        ↓
    Lissage gaussien (σ réglable)
        ↓
    Seuillage Otsu → image binaire
        ↓
    fill_holes (ferme les trous dans les blobs)
        ↓
    opening morphologique (supprime le bruit)
        ↓
    Extraction de features → vecteur numérique

Les features extraites sont ensuite utilisées par le modèle de régression
pour prédire le nombre de pièces dans l'image.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening

from otsu import trouver_seuil_otsu


# ──────────────────────────────────────────────────────────────
# CONSTANTES PAR DÉFAUT (hyperparamètres à régler sur validation)
# ──────────────────────────────────────────────────────────────

SIGMA_DEFAUT = 2          # Intensité du lissage gaussien
NOYAU_DEFAUT = 5          # Taille du noyau d'opening morphologique
SEUIL_TAILLE_REL = 0.2   # Un blob doit faire au moins 20% du plus grand


# ──────────────────────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Charge une image depuis le disque et la convertit en niveaux de gris.

    Pourquoi niveaux de gris ?
    Les pièces sont détectées par leur forme (cercles), pas leur couleur.
    Réduire à 1 canal simplifie le traitement et rend le seuillage Otsu applicable.

    Retourne : np.ndarray 2D de dtype uint8 (valeurs de 0 à 255)
    """
    img = Image.open(path).convert("L")
    return np.array(img)


# ──────────────────────────────────────────────────────────────
# SEUILLAGE
# ──────────────────────────────────────────────────────────────

def otsu_segmentation(image_array: np.ndarray) -> np.ndarray:
    """
    Applique le seuillage d'Otsu pour binariser l'image.

    L'image binaire résultante :
        1 (blanc) = zones claires = potentiellement des pièces
        0 (noir)  = fond sombre

    Note : selon l'éclairage, les pièces peuvent être plus claires
    ou plus sombres que le fond. Si les résultats sont mauvais,
    on peut inverser la binarisation (binary = 1 - binary).

    Retourne : np.ndarray 2D de dtype uint8 (valeurs 0 ou 1)
    """
    seuil = trouver_seuil_otsu(image_array)
    binary = (image_array >= seuil).astype(np.uint8)
    return binary


# ──────────────────────────────────────────────────────────────
# EXTRACTION DE FEATURES
# ──────────────────────────────────────────────────────────────

def extract_features(binary_image: np.ndarray,
                     seuil_taille_rel: float = SEUIL_TAILLE_REL) -> list:
    """
    Extrait un vecteur de 5 features à partir de l'image binaire.

    Ces features résument les propriétés géométriques des blobs détectés,
    et servent d'entrée au modèle de régression.

    Features extraites :
        [0] nb_pixels_utiles  : somme des pixels des blobs valides
        [1] nb_objets_valides : nombre de blobs (≈ nombre de pièces)
        [2] aire_moyenne      : aire moyenne des blobs valides (pixels)
        [3] aire_std          : écart-type des aires (mesure la régularité)
        [4] circ_moyenne      : circularité moyenne ∈ [0, 1]

    Paramètres :
        binary_image      : image binaire (0/1)
        seuil_taille_rel  : un blob est "valide" si son aire ≥ ce ratio
                            × l'aire du plus grand blob

    Retourne : liste de 5 floats (le vecteur de features)

    ------------------------------------------------------------------
    Détail de la circularité :
        circ = (4π × aire) / périmètre²
        Un cercle parfait → circ = 1.0
        Une forme allongée → circ ≈ 0.1-0.3
    Les pièces étant rondes, on attend circ ≈ 0.7-1.0 pour les vrais blobs.
    ------------------------------------------------------------------
    """
    labeled_array, nb_objets = ndimage.label(binary_image)

    # Cas dégénéré : aucun blob détecté
    if nb_objets == 0:
        return [0, 0, 0.0, 0.0, 0.0]

    # Calcul de l'aire de chaque blob
    aires_objets = [np.sum(labeled_array == i) for i in range(1, nb_objets + 1)]
    seuil_taille = max(aires_objets) * seuil_taille_rel

    # On ne garde que les blobs assez grands (filtre les artefacts)
    aires_valides = [a for a in aires_objets if a > seuil_taille]
    nb_objets_valides = len(aires_valides)

    if nb_objets_valides == 0:
        return [0, 0, 0.0, 0.0, 0.0]

    aire_moyenne = float(np.mean(aires_valides))
    aire_std = float(np.std(aires_valides))
    nb_pixels_utiles = int(sum(aires_valides))

    # Calcul de la circularité de chaque blob valide
    circularites = []
    for i in range(1, nb_objets + 1):
        aire_blob = np.sum(labeled_array == i)
        if aire_blob <= seuil_taille:
            continue  # blob trop petit, ignoré

        blob = (labeled_array == i).astype(np.uint8)

        # Périmètre estimé par les gradients de Sobel
        # (pixels non nuls dans au moins un des deux gradients = pixels de bord)
        contour_h = ndimage.sobel(blob, axis=0)
        contour_v = ndimage.sobel(blob, axis=1)
        perimetre = np.sum((contour_h != 0) | (contour_v != 0))

        if perimetre > 0:
            circularite = (4 * np.pi * aire_blob) / (perimetre ** 2)
            circularites.append(circularite)

    circ_moyenne = float(np.mean(circularites)) if circularites else 0.0

    return [nb_pixels_utiles, nb_objets_valides, aire_moyenne, aire_std, circ_moyenne]


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLET
# ──────────────────────────────────────────────────────────────

def preprocess_image(path: str,
                     sigma: float = SIGMA_DEFAUT,
                     noyau: int = NOYAU_DEFAUT) -> list:
    """
    Pipeline de prétraitement complet pour une image.

    Étapes :
        1. Chargement en niveaux de gris
        2. Lissage gaussien (paramètre : sigma)
        3. Seuillage Otsu → image binaire
        4. Remplissage des trous (binary_fill_holes)
        5. Opening morphologique (paramètre : noyau)
        6. Extraction des features

    Paramètres :
        path  : chemin vers l'image
        sigma : écart-type du filtre gaussien (réglé sur validation)
        noyau : taille du carré pour l'opening (réglé sur validation)

    Retourne :
        liste de 5 features (vecteur d'entrée du modèle)
    """
    # 1. Chargement
    img_array = load_image(path)

    # 2. Lissage gaussien
    #    Réduit le bruit avant seuillage pour éviter les faux positifs
    img_smooth = gaussian_filter(img_array.astype(np.float32), sigma=sigma)
    img_smooth = img_smooth.astype(np.uint8)

    # 3. Seuillage Otsu
    binary = otsu_segmentation(img_smooth)

    # 4. Remplissage des trous
    #    Une pièce peut avoir des reflets qui créent des "trous" internes.
    #    binary_fill_holes les ferme pour que chaque pièce soit un blob continu.
    binary = binary_fill_holes(binary).astype(np.uint8)

    # 5. Opening morphologique (érosion puis dilatation)
    #    Supprime les petits artefacts (bruit, poussière) sans déformer
    #    les grosses formes (les pièces).
    struct = np.ones((noyau, noyau))
    binary = binary_opening(binary, structure=struct).astype(np.uint8)

    # 6. Extraction des features
    return extract_features(binary)


def preprocess_dataset(image_paths: list,
                       sigma: float = SIGMA_DEFAUT,
                       noyau: int = NOYAU_DEFAUT) -> np.ndarray:
    """
    Applique preprocess_image à une liste d'images.

    Retourne une matrice X de forme (N, 5) où N = nombre d'images.
    Chaque ligne est le vecteur de features d'une image.

    Utilisé pour construire les sets train, val et test.
    """
    X = []
    for path in image_paths:
        try:
            features = preprocess_image(path, sigma=sigma, noyau=noyau)
            X.append(features)
        except Exception as e:
            print(f"[WARN] Erreur sur {path} : {e}")
            X.append([0, 0, 0.0, 0.0, 0.0])
    return np.array(X, dtype=np.float64)
