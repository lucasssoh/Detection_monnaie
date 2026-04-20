"""
otsu.py
-------
Seuillage automatique d'Otsu.

Principe : on cherche le seuil t* qui minimise la variance intra-classe,
c'est-à-dire la somme pondérée des variances des deux groupes (fond / objet).

Minimiser la variance intra-classe est équivalent à maximiser la variance
inter-classe (propriété mathématique d'Otsu, 1979).

Complexité : O(256) — on teste tous les seuils possibles sur l'histogramme.
"""

import numpy as np


def calculer_histogramme(image: np.ndarray) -> np.ndarray:
    """
    Calcule l'histogramme d'une image en niveaux de gris (uint8).
    Retourne un vecteur de 256 valeurs : hist[i] = nb pixels d'intensité i.
    """
    return np.bincount(image.ravel(), minlength=256)


def calculer_poids(
    hist: np.ndarray, 
    debut: int, 
    fin: int, 
    total_pixels: int) -> float:
    """
    Poids d'une classe = proportion de pixels dans [debut, fin].
    w = Σ hist[i] / N   pour i ∈ [debut, fin]
    """

    return np.sum(hist[debut:fin + 1]) / total_pixels


def calculer_moyenne(
    hist: np.ndarray, 
    debut: int, 
    fin: int, 
    poids: float) -> float:
    """
    Moyenne de l'intensité dans la classe [debut, fin].
    μ = (1 / (w * N)) * Σ i * hist[i]   pour i ∈ [debut, fin]
    """

    if poids == 0:
        return 0.0
    intensites = np.arange(debut, fin + 1)
    total_pixels = np.sum(hist)
    somme_ponderee = np.sum(intensites * hist[debut:fin + 1])
    return (somme_ponderee / total_pixels) / poids


def calculer_variance(
    hist: np.ndarray, 
    debut: int, 
    fin: int,
    poids: float, moyenne: float) -> float:
    """
    Variance intra-classe pour la classe [debut, fin].
    σ² = Σ (i - μ)² * p(i|classe)   pour i ∈ [debut, fin]
    où p(i|classe) = hist[i] / (N * w)
    """

    if poids == 0:
        return 0.0
    intensites = np.arange(debut, fin + 1)
    total_pixels = np.sum(hist)
    probabilites = hist[debut:fin + 1] / (total_pixels * poids)
    return np.sum(((intensites - moyenne) ** 2) * probabilites)


def evaluer_seuil_intra_classe(
    hist: np.ndarray, 
    t: int, 
    total_pixels: int) -> float:
    """
    Calcule la variance intra-classe totale pour un seuil t :
        σ²_intra(t) = w1(t) * σ²1(t) + w2(t) * σ²2(t)
    Classe 1 : pixels d'intensité [0, t-1]  (fond sombre)
    Classe 2 : pixels d'intensité [t, 255]  (objets clairs)
    """

    w1 = calculer_poids(hist, 0, t - 1, total_pixels)
    m1 = calculer_moyenne(hist, 0, t - 1, w1)
    v1 = calculer_variance(hist, 0, t - 1, w1, m1)

    w2 = calculer_poids(hist, t, 255, total_pixels)
    m2 = calculer_moyenne(hist, t, 255, w2)
    v2 = calculer_variance(hist, t, 255, w2, m2)

    return (w1 * v1) + (w2 * v2)


def get_width(image: np.ndarray) -> int:
    return image.shape[1]


def get_height(image: np.ndarray) -> int:
    return image.shape[0]


def trouver_seuil_otsu(image: np.ndarray) -> int:
    """
    Trouve le seuil optimal d'Otsu pour une image en niveaux de gris.

    Algorithme :
      1. Calculer l'histogramme de l'image
      2. Pour chaque seuil t de 1 à 255 :
           calculer la variance intra-classe σ²_intra(t)
      3. Retourner t* = argmin σ²_intra(t)

    Paramètre :
        image : np.ndarray 2D (uint8 ou float converti en uint8)

    Retourne :
        seuil (int) : valeur entre 1 et 255
    """
    img_array = np.asarray(image)
    hist = calculer_histogramme(img_array)
    total_pixels = get_width(img_array) * get_height(img_array)

    meilleur_seuil = 1
    variance_minimale = float("inf")

    for t in range(1, 256):
        v_intra = evaluer_seuil_intra_classe(hist, t, total_pixels)
        if v_intra < variance_minimale:
            variance_minimale = v_intra
            meilleur_seuil = t

    return meilleur_seuil
