import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_opening
from otsu import trouver_seuil_otsu

def otsu_segmentation(image_array):
    """
    Retourne une image binaire après le seuillage (ici on utilise Otsu)
    image_array : image numpy array en niveau de gris
    """
    seuil = trouver_seuil_otsu(image_array)
    binary = (image_array >= seuil).astype(np.uint8)
    return binary

def extract_features(binary_image):
    """
    Extrait des features pour le comptage des pieces.
    features : les caracteristiques pour definir une piece (c'est une approximation)
    """
    """
    ndimage.label parcourt l'image binaire et regroupe tous les pixels connectés (1) en "blobs" distincts.
    Chaque blob correspond à un objet séparé (ici, potentiellement une pièce). 
    La fonction renvoie :
    - labeled_array : un tableau de la même taille que l'image, où chaque pixel blanc appartient à un blob identifié par un entier unique (1, 2, 3…)
    - nb_objets : le nombre total de blobs détectés (approximation du nombre de pièces)
    Exemple :
    binary_image :
    0 0 1 1 0
    0 0 1 1 0
    1 0 0 0 1
    Après label :
    0 0 1 1 0
    0 0 1 1 0
    2 0 0 0 3
    Ici, nb_objets = 3
    """
    labeled_array, nb_objets = ndimage.label(binary_image)
    
    if nb_objets > 0:
        aires_objets = [np.sum(labeled_array == i) for i in range(1, nb_objets + 1)]
        """
        Un objet valide doit avoir au moins 20% de la taille de l'objet le plus gros
        """
        seuil_taille = np.max(aires_objets) * 0.2
        aires_valides = [a for a in aires_objets if a > seuil_taille]
        nb_objets_valides = len(aires_valides)
        aire_moyenne = np.mean(aires_valides) if nb_objets_valides > 0 else 0
        nb_pixels_utiles = sum(aires_valides)

        """
        Calcul de la circularité moyenne des objets valides.
        La circularité mesure à quel point un blob ressemble à un cercle.
        
        Formule : circularité = (4 * pi * aire) / périmètre²
        
        - Un cercle parfait donne une circularité de 1.0
        - Une forme allongée ou irrégulière donne une valeur proche de 0
        
        Pour calculer le périmètre, on utilise le filtre de Sobel qui détecte
        les contours (transitions 0->1 ou 1->0) dans le blob.
        On compte ensuite le nombre de pixels de contour détectés.
        
        Exemple :
        blob d'une pièce ronde  -> circularité ≈ 0.85~1.0
        blob d'un artefact fin  -> circularité ≈ 0.1~0.3
        
        Cela permet de filtrer les faux positifs qui ne sont pas des pièces rondes.
        """
        circularites = []
        for i in range(1, nb_objets + 1):
            aire_blob = np.sum(labeled_array == i)
            if aire_blob > seuil_taille:
                blob = (labeled_array == i).astype(np.uint8)
                """
                ndimage.sobel calcule le gradient (les contours) selon l'axe horizontal (axis=0)
                et vertical (axis=1). On combine les deux pour obtenir tous les pixels de bord.
                Un pixel de contour est non nul dans au moins l'un des deux gradients.
                """
                contour_h = ndimage.sobel(blob, axis=0)
                contour_v = ndimage.sobel(blob, axis=1)
                perimetre = np.sum((contour_h != 0) | (contour_v != 0))
                if perimetre > 0:
                    circularite = (4 * np.pi * aire_blob) / (perimetre ** 2)
                    circularites.append(circularite)

        circ_moyenne = np.mean(circularites) if circularites else 0
        aire_std = np.std(aires_valides) if nb_objets_valides > 0 else 0

    return [nb_pixels_utiles, nb_objets_valides, aire_moyenne, aire_std, circ_moyenne]

def load_image(path):
    """
    Charge une image en niveau de gris et retourne un tableau numpy
    """
    img = Image.open(path).convert("L")
    return np.array(img)

def preprocess_image(path):
    """
    Processus de pretraitement complet : chargement -> lissage -> seuillage -> morphologie -> features
    """
    img_array = load_image(path)

    """
    Lissage gaussien : on applique un flou gaussien sur l'image avant le seuillage.
    
    Pourquoi ? Une image brute contient souvent du bruit (pixels parasites, variations
    d'éclairage locales). Sans lissage, ces pixels bruités créent de nombreux petits blobs
    indésirables après seuillage, ce qui fausse le comptage.
    
    Le paramètre sigma contrôle l'intensité du flou :
    - sigma faible (ex: 1) -> léger lissage, conserve les détails fins
    - sigma élevé (ex: 3+) -> fort lissage, peut fusionner des pièces proches
    sigma=2 est un bon compromis pour des pièces bien séparées.
    
    On convertit en float32 pour le calcul, puis on repasse en uint8 (0-255) pour Otsu.
    """
    img_smooth = gaussian_filter(img_array.astype(np.float32), sigma=2)
    img_smooth = img_smooth.astype(np.uint8)

    binary = otsu_segmentation(img_smooth)

    """
    Remplissage des trous : binary_fill_holes remplit les pixels noirs (0) complètement
    entourés par des pixels blancs (1) à l'intérieur d'un blob.
    
    Pourquoi ? Une pièce peut avoir des reflets ou zones sombres en son centre,
    ce qui crée des "trous" dans le blob après seuillage. Ces trous fragmentent
    la pièce en plusieurs blobs, ce qui fausse le comptage.
    
    Exemple :
    Avant fill_holes :     Après fill_holes :
    0 1 1 1 0              0 1 1 1 0
    1 1 0 1 1    ->        1 1 1 1 1
    1 0 0 0 1              1 1 1 1 1
    1 1 0 1 1              1 1 1 1 1
    0 1 1 1 0              0 1 1 1 0
    """
    binary = binary_fill_holes(binary).astype(np.uint8)

    """
    Ouverture morphologique : binary_opening effectue une érosion suivie d'une dilatation.
    
    - Érosion  : réduit chaque blob en rongeant ses bords -> supprime les petits blobs (bruit)
    - Dilatation : regonfle les blobs restants à leur taille d'origine
    
    Résultat : les petits artefacts (poussière, reflets parasites) disparaissent,
    tandis que les grandes formes (les pièces) sont conservées avec leur forme initiale.
    
    La structure définit le "pinceau" utilisé pour l'érosion/dilatation.
    np.ones((5,5)) = carré 5x5 de 1 -> opération appliquée dans un voisinage 5x5.
    Un noyau plus grand supprime des objets plus gros mais peut aussi fusionner des pièces proches.
    """
    struct = np.ones((5, 5))
    binary = binary_opening(binary, structure=struct).astype(np.uint8)

    features = extract_features(binary)
    return features
