import numpy as np

def calculer_histogramme(image):
    return np.bincount(image.ravel(), minlength=256)

def calculer_poids(hist, debut, fin, total_pixels):
    nb_pixels_classe = np.sum(hist[debut:fin+1])
    return nb_pixels_classe / total_pixels

def calculer_moyenne(hist, debut, fin, poids):
    if poids == 0:
        return 0.0
    
    intensites = np.arange(debut, fin + 1)
    total_pixels = np.sum(hist) 
    somme_ponderee = np.sum(intensites * hist[debut:fin+1])
    
    return (somme_ponderee / total_pixels) / poids

def calculer_variance(hist, debut, fin, poids, moyenne):
    if poids == 0:
        return 0.0
    
    intensites = np.arange(debut, fin + 1)
    total_pixels = np.sum(hist)
    
    probabilites = hist[debut:fin+1] / (total_pixels * poids)
    variance = np.sum(((intensites - moyenne) ** 2) * probabilites)
    
    return variance

def evaluer_seuil_intra_classe(hist, t, total_pixels):
    w1 = calculer_poids(hist, 0, t-1, total_pixels)
    m1 = calculer_moyenne(hist, 0, t-1, w1)
    v1 = calculer_variance(hist, 0, t-1, w1, m1)
    
    w2 = calculer_poids(hist, t, 255, total_pixels)
    m2 = calculer_moyenne(hist, t, 255, w2)
    v2 = calculer_variance(hist, t, 255, w2, m2)
    
    return (w1 * v1) + (w2 * v2)

def get_width(image):
    return image.shape[1]

def get_height(image):
    return image.shape[0]

def trouver_seuil_otsu(image):
    img_array = np.asarray(image)
    hist = calculer_histogramme(img_array)
    total_pixels = get_width(img_array) * get_height(img_array)

    meilleur_seuil = 0
    variance_minimale = float('inf')

    for t in range(1, 256):
        v_intra = evaluer_seuil_intra_classe(hist, t, total_pixels)

        if v_intra < variance_minimale:
            variance_minimale = v_intra
            meilleur_seuil = t        
    
    return meilleur_seuil