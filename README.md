# Documentation Technique : Projet de Comptage de Pièces

Ce projet implémente une chaîne de traitement (pipeline) allant de l'image brute à la prédiction numérique du nombre de pièces.

---

## 1. Extraction des caractéristiques (`preprocessing.py`)

L'objectif est de transformer une image complexe en données chiffrées exploitables par un modèle mathématique.

### a. Segmentation de l'image
* **Conversion en gris** : Réduction de l'information couleur pour ne garder que l'intensité lumineuse.
* **Flou Gaussien (`sigma`)** : Élimination du bruit numérique (les "grains" de l'image) pour lisser les formes.
* **Seuillage d'Otsu** : Algorithme qui sépare automatiquement l'image en deux : les objets (blanc) et le fond (noir).
* **Nettoyage Morphologique** : 
    * `fill_holes` : Rebouche les trous créés par les reflets sur le métal des pièces.
    * `opening` (`noyau`) : Supprime les petits résidus blancs qui ne sont pas des pièces.

### b. Mesures effectuées (`extract_features`)
Pour chaque image, on extrait un vecteur de **5 caractéristiques (features)** :
1.  **Aire totale** : Somme des pixels blancs.
2.  **Nombre de blobs** : Nombre de formes distinctes détectées.
3.  **Aire moyenne** : Taille moyenne des objets.
4.  **Écart-type des aires** : Indique si les objets ont des tailles similaires ou variées.
5.  **Circularité moyenne** : Score de "rondeur" des objets.
    * **Formule** : $$C = \frac{4\pi \times A}{P^2}$$
    * *(A = Aire, P = Périmètre). Un score proche de 1 indique un cercle parfait.*

---

## 2. Modèle Statistique (`regression.py`)

On utilise les 5 mesures précédentes pour estimer le nombre exact de pièces, notamment pour corriger les erreurs de la segmentation (ex: deux pièces collées).

### a. Préparation des données
* **Normalisation** : On ramène toutes les mesures entre 0 et 1. Cela évite qu'une grande valeur (l'aire) ne masque une petite valeur importante (la circularité).
* **Ajout du Biais** : On insère une colonne de "1" dans la matrice de données. Mathématiquement, cela permet au modèle de calculer une constante (l'ordonnée à l'origine), garantissant que la droite de prédiction ne passe pas forcément par l'origine (0,0).

### b. Choix du modèle
* **Régression Linéaire** : Calcule un poids pour chaque caractéristique.
    * Équation : $\hat{y} = X \cdot w$
    * Résolution : $w = (X^T X)^{-1} X^T y$
* **Régression Polynomiale** : Si le degré est > 1, le script crée des combinaisons de variables (ex: $aire^2$). Utile si la relation entre les mesures et le nombre de pièces n'est pas une ligne droite parfaite.

### c. Métriques de performance
* **MAE (Mean Absolute Error)** : Erreur moyenne absolue. Si MAE = 0.2, le modèle se trompe en moyenne de 0.2 pièce par image.
* **MSE (Mean Squared Error)** : Erreur quadratique. Elle pénalise plus lourdement les grosses erreurs de prédiction.

---

## 3. Stratégie d'Entraînement (`algo.py`)

### a. Organisation des données (Split)
Pour garantir la fiabilité du modèle, on sépare les images en trois groupes distincts :
1.  **Train** : Images utilisées pour apprendre les poids du modèle.
2.  **Validation** : Images utilisées pour tester différentes combinaisons d'hyperparamètres (choisir le meilleur flou, le meilleur degré, etc.).
3.  **Test** : Images conservées pour l'évaluation finale, jamais vues par le modèle durant son réglage.

### b. Recherche d'optimum (Grid Search)
Le script teste systématiquement tous les réglages définis dans la grille de paramètres (`sigma`, `noyau`, `degre`). Il sélectionne la combinaison qui minimise la MAE sur le jeu de **Validation**.

### c. Prédiction finale
Le résultat prédit par la régression est un nombre décimal (ex: 4.82). 
* **Post-traitement** : On applique un arrondi à l'entier le plus proche (`round`) et on s'assure que le résultat n'est jamais inférieur à 0.