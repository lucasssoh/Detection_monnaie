# Projet de Comptage de Pièces Euro
## Traitement des Images Numériques — Université Paris Cité

Ce projet implémente une chaîne de traitement complète (*pipeline*) allant de l'image brute à la prédiction numérique du nombre de pièces euro présentes dans une photo.

---

## Structure du projet

```
Detection_monnaie/
├── base_images/           ← images de pièces (non incluses, voir ci-dessous)
├── base_annotations/      ← annotations JSON au format LabelMe
├── algo.py                ← script principal : entraînement et évaluation
├── otsu.py                ← implémentation du seuillage d'Otsu
├── preprocessing.py       ← prétraitement et extraction de features
├── regression.py          ← modèles de régression (linéaire et polynomiale)
├── split_data.py          ← découpage du dataset en train / val / test
├── diagnostic.py          ← vérification de l'intégrité des données
├── split.json             ← split figé (généré une seule fois)
└── README.md
```

---

## Prérequis

Python 3.10+ et les bibliothèques suivantes :

```bash
pip install numpy pillow scipy matplotlib
```

---

## Données requises (non incluses dans le rendu)

Les dossiers d'images ne sont pas inclus car ils sont trop volumineux.
Pour reproduire les résultats, il faut placer à la racine du projet :

- `base_images/` : les photos de pièces (formats `.jpg`, `.jpeg`, `.JPG`, `.png`)
- `base_annotations/` : les fichiers JSON au format LabelMe correspondants
  - Un fichier JSON par image, même nom que l'image
  - Chaque *shape* dans le JSON correspond à une pièce annotée
  - Les labels sont les valeurs des pièces : `1_cents`, `2_euros`, etc.

---

## Ordre d'exécution

### Étape 0 — Vérification des données

Lancer ce script en premier pour s'assurer que tout est en ordre.
Il ne modifie rien, il lit et affiche uniquement.

```bash
python diagnostic.py
```

Ce qu'on doit voir :
- Toutes les annotations lues sans erreur
- La distribution du nombre de pièces par image
- La correspondance annotations ↔ images (aucune image introuvable)

---

### Étape 1 — Création du split (une seule fois)

```bash
python split_data.py
```

Crée le fichier `split.json` qui répartit les images en trois groupes de manière **stratifiée** : chaque groupe contient la même proportion d'images par nombre de pièces. Ce fichier est généré une seule fois et ne doit pas être recréé, pour garantir la reproductibilité des résultats.

> Si `split.json` existe déjà, `algo.py` le charge directement sans le recréer.

---

### Étape 2 — Entraînement et évaluation

```bash
python algo.py
```

Ce script exécute l'intégralité du pipeline :

1. Charge le split depuis `split.json`
2. Pré-calcule les features pour toutes les combinaisons sigma/noyau
3. Teste la grille d'hyperparamètres sur la base de **validation**
4. Sélectionne les meilleurs hyperparamètres (MAE minimale sur validation)
5. Évalue le modèle final **une seule fois** sur la base de **test**

---

## Documentation Technique

### 1. Extraction des caractéristiques (`preprocessing.py`)

L'objectif est de transformer une image complexe en données chiffrées exploitables par un modèle mathématique.

#### a. Segmentation de l'image

- **Conversion en gris** : réduction de l'information couleur pour ne garder que l'intensité lumineuse.
- **Flou Gaussien (`sigma`)** : élimination du bruit numérique (les "grains" de l'image) pour lisser les formes avant le seuillage.
- **Seuillage d'Otsu** : algorithme qui sépare automatiquement l'image en deux classes — les objets (blanc) et le fond (noir) — en minimisant la variance intra-classe sur l'ensemble des 256 seuils possibles. Il garantit toujours l'optimum global, contrairement à K-means qui converge vers un optimum local dépendant de l'initialisation.
- **Nettoyage Morphologique** :
  - `fill_holes` : rebouche les trous créés par les reflets sur le métal des pièces.
  - `opening` (`noyau`) : érosion suivie d'une dilatation — supprime les petits résidus blancs sans déformer les grandes formes.

#### b. Mesures effectuées (`extract_features`)

Pour chaque image, on extrait un vecteur de **5 caractéristiques (features)**. Seuls les blobs faisant au moins 20 % de la taille du plus grand blob sont considérés comme valides (filtrage des artefacts résiduels).

| # | Feature | Description |
|---|---|---|
| 1 | `nb_pixels_utiles` | Somme des pixels blancs des blobs valides |
| 2 | `nb_objets_valides` | Nombre de blobs valides détectés (≈ nb pièces) |
| 3 | `aire_moyenne` | Taille moyenne des blobs valides (en pixels) |
| 4 | `aire_std` | Écart-type des tailles — monte si des pièces se touchent |
| 5 | `circ_moyenne` | Score de rondeur moyen des blobs valides |

**Formule de la circularité :**

$$C = \frac{4\pi \times A}{P^2}$$

*(A = Aire, P = Périmètre estimé par gradient de Sobel). Un cercle parfait donne C = 1. Un artefact fin ou irrégulier donne C ≈ 0.1–0.3.*

La régression n'utilise jamais les pixels bruts — elle travaille uniquement sur ce vecteur de 5 nombres.

---

### 2. Modèle Statistique (`regression.py`)

On utilise les 5 features pour estimer le nombre de pièces, notamment pour corriger les erreurs de segmentation (ex : deux pièces collées forment un seul blob, mais `aire_std` monte et le modèle le détecte).

#### a. Préparation des données

- **Normalisation min-max** : toutes les features sont ramenées entre 0 et 1. Calculée sur le train uniquement, appliquée à val et test — pour éviter toute fuite d'information.
- **Ajout du biais** : une colonne de `1` est ajoutée à la matrice. Cela permet au modèle d'avoir une constante (ordonnée à l'origine), sans forcer la droite à passer par (0, 0).

#### b. Choix du modèle

- **Régression Linéaire** : calcule un poids pour chaque feature.
  - Équation : $\hat{y} = X \cdot w$
  - Résolution analytique (moindres carrés) : $w^* = (X^T X)^{-1} X^T y$
  - Implémentée via `np.linalg.lstsq` (plus stable numériquement que l'inverse explicite)

- **Régression Polynomiale** : si le degré est > 1, des termes croisés sont ajoutés ($aire^2$, $aire \times circularité$, etc.). Utile si la relation entre features et nombre de pièces n'est pas linéaire.

#### c. Métriques de performance

- **MAE (Mean Absolute Error)** : $\frac{1}{N}\sum|y - \hat{y}|$ — erreur moyenne en nombre de pièces. Si MAE = 0.8, le modèle se trompe en moyenne de moins d'une pièce.
- **MSE (Mean Squared Error)** : $\frac{1}{N}\sum(y - \hat{y})^2$ — pénalise plus lourdement les grosses erreurs.

---

### 3. Stratégie d'Entraînement (`algo.py`)

#### a. Organisation des données — Split stratifié

Les images sont réparties en trois groupes de façon **stratifiée par nombre de pièces** : pour chaque valeur (1 pièce, 2 pièces, etc.), on répartit proportionnellement dans chaque groupe. Cela garantit que train, val et test voient la même diversité de cas.

| Set | Proportion | Rôle |
|---|---|---|
| **Train** | 60 % (~105 images) | Apprentissage des poids du modèle |
| **Validation** | 20 % (~35 images) | Réglage des hyperparamètres |
| **Test** | 20 % (~35 images) | Évaluation finale — utilisé une seule fois |


#### b. Recherche d'optimum — Grid Search

Le script teste systématiquement toutes les combinaisons de la grille suivante :

| Hyperparamètre | Valeurs testées | Rôle |
|---|---|---|
| `sigma` | 1, 2, 3 | Intensité du flou gaussien |
| `noyau` | 3, 5, 7 | Taille de l'opening morphologique |
| `degre` | 1, 2 | Degré de la régression polynomiale |

Soit **18 combinaisons** au total. Les features sont calculées une seule fois par paire (sigma, noyau) et mises en cache — seule la régression (instantanée) est relancée pour chaque degré.

La combinaison retenue est celle qui minimise la **MAE sur la base de validation**.

#### c. Prédiction finale

Le résultat de la régression est un nombre décimal (ex : 4.82). Un arrondi à l'entier le plus proche est appliqué, et le résultat est contraint à être ≥ 0.

---

## Limites connues du système

- **Pièces qui se touchent** : deux pièces collées forment un seul blob. Le modèle tente de compenser via `aire_std`, mais sous-compte dans les cas extrêmes.
- **Déséquilibre du dataset** : 90 des 176 images n'ont qu'une seule pièce, ce qui biaise légèrement le modèle vers les faibles valeurs.
- **Image outlier** : une image contient 22 pièces — cas unique dans le dataset, probablement mal prédit.
- **Éclairage variable** : les photos de groupes ont été prises dans des conditions différentes, ce qui affecte la robustesse du seuillage Otsu.
