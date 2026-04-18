"""
split_data.py
-------------
Divise la base d'images en 3 sets : train, val, test.
On garde les mêmes noms de fichiers que dans base_annotations/.

Stratégie : split aléatoire stratifié sur le nombre de pièces
afin d'avoir une distribution équilibrée dans chaque set.
"""

import json
import os
import random
import math
from collections import defaultdict


def charger_verite_terrain(annotation_dir: str) -> dict:
    """
    Lit tous les fichiers JSON d'annotations et retourne un dict :
        { nom_image: nb_pieces }

    On cherche dans chaque JSON la clé qui contient le total des pièces.
    Le format attendu dans chaque JSON est une liste d'annotations,
    et on compte le nombre d'entrées comme vérité terrain.
    """
    vt = {}
    for fname in os.listdir(annotation_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(annotation_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Le JSON contient une liste d'objets annotés (une entrée = une pièce)
        # On adapte selon le format réel — ici on suppose une liste
        if isinstance(data, list):
            nb_pieces = len(data)
        elif isinstance(data, dict) and "pieces" in data:
            nb_pieces = len(data["pieces"])
        else:
            # Fallback : si le format est différent, on met 0 et on avertit
            nb_pieces = 0
            print(f"[WARN] Format inconnu pour {fname}, nb_pieces mis à 0")

        nom_image = fname.replace(".json", "")
        vt[nom_image] = nb_pieces

    return vt


def split_stratifie(vt: dict, ratio_train=0.6, ratio_val=0.2, seed=42):
    """
    Split stratifié par nombre de pièces.

    On regroupe les images par nb_pieces, puis dans chaque groupe
    on fait un split proportionnel. Cela garantit que chaque set
    voit des images avec différents nombres de pièces.

    Paramètres :
        vt          : dict { nom_image: nb_pieces }
        ratio_train : proportion du set d'apprentissage (défaut 60%)
        ratio_val   : proportion du set de validation (défaut 20%)
        seed        : graine pour la reproductibilité

    Retourne :
        (train, val, test) : trois listes de noms d'images
    """
    random.seed(seed)

    # Regrouper par nb_pieces
    groupes = defaultdict(list)
    for nom, nb in vt.items():
        groupes[nb].append(nom)

    train, val, test = [], [], []

    for nb_pieces, images in groupes.items():
        random.shuffle(images)
        n = len(images)
        n_train = math.ceil(n * ratio_train)
        n_val = math.ceil(n * ratio_val)
        # Le reste va dans test
        train.extend(images[:n_train])
        val.extend(images[n_train:n_train + n_val])
        test.extend(images[n_train + n_val:])

    print(f"Split effectué :")
    print(f"  Train : {len(train)} images ({len(train)/len(vt)*100:.1f}%)")
    print(f"  Val   : {len(val)} images   ({len(val)/len(vt)*100:.1f}%)")
    print(f"  Test  : {len(test)} images  ({len(test)/len(vt)*100:.1f}%)")

    return train, val, test


def sauvegarder_split(train, val, test, output_path="split.json"):
    """
    Sauvegarde le split dans un fichier JSON pour reproductibilité.
    On ne refait le split qu'une fois — on réutilise ce fichier ensuite.
    """
    split = {"train": train, "val": val, "test": test}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"Split sauvegardé dans {output_path}")


def charger_split(split_path="split.json"):
    """Charge un split déjà calculé."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ANNOTATION_DIR = "base_annotations"

    vt = charger_verite_terrain(ANNOTATION_DIR)
    print(f"{len(vt)} images trouvées, nb_pieces min={min(vt.values())}, max={max(vt.values())}")

    train, val, test = split_stratifie(vt, ratio_train=0.6, ratio_val=0.2, seed=42)
    sauvegarder_split(train, val, test, "split.json")
