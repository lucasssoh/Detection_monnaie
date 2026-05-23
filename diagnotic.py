"""
diagnostic.py
-------------
Lance ce script EN PREMIER pour vérifier que tout est en ordre.
Il ne modifie rien, il lit et affiche seulement.

Usage : python diagnostic.py
"""

import json
import os
from collections import Counter

ANNOTATION_DIR = "base_annotations"
IMAGE_DIR = "base_images"

print("=" * 55)
print("DIAGNOSTIC DU PROJET")
print("=" * 55)

# ── 1. Lire tous les JSON ─────────────────────────────────────
print("\n[1] Lecture des annotations...")
vt = {}
erreurs = []

for fname in sorted(os.listdir(ANNOTATION_DIR)):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(ANNOTATION_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Chaque shape = une pièce (label = valeur : 1_cents, 2_euros, etc.)
        nb = len(data.get("shapes", []))
        vt[fname.replace(".json", "")] = nb
    except Exception as e:
        erreurs.append(f"  ✗ {fname} : {e}")

print(f"  → {len(vt)} annotations lues")
if erreurs:
    print("  Erreurs :")
    for e in erreurs: print(e)

# ── 2. Distribution du nb de pièces ──────────────────────────
print("\n[2] Distribution du nombre de pièces :")
compt = Counter(vt.values())
for nb in sorted(compt):
    barre = "█" * compt[nb]
    print(f"  {nb:2d} pièce(s) : {barre} ({compt[nb]} images)")
print(f"  Min={min(vt.values())}  Max={max(vt.values())}  "
      f"Moy={sum(vt.values())/len(vt):.1f}")

# ── 3. Vérifier que chaque annotation a son image ────────────
print("\n[3] Correspondance annotations ↔ images...")
extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
manquantes = []
for nom in vt:
    trouve = any(
        os.path.exists(os.path.join(IMAGE_DIR, nom + ext))
        for ext in extensions
    )
    if not trouve:
        # essai avec le nom seul (contient peut-être déjà une extension)
        trouve = os.path.exists(os.path.join(IMAGE_DIR, nom))
    if not trouve:
        manquantes.append(nom)

if manquantes:
    print(f"  ✗ {len(manquantes)} images introuvables :")
    for m in manquantes[:10]:
        print(f"    - {m}")
    if len(manquantes) > 10:
        print(f"    ... et {len(manquantes)-10} autres")
else:
    print(f"  ✓ Toutes les {len(vt)} images trouvées")

# ── 4. Images sans annotation ─────────────────────────────────
print("\n[4] Images sans annotation (ignorées) :")
images_dossier = set()
for f in os.listdir(IMAGE_DIR):
    nom_sans_ext = os.path.splitext(f)[0]
    images_dossier.add(nom_sans_ext)
sans_annot = images_dossier - set(vt.keys())
print(f"  → {len(sans_annot)} images sans JSON (ne seront pas utilisées)")

# ── 5. Résumé final ───────────────────────────────────────────
print("\n" + "=" * 55)
n_utilisables = len(vt) - len(manquantes)
print(f"  Images utilisables : {n_utilisables}")
print(f"  Split prévu : ~{int(n_utilisables*0.6)} train  "
      f"/ ~{int(n_utilisables*0.2)} val  "
      f"/ ~{int(n_utilisables*0.2)} test")

if n_utilisables < 10:
    print("\n  ⚠ Moins de 10 images utilisables — résultats non significatifs")
elif n_utilisables < 30:
    print("\n  ⚠ Peu de données — attends-toi à une MAE variable")
else:
    print("\n  ✓ Prêt à lancer : python split_data.py")
print("=" * 55)
