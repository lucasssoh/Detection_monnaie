import os
import json

def f(src_dir):
    results = {}

    for fichier in os.listdir(src_dir):
        if fichier.endswith('.json'):
            chemin_complet = os.path.join(src_dir, fichier)

            try:
                with open(chemin_complet, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    shapes = data.get('shapes', [])

                    results[fichier] = len(shapes)
            except Exception as e:
                print(f"erreur de lecture de {fichier} : {e}")
    return results

stats = f("base_annotations")

with open('verite_terrain.json', 'w', encoding='utf-8') as f_out:
    json.dump(stats, f_out, indent=4)

