import cv2
import numpy as np
import matplotlib.pyplot as plt

import descente_diff_fini_6
import gradient_sensibilite_6
import gradient_sensibilite_12

def get_params():
    # conditions initiales
    params_6 = np.array([
        180, 0.0, 0.0,   # gamma, beta, alpha
        10.0, 10.0, 50.0  # tx, ty, tz
    ], dtype=float)
    params_12 = np.array([
        1, 0, 0,   # première ligne de la matrice (identité)
        0, 0, -1,   # deuxième ligne
        0, 0, 0,   # troisième ligne
        10, 10, 50   # translation (tx, ty, tz)
    ], dtype=float) 
    
    return params_6, params_12

# Images et descriptions
image_files = ["img_tag_mur.jpg", "img_tag_sol.jpg", "img_2_tags.jpg"]
img_descriptions = [
    "Image contenant 1 tag au mur (4 points)",
    "Image contenant 1 tag au sol (4 points)",
    "Image contenant 2 tags non coplanaires (8 points)"
]

# Vrai vecteur de translation et angles (en degrés)
t_vrai = np.array([11.7, 20.7, 100])
angles_vrais_deg = np.array([180, 0.0, 0.0])  # Mets ici les vrais angles si tu les as

def angle_diff(a, b):
    diff = a - b
    diff = (diff + 180) % 360 - 180
    return diff

# Stockage résultats
t_estimations = []
angles_estimations = []
names_list = [
    "Gradient avec différence finie à 6 paramètres",
    "Gradient avec fonction de sensibilité à 6 paramètres",
    "Gradient avec fonction de sensibilité à 12 paramètres"
]

for img_file, img_desc in zip(image_files, img_descriptions):

    img = cv2.imread(img_file)

    cost_history_list = []
    iter_checkpoints = []
    t_list = []
    angles_list = []

    params_6, params_12 = get_params()

    # Méthode 1
    _, angles_deg, t, cost_history, iter_checkpoint, p = descente_diff_fini_6.calc_descente_diff_fini_6(img, params_6)
    cost_history_list.append(cost_history)
    iter_checkpoints.append(iter_checkpoint)
    t_list.append(np.array(t))
    angles_list.append(np.array(angles_deg))

    params_6, params_12 = get_params()

    # Méthode 2
    _, angles_deg, t, cost_history, iter_checkpoint, p = gradient_sensibilite_6.calc_gradient_sensibilite_6(img, params_6)
    cost_history_list.append(cost_history)
    iter_checkpoints.append(iter_checkpoint)
    t_list.append(np.array(t))
    angles_list.append(np.array(angles_deg))

    params_6, params_12 = get_params()

    # Méthode 3
    _, angles_deg, t, cost_history, iter_checkpoint, p = gradient_sensibilite_12.calc_gradient_sensibilite_12(img, params_12)
    cost_history_list.append(cost_history)
    iter_checkpoints.append(iter_checkpoint)
    t_list.append(np.array(t))
    angles_list.append(np.array(angles_deg))


    t_estimations.append(t_list)
    angles_estimations.append(angles_list)

    # Affichage évolution fonction coût pour cette image
    plt.figure(figsize=(10, 6))
    for name, cost_history, checkpoint in zip(names_list, cost_history_list, iter_checkpoints):
        iterations = [checkpoint * i for i in range(1, len(cost_history) + 1)]
        plt.plot(iterations, cost_history, label=name)

    plt.yscale('log')
    plt.xlabel("Itérations")
    plt.ylabel("Coût (échelle log)")
    plt.title(f"Évolution du coût - {img_desc}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

t_estimations = np.array(t_estimations)
angles_estimations = np.array(angles_estimations)

# Calcul erreurs absolues translation et angles
erreur_t = np.abs(t_estimations - t_vrai)

erreur_angles = np.zeros_like(angles_estimations)
for i_img in range(len(image_files)):
    for i_meth in range(3):
        erreur_angles[i_img, i_meth] = np.abs(angle_diff(angles_estimations[i_img, i_meth], angles_vrais_deg))

method_labels = [
    "Différence finie 6 params",
    "Sensibilité 6 params",
    "Sensibilité 12 params"
]

# Affichage erreurs translation
components = ['X', 'Y', 'Z']
for i_comp in range(3):
    plt.figure(figsize=(8, 5))
    for i_meth in range(3):
        plt.plot(img_descriptions, erreur_t[:, i_meth, i_comp], marker='o', label=method_labels[i_meth])
    plt.title(f"Erreur absolue translation {components[i_comp]}")
    plt.xlabel("Image")
    plt.ylabel("Erreur (unités de t)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# Affichage erreurs angles
angle_components = ['gamma', 'beta', 'alpha']
for i_comp in range(3):
    plt.figure(figsize=(8, 5))
    for i_meth in range(3):
        plt.plot(img_descriptions, erreur_angles[:, i_meth, i_comp], marker='o', label=method_labels[i_meth])
    plt.title(f"Erreur absolue angle {angle_components[i_comp]}")
    plt.xlabel("Image")
    plt.ylabel("Erreur (degrés)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

plt.pause(0)
