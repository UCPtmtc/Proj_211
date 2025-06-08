import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import robotpy_apriltag as ap
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

######################################################################################################################
####################### Initialisation ###############################################################################
######################################################################################################################

K = np.array([ ## matrice des paramètre intrinsèques de la caméra
    [613.74650358, 0., 316.93428816],
    [0., 612.77437775, 235.68769803],
    [0., 0., 1.]
])

distorsion_coeffs = np.array([0.01067076, 0.03361432, -0.00278563, -0.00074779, -0.5557546]) ## coefficient de distortion utilisé pour corriger la déformation de l'image avant nos analyses

tag_size = 15.9 ## taille du tag en cm
tag_coordinates = { ##coordonnée des 4 coins des tags dans leur réferenciel
    3 : np.array([[0, 11.1, 0], [0 + tag_size, 11.1, 0], [0 + tag_size, 11.1 + tag_size, 0], [0, 11.1+ tag_size, 0]]),
    0 : np.array([[0, 0, 10.2], [0 + tag_size, 0, 20.2], [0 + tag_size, 0, 20.2 - tag_size], [0, 0, 20.2 - tag_size]]),
}

detector = ap.AprilTagDetector() ## on initialise le détecteur de AprilTag
current_params = detector.getQuadThresholdParameters()
current_params.minClusterPixels = 5 # pour faciliter la détection des tags
current_params.minWhiteBlackDiff = 3 # pour faciliter la détection des tags
detector.setQuadThresholdParameters(current_params)
detector.addFamily("tag36h11") ##on décide d'utiliser la "famille" de Tag "36h11"

######################################################################################################################
####################### partie des fonctions utilisé dans le programme ###############################################
######################################################################################################################

## fonction qui renvoie les coordonnées en pixel des points detectées du AprilTag sur l'image et les points 3D des coins dans le repère considéré défini plus haut (on va pas commenté spécifiquement la fonction puisque pas au coeur du sujet)
def retreive_coordinates(img):
    img_undistorted = cv2.undistort(img, K, distorsion_coeffs)
    img_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(img_gray)
    pts_image = []
    pts_world = []
    for tag in tags:
        tid = tag.getId()
        if tid not in tag_coordinates:
            continue
        corners_img = np.array(tag.getCorners((0,0,0,0,0,0,0,0))).reshape(4,2)
        corners_world = tag_coordinates[tid][:,:2]
        pts_image.extend(corners_img)
        pts_world.extend(corners_world)
        cv2.polylines(img_undistorted, [corners_img.astype(np.int32).reshape(-1,1,2)], True, (0,255,0), 2)
        center = np.mean(corners_img, axis=0).astype(int)
        cv2.putText(img_undistorted, f"ID:{tid}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    if len(pts_image) == 0:
        return None, None, img_undistorted
    return np.array(pts_image), np.hstack((np.array(pts_world), np.zeros((len(pts_world),1)))), img_undistorted

##fonction de calcul utile

def project_points(points_3d, R, t, K): ## cet algorithme calcul la projection des point 3D du monde sur le plan homogène de la caméra
    pts_cam = (R @ points_3d.T) + t.reshape(3,1) #on place le point 3D dans e repère propre de la caméra
    pts_proj = K @ pts_cam ##on calcule les coordonnées 2D à une constante près des coordonnées homogènes de l'image (plan de référence)
    pts_proj = pts_proj[:2,:] / pts_proj[2,:] ##on normalise sur dernière coordonnée pour se ramener à des coordonnées homogènes (référence)
    return pts_proj.T

def sensibility_matrix(param, K, pts_3d): ## cette fonction créer notre matrice jacobienne, elle n'est pas trop lourde alors on l'integre dans le programme lui même
    sens_matrix = []
    R_flat = param[:9] ## on construit R et t directement avec nos paramètres
    R = R_flat.reshape(3, 3)
    t = param[9:]
    fx, fy = K[0, 0], K[1, 1] # Focal lengths

    for pt_3d in pts_3d: ##on itère pour chaque points
        X, Y, Z = pt_3d
        u_w = np.array([[X], [Y], [Z]])
        u_c = (R @ u_w) + t.reshape(3, 1)
        X_c, Y_c, Z_c = u_c[0, 0], u_c[1, 0], u_c[2, 0] # par commodité de calcul, on saisi ces grandeurs remarquables dans notre calcul des fonctions de sensibilité

        row_umx = np.zeros(12) ##on part de termes nul car un certain nombre le seront
        row_umy = np.zeros(12)

        ## ligne des fonctions de sensibilité de umx
        row_umx[0] = fx * X / Z_c ##calcul d'après l'annexe du rapport
        row_umx[1] = fx * Y / Z_c
        row_umx[2] = fx * Z / Z_c
        row_umx[6] = -fx * X_c * X / (Z_c**2) ## on saute les fonctions de sensibilité nulle du au np.zeros plus haut
        row_umx[7] = -fx * X_c * Y / (Z_c**2)
        row_umx[8] = -fx * X_c * Z / (Z_c**2)
        row_umx[9] = fx / Z_c
        row_umx[10] = 0
        row_umx[11] = -fx * X_c / (Z_c**2)

        ## ligne des fonctions de sensibilité de umy
        row_umy[3] = fy * X / Z_c
        row_umy[4] = fy * Y / Z_c
        row_umy[5] = fy * Z / Z_c
        row_umy[6] = -fy * Y_c * X / (Z_c**2)
        row_umy[7] = -fy * Y_c * Y / (Z_c**2)
        row_umy[8] = -fy * Y_c * Z / (Z_c**2)
        row_umy[9] = 0
        row_umy[10] = fy / Z_c
        row_umy[11] = -fy * Y_c / (Z_c**2)

        sens_matrix.append(row_umx) ##on concatène nos deux lignes
        sens_matrix.append(row_umy)
    return np.array(sens_matrix)

##partie descente optimisation paramétrique

def cost(pts_proj, pts_2d): ##la fonction cout qui est un simple moindre carré sur les points de l'image en 2D et les points projetés sur l'image d'après nos paramètres
    return np.sum((pts_2d - pts_proj)**2)

def gradient_sens(params, K, pts_proj, pts_2d, pts_3d): ## cette fonction renvoie le gradient calculé à partir des fonctions de sensibilités selon la relation du rapport
    sens_matrix = sensibility_matrix(params, K, pts_3d) ##on récupère la matrice jacobienne des fonctions de sensibilité
    grad = 2 * (pts_proj.reshape(1, -1)[0] - pts_2d.reshape(1, -1)[0]) @ sens_matrix
    grad=grad/np.linalg.norm(grad) #on normalise le gradient pour garder que l'information de la direction. cela peut limiter les calculs du lambda optimisé
    return grad

def optimize_lamb(params, pts_2d, pts_3d, current_cost, current_lamb, direction):## on optimise le pas pour limiter les itérations nécessaire (couteux en calcul à cause de la matrice de rotation notament, beaucoup recalculé)
    while True: ##on contraint le lambda à être le plus grand possible (plus grand avancement en norme quand on va dans la bonne direction)
        test_params = params - current_lamb * direction ## on test avec des nouveaux paramètres
        R_approx = test_params[:9].reshape(3, 3)
        U, S, Vt = np.linalg.svd(R_approx) ## on calcul la SVD pour compute la matrice de rotation proche de celle définie par nos actuels coefficients
        R_test = U @ Vt
        test_params[:9] = R_test.flatten()
        t_test = test_params[9:]
        test_proj = project_points(pts_3d, R_test, t_test, K)
        if cost(test_proj, pts_2d) < current_cost :
            current_lamb *= 1.5
        else:
            break
    while True: ## on contraint le fait de réduire la fonction cout
        test_params = params - current_lamb * direction
        R_approx = test_params[:9].reshape(3, 3) ## on test avec des nouveaux paramètres
        U, S, Vt = np.linalg.svd(R_approx) ## on calcul la SVD pour compute la matrice de rotation proche de celle définie par nos actuels coefficients
        R_test = U @ Vt
        test_params[:9] = R_test.flatten()
        t_test = test_params[9:]
        test_proj = project_points(pts_3d, R_test, t_test, K)
        if cost(test_proj, pts_2d) > current_cost and current_lamb<10e-12 :
            current_lamb *= 0.5
        else:
            break
    return current_lamb ##renvoie le lambda optimisé

######################################################################################################################
####################### contenu de la boucle principale du programme #################################################
######################################################################################################################

params = np.array([
    1, 0, 0,   # première ligne de la matrice (identité)
    0, 0, -1,   # deuxième ligne
    0, 0, 0,   # troisième ligne
    10, 10, 50     # translation (tx, ty, tz)
], dtype=float) # on impose des paramètres initiaux qui correspondent à peu près à la situation qu'on effectue (April tag face caméra à une certaine distance sur z)


def calc_gradient_sensibilite_12(img, params):
    lamb = 1.0 ##pas initial pour le gradient
    max_iter = 300000 ##nombre maximal d'iterations de la descente de gradient
    cost_evo_min = 1e-3 ##condition d'arrêt par stagnation sur la fonction cout
    iter_checkpoint = 500 ## nombre d'itérations avant de faire un point sur diverses informations de la déscente de gradient (fonction cout, stagnation)
    params = np.array([
        1, 0, 0,   # première ligne de la matrice (identité)
        0, 0, -1,   # deuxième ligne
        0, 0, 0,   # troisième ligne
        10, 10, 50     # translation (tx, ty, tz)
    ], dtype=float) # on impose des paramètres initiaux qui correspondent à peu près à la situation qu'on effectue (April tag face caméra à une certaine distance sur z)

    cost_history = [] #stockage des valeurs de la fonction cout

    pts_2d, pts_3d, img_out = retreive_coordinates(img) ## récupération des données d'entrée
    if pts_2d is None or pts_3d is None or pts_2d.shape[0] < 4: ## on test la bonne detection du april tag, c'est à dire ses 4 coins
        cv2_imshow(img_out)
        cv2.waitKey(1) 
        return None, None, None, None, None, params

    R = params[:9].reshape(3, 3) ## initialisation avec les paramètres de départ
    t = params[9:]
    pts_proj = project_points(pts_3d, R, t, K)
    
    old_c, c = 0, cost(pts_proj, pts_2d) 
    i = 0
    while i < max_iter and (i < 2 * iter_checkpoint or abs(1 - (old_c / c)) > cost_evo_min): ## condition d'arrêt : nombre d'itérations maximal et stagnation
        R = params[:9].reshape(3, 3) ##on estime à chaque itération la projection des points 3D selon nos paramètres actuels
        t = params[9:]
        pts_proj = project_points(pts_3d, R, t, K)
        base_cost = cost(pts_proj, pts_2d)  ##clacul de la fonction cout
        g = gradient_sens(params, K, pts_proj, pts_2d, pts_3d) ##caclul du gradient
        lamb = optimize_lamb(params, pts_2d, pts_3d, base_cost, lamb, g)  ##calcul du lambda optimisé
        params -= lamb * g ##mise à jour des paramètres

        R_approx = params[:9].reshape(3, 3) ## on corrige la matrice de rotation obtenue via la méthode de la SVD
        U, S, Vt = np.linalg.svd(R_approx)
        R_orthogonal = U @ Vt # on compute une matrice de rotation proche de la matrice obtenue avec nos parametres
        params[:9] = R_orthogonal.flatten()

        if i % iter_checkpoint == 0:
            old_c=c
            c = cost(pts_proj, pts_2d)
            cost_history.append(c)
            print(f"Iter {i}: cost={c:.4f}, cost_evo={abs(1 - (old_c / c)) :.6f}, lambda={lamb:.4f}")

        i += 1

    T = params[:9].reshape(3, 3) ##on estime avec les paramètres finaux
    t = params[9:]

    # Affichage des résultats
    ang, _ = cv2.Rodrigues(R)  # permet de convertir les angles dans l'intervalle [-pi, pi] pour contraindre nos angles à un voisinage permettant l'identification du système
    angles_deg = np.degrees(ang.flatten()) #conversion en degrés
    print(f"Translation: {t}") ##plot des valeurs obtenues au final sur l'image du test
    print(f"Rotation (deg): gamma={angles_deg[0]:.2f}, beta={angles_deg[1]:.2f}, alpha={angles_deg[2]:.2f}")

    cv2.putText(img_out, f"t = [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_out, f"gamma: {angles_deg[0]:.1f} deg", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_out, f"beta: {angles_deg[1]:.1f} deg", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_out, f"alpha: {angles_deg[2]:.1f} deg", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    pts_proj = project_points(pts_3d, R, t, K) ##on projete avec nos parametre finaux pour plot des points jaune là ou notre descente de gradient l'estime (control visuel)
    for pt_proj in pts_proj:
        cv2.circle(img_out, (int(pt_proj[0]), int(pt_proj[1])), 5, (0, 255, 255), -1)
    cv2_imshow(img_out)

    return R, angles_deg, t, cost_history, iter_checkpoint, params


######################################################################################################################
####################### appel de la boucle principale du programme ###################################################
######################################################################################################################

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) ##on paramètre la webcam comme périphérique de capture d'image
    while True: ##boucle principale
        ret, img = cap.read()
        if ret:
            _, _, _, _, _, params = calc_gradient_sensibilite_12(img, params)
        cv2.waitKey(1)
    cap.release() ## fermer les fenêtres à la fin du programme
    cv2.destroyAllWindows()
