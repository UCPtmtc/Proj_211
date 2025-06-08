import numpy as np
import cv2
import robotpy_apriltag as ap
#import gradient_descente as g
import fonction_sensibilite_6 as fs

######################################################################################################################
####################### Initialisation ###############################################################################
######################################################################################################################

K = np.array([ ## matrice des paramètre intrinsèque de la caméra
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
current_params.minClusterPixels = 8 # pour faciliter la détection des tags
current_params.minWhiteBlackDiff = 4 # pour faciliter la détection des tags
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

def euler_to_R(gamma, beta, alpha): ## cette fonction prend en argument les 3 angles de Tait-Bryan et renvoie la matrice de rotation correspondante
    cos_gamma, sin_gamma, cos_beta, sin_beta, cos_alpha, sin_alpha =np.cos(gamma), np.sin(gamma), np.cos(beta), np.sin(beta), np.cos(alpha), np.sin(alpha)
    Rx = np.array([[1,0,0],
                   [0,cos_gamma, -sin_gamma], ## on stock les valeurs trigonometriques pour optimiser les calculs (éviter de repasser par les fonctions numpy inutilement)
                   [0,sin_gamma, cos_gamma]])
    Ry = np.array([[cos_beta,0,sin_beta],
                   [0,1,0],
                   [-sin_beta,0,cos_beta]])
    Rz = np.array([[cos_alpha, -sin_alpha,0],
                   [sin_alpha, cos_alpha,0],
                   [0,0,1]])
    return Rz @ Ry @ Rx

def project_points(points_3d, R, t, K): ## cet algorithme calcul la projection des point 3D du monde sur le plan homogène de la caméra
    pts_cam = (R @ points_3d.T) + t.reshape(3,1) #on place le point 3D dans e repère propre de la caméra
    pts_proj = K @ pts_cam ##on calcule les coordonnées 2D à une constante près des coordonnées homogènes de l'image (plan de référence)
    pts_proj = pts_proj[:2,:] / pts_proj[2,:] ##on normalise sur dernière coordonnée pour se ramener à des coordonnées homogènes (référence)
    return pts_proj.T

##partie descente optimisation paramétrique

def cost(pts_proj, pts_2d): ##la fonction cout qui est un simple moindre carré sur les points de l'image en 2D et les points projetés sur l'image d'après nos paramètres
    return np.sum((pts_2d - pts_proj)**2)

def gradient_sens(params, K, pts_proj, pts_2d, pts_3d): ## renvoie le gradient calculé à partir des fonctions de sensibilité
    sens_matrix=fs.sensibility_matrix(params, K, pts_3d) ##on récupère la matrice jacobienne des fonctions de sensibilité
    pts_2d=pts_2d.reshape(1,-1)[0] ## on passe les point du plan image en vecteur ligne pour simplifier le calcul avec la jacobienne d'après la formule de compte rendu
    pts_proj=pts_proj.reshape(1,-1)[0]
    grad=2*(pts_proj-pts_2d)@sens_matrix #formule du compte rendu
    grad=grad/np.linalg.norm(grad) #on normalise le gradient pour garder que l'information de la direction. cela peut limiter les calcul du lambda optimisé
    return grad

def optimize_lamb(params ,pts_2d,pts_3d, current_cost, current_lamb, direction): ## on optimise le pas pour limiter les itérations nécessaire (couteux en calcul à cause de la matrice de rotation notament, beaucoup recalculé)
    while True : ##on contraint le lambda à être le plus grand possible (plus grand avancement en norme quand on va dans la bonne direction)
        current_lamb=2*current_lamb
        params_test = params - current_lamb * direction
        gamma, beta, alpha, tx, ty, tz = params_test
        R = euler_to_R(gamma, beta, alpha)
        t = np.array([tx, ty, tz])
        pts_proj = project_points(pts_3d, R, t, K)
        if cost(pts_proj, pts_2d) < current_cost :
            current_lamb *= 1.5
        else :
            break
    while True : ## on contraint le fait de réduire la fonction cout
        current_lamb = (1/2) * current_lamb
        params_test = params - current_lamb * direction
        gamma, beta, alpha, tx, ty, tz = params_test
        R = euler_to_R(gamma, beta, alpha)
        t = np.array([tx, ty, tz])
        pts_proj = project_points(pts_3d, R, t, K)
        if cost(pts_proj, pts_2d) > current_cost :
            current_lamb *= 0.5
        else :
            break
    return current_lamb ##renvoie le lambda optimisé

######################################################################################################################
####################### contenu de la boucle principale du programme #################################################
######################################################################################################################

params = np.array([
    180, 0.0, 0.0,   # gamma, beta, alpha
    10.0, 10.0, 50.0  # tx, ty, tz
], dtype=float)
 # on impose des paramètres initiaux qui correspondent à peu près à la situation qu'on effectue (April tag face caméra à une certaine distance sur z)

def calc_gradient_sensibilite_6(img, params):
    lamb = 1.0 ##pas initial pour le gradient
    max_iter = 300000 ##nombre maximal d'iterations de la descente de gradient
    cost_evo_min = 1e-3 ##condition d'arrêt par stagnation sur la fonction cout
    iter_checkpoint = 500 ## nombre d'itérations avant de faire un point sur diverses informations de la déscente de gradient (fonction cout, stagnation)
    params = np.array([
        180, 0.0, 0.0,   # gamma, beta, alpha
        10.0, 10.0, 50.0  # tx, ty, tz
    ], dtype=float)
    # on impose des paramètres initiaux qui correspondent à peu près à la situation qu'on effectue (April tag face caméra à une certaine distance sur z)

    cost_history = [] #stockage des valeurs de la fonction cout

    pts_2d, pts_3d, img_out = retreive_coordinates(img) ## récupération des données d'entrée
    if pts_2d is None or pts_3d is None or pts_2d.shape[0] < 4: ## on test la bonne detection du april tag, c'est à dire ses 4 coins
        cv2.imshow("Pose Estimation", img_out)
        cv2.waitKey(1) 
        return None, None, None, None, None, params

    gamma, beta, alpha, tx, ty, tz = params ## initialisation avec les paramètres de départ
    R = euler_to_R(gamma, beta, alpha)
    t = np.array([tx, ty, tz])
    pts_proj = project_points(pts_3d, R, t, K)

    old_c, c = 0, cost(pts_proj, pts_2d)
    i = 0
    while i < max_iter and (i < 2 * iter_checkpoint or abs(1 - (old_c / c)) > cost_evo_min): ## condition d'arrêt : nombre d'itérations maximal et stagnation
        gamma, beta, alpha, tx, ty, tz = params ##on estime à chaque itération la projection des points 3D selon nos paramètres actuels
        R = euler_to_R(gamma, beta, alpha)
        t = np.array([tx, ty, tz])
        pts_proj = project_points(pts_3d, R, t, K)
        base_cost = cost(pts_proj, pts_2d) ##clacul de la fonction cout
        g = gradient_sens(params, K, pts_proj, pts_2d, pts_3d) ##caclul du gradient
        lamb=optimize_lamb(params ,pts_2d,pts_3d, base_cost, lamb, g) ##calcul du lambda optimisé
        params -= lamb * g ##mise à jour des paramètres

        if i % iter_checkpoint == 0 and i>0: ## périodiquement on fait le point sur la fonction cout, l'évolution du cout et du pas
            old_c=c
            c = cost(pts_proj, pts_2d)
            cost_history.append(c)
            print(f"Iter {i}: cost={c:.4f} ; cost_evo_min={abs(1 - (old_c / c))}, lambda={lamb}")

        i += 1

    gamma, beta, alpha, tx, ty, tz = params ##on estime avec les paramètres finaux
    R = euler_to_R(gamma, beta, alpha)
    t = np.array([tx, ty, tz])

    # Affichage des résultats
    ang, _ = cv2.Rodrigues(R) # permet de convertir les angles dans l'intervalle [-pi, pi] pour contraindre nos angles à un voisinage permettant l'identification du système
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

    pts_proj = project_points(pts_3d, R, t, K) ##on projete avec nos parametres finaux pour plot des points jaune là ou notre descente de gradient l'estime (control visuel)
    for pt_proj in pts_proj:
        cv2.circle(img_out, (int(pt_proj[0]), int(pt_proj[1])), 5, (0, 255, 255), -1)

    cv2.imshow("Pose Estimation", img_out)
    
    return R, angles_deg, t, cost_history, iter_checkpoint, params

######################################################################################################################
####################### appel de la boucle principale du programme ###################################################
######################################################################################################################

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) ##on paramètre la webcam comme périphérique de capture d'image
    while True: ##boucle principale
        ret, img = cap.read()
        if ret:
            _, _, _, _, _, params = calc_gradient_sensibilite_6(img, params)
        cv2.waitKey(1)
    cap.release() ## fermer les fenêtres à la fin du programme
    cv2.destroyAllWindows()