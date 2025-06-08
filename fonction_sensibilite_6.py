import numpy as np

# Nous utiliserons la même structure pour l'optimisation, en pré-calculant les valeurs trigonométriques.
# La fonction sens_matrix organisera toujours les dérivées.

######################################################################################################################
####################### Fonctions initiales (à titre de référence, non utilisées directement dans le calcul de la matrice de sensibilité) #####
######################################################################################################################

## Fonction umx (coordonnée x projetée), non utilisée directement dans la matrice de sensibilité
def umx(fx, X, Y, Z, tx, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma, cx):
    numerateur_x = X * cos_beta * cos_alpha + \
                   Y * (sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha) + \
                   Z * (sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha) + tx
    return fx * (numerateur_x / denominateur) + cx

## Fonction umy (coordonnée y projetée), non utilisée directement dans la matrice de sensibilité
def umy(fy, X, Y, Z, ty, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma, cy):
    numerateur_y = X * cos_beta * sin_alpha + \
                   Y * (sin_gamma * sin_beta * sin_alpha + cos_gamma * cos_alpha) + \
                   Z * (cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha) + ty
    return fy * (numerateur_y / denominateur) + cy


######################################################################################################################
####################### Fonctions de sensibilité (Dérivées) ##########################################################
######################################################################################################################

# Termes communs pour la lisibilité :
# D = denominateur = -X * sin_beta + Y * sin_gamma * cos_beta + Z * cos_gamma * cos_beta + tz
# Nx = numerateur_x (de la fonction umx)
# Ny = numerateur_y (de la fonction umy)

# Fonctions de sensibilité pour umx (u_mc_x)
def d_umx_d_alpha(fx, X, Y, Z, denominateur, cos_beta, sin_gamma, sin_beta, sin_alpha, cos_gamma, cos_alpha):
    # Correspond à LaTeX : - (fx * (X sin(alpha) cos(beta) + ...)) / D
    numerateur_deriv_alpha = X * sin_alpha * cos_beta + \
                             Y * (sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma) + \
                             Z * (sin_alpha * sin_beta * cos_gamma - sin_gamma * cos_alpha)
    return -fx * numerateur_deriv_alpha / denominateur

def d_umx_d_beta(fx, X, Y, Z, tx, denominateur, sin_beta, cos_beta, sin_gamma, cos_gamma, sin_alpha, cos_alpha):
    # Numérateur de umx (Nx)
    num_umx = X * cos_beta * cos_alpha + \
              Y * (sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha) + \
              Z * (sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha) + tx

    # Dérivée du numérateur de umx par rapport à beta (dNx/d_beta)
    d_num_umx_d_beta_val = -X * sin_beta * cos_alpha + \
                           Y * (sin_gamma * cos_beta * cos_alpha) + \
                           Z * (cos_gamma * cos_beta * cos_alpha)

    # Dérivée du denominateur par rapport à beta (dD/d_beta)
    d_denominateur_d_beta_val = -X * cos_beta - Y * sin_gamma * sin_beta - Z * cos_gamma * sin_beta

    # Correspond à LaTeX : fx * ( (dNx/d_beta) * D - Nx * (dD/d_beta) ) / D^2
    return fx * (d_num_umx_d_beta_val * denominateur - num_umx * d_denominateur_d_beta_val) / denominateur**2

def d_umx_d_gamma(fx, X, Y, Z, tx, denominateur, sin_gamma, cos_gamma, sin_beta, cos_beta, sin_alpha, cos_alpha):
    # Numérateur de umx (Nx)
    num_umx = X * cos_beta * cos_alpha + \
              Y * (sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha) + \
              Z * (sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha) + tx

    # Dérivée du numérateur de umx par rapport à gamma (dNx/d_gamma)
    d_num_umx_d_gamma_val = Y * (cos_gamma * sin_beta * cos_alpha + sin_gamma * sin_alpha) + \
                            Z * (cos_gamma * sin_alpha - sin_gamma * sin_beta * cos_alpha)

    # Dérivée du denominateur par rapport à gamma (dD/d_gamma)
    d_denominateur_d_gamma_val = Y * cos_gamma * cos_beta - Z * sin_gamma * cos_beta

    # Correspond à LaTeX : fx * ( (dNx/d_gamma) * D - Nx * (dD/d_gamma) ) / D^2
    return fx * (d_num_umx_d_gamma_val * denominateur - num_umx * d_denominateur_d_gamma_val) / denominateur**2

def d_umx_d_tx(fx, denominateur):
    # Correspond à LaTeX : fx / D
    return fx / denominateur

def d_umx_d_ty():
    # Correspond à LaTeX : 0
    return 0

def d_umx_d_tz(fx, X, Y, Z, tx, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma):
    # Numérateur de umx (Nx)
    num_umx = X * cos_beta * cos_alpha + \
              Y * (sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha) + \
              Z * (sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha) + tx
    # Correspond à LaTeX : - fx * Nx / D^2
    return -fx * num_umx / denominateur**2


# Fonctions de sensibilité pour umy (u_mc_y)
def d_umy_d_alpha(fy, X, Y, Z, denominateur, cos_beta, sin_gamma, sin_beta, sin_alpha, cos_gamma, cos_alpha):
    # Correspond à LaTeX : fy * (X cos(alpha) cos(beta) + ...) / D
    numerateur_deriv_alpha = X * cos_alpha * cos_beta + \
                             Y * (sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha) + \
                             Z * (cos_gamma * sin_beta * cos_alpha + sin_gamma * sin_alpha) # Ce terme peut avoir un signe différent selon la version précédente de Python, maintenant correspond à LaTeX
    return fy * numerateur_deriv_alpha / denominateur

def d_umy_d_beta(fy, X, Y, Z, ty, denominateur, sin_beta, cos_beta, sin_gamma, cos_gamma, sin_alpha, cos_alpha):
    # Numérateur de umy (Ny)
    num_umy = X * cos_beta * sin_alpha + \
              Y * (sin_gamma * sin_beta * sin_alpha + cos_gamma * cos_alpha) + \
              Z * (cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha) + ty

    # Dérivée du numérateur de umy par rapport à beta (dNy/d_beta)
    d_num_umy_d_beta_val = -X * sin_beta * sin_alpha + \
                           Y * (sin_gamma * cos_beta * sin_alpha) + \
                           Z * (cos_gamma * cos_beta * sin_alpha)

    # Dérivée du denominateur par rapport à beta (dD/d_beta)
    d_denominateur_d_beta_val = -X * cos_beta - Y * sin_gamma * sin_beta - Z * cos_gamma * sin_beta

    # Correspond à LaTeX : fy * ( (dNy/d_beta) * D - Ny * (dD/d_beta) ) / D^2
    return fy * (d_num_umy_d_beta_val * denominateur - num_umy * d_denominateur_d_beta_val) / denominateur**2

def d_umy_d_gamma(fy, X, Y, Z, ty, denominateur, sin_gamma, cos_gamma, sin_beta, cos_beta, sin_alpha, cos_alpha):
    # Numérateur de umy (Ny)
    num_umy = X * cos_beta * sin_alpha + \
              Y * (sin_gamma * sin_beta * sin_alpha + cos_gamma * cos_alpha) + \
              Z * (cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha) + ty

    # Dérivée du numérateur de umy par rapport à gamma (dNy/d_gamma)
    d_num_umy_d_gamma_val = Y * (cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha) + \
                            Z * (-sin_gamma * sin_beta * sin_alpha - cos_gamma * cos_alpha) # Ce terme peut avoir un signe différent, maintenant correspond à LaTeX

    # Dérivée du denominateur par rapport à gamma (dD/d_gamma)
    d_denominateur_d_gamma_val = Y * cos_gamma * cos_beta - Z * sin_gamma * cos_beta

    # Correspond à LaTeX : fy * ( (dNy/d_gamma) * D - Ny * (dD/d_gamma) ) / D^2
    return fy * (d_num_umy_d_gamma_val * denominateur - num_umy * d_denominateur_d_gamma_val) / denominateur**2

def d_umy_d_tx():
    # Correspond à LaTeX : 0
    return 0

def d_umy_d_ty(fy, denominateur):
    # Correspond à LaTeX : fy / D
    return fy / denominateur

def d_umy_d_tz(fy, X, Y, Z, ty, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma):
    # Numérateur de umy (Ny)
    num_umy = X * cos_beta * sin_alpha + \
              Y * (sin_gamma * sin_beta * sin_alpha + cos_gamma * cos_alpha) + \
              Z * (cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha) + ty
    # Correspond à LaTeX : - fy * Ny / D^2
    return -fy * num_umy / denominateur**2

# ---

def sensibility_matrix(param, K, pts_3d): ## Nous créons une matrice Jacobienne pour faciliter le calcul
    sens_matrix = []
    gamma, beta, alpha, tx, ty, tz = param ## Nous récupérons les paramètres nécessaires au calcul
    fx, fy = K[0, 0], K[1, 1]

    # Nous pré-calculons une fois les fonctions trigonométriques pour optimiser
    cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)
    cos_beta, sin_beta = np.cos(beta), np.sin(beta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)

    for pt_3d in pts_3d: ## Parcourir tous les points
        X, Y, Z = pt_3d

        # Calculer une seule fois le dénominateur commun pour optimiser
        denominateur = -X * sin_beta + Y * sin_gamma * cos_beta + Z * cos_gamma * cos_beta + tz

        # Calculer les fonctions de sensibilité pour umx sur la première ligne (pour le point courant)
        sens_matrix.append(np.array([
            d_umx_d_gamma(fx, X, Y, Z, tx, denominateur, sin_gamma, cos_gamma, sin_beta, cos_beta, sin_alpha, cos_alpha),
            d_umx_d_beta(fx, X, Y, Z, tx, denominateur, sin_beta, cos_beta, sin_gamma, cos_gamma, sin_alpha, cos_alpha),
            d_umx_d_alpha(fx, X, Y, Z, denominateur, cos_beta, sin_gamma, sin_beta, sin_alpha, cos_gamma, cos_alpha),
            d_umx_d_tx(fx, denominateur),
            d_umx_d_ty(),
            d_umx_d_tz(fx, X, Y, Z, tx, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma)
        ]))

        # Calculer les fonctions de sensibilité pour umy sur la deuxième ligne (pour le point courant)
        sens_matrix.append(np.array([
            d_umy_d_gamma(fy, X, Y, Z, ty, denominateur, sin_gamma, cos_gamma, sin_beta, cos_beta, sin_alpha, cos_alpha),
            d_umy_d_beta(fy, X, Y, Z, ty, denominateur, sin_beta, cos_beta, sin_gamma, cos_gamma, sin_alpha, cos_alpha),
            d_umy_d_alpha(fy, X, Y, Z, denominateur, cos_beta, sin_gamma, sin_beta, sin_alpha, cos_gamma, cos_alpha),
            d_umy_d_tx(),
            d_umy_d_ty(fy, denominateur),
            d_umy_d_tz(fy, X, Y, Z, ty, denominateur, cos_beta, cos_alpha, sin_gamma, sin_beta, sin_alpha, cos_gamma)
        ]))
    return np.array(sens_matrix)
