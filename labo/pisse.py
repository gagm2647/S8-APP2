"""
Fonctions utiles pour le traitement de données
APP2 S8 GIA
Classe disponible:
    Extent: bornes ou plage utile de données
Fonctions disponibles:
    calcModeleGaussien: calcule les stats de base d'une série de données
    viewEllipse: ajoute une ellipse à 1 sigma sur un graphique
    view_classes: affiche sur un graphique 2D les points de plusieurs classes
    creer_hist2D: crée la densité de probabilité d'une série de points 2D
"""

from cgitb import small
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm


class Extent:
    """
    classe pour regrouper les min et max de données 2D
    """
    def __init__(self, xmin=0, xmax=10, ymin=0, ymax=10, array=None):
        """
        Constructeur
        2 options:
            passer 4 arguments min et max
            passer 1 array qui contient les 4 mêmes quantités
        """
        if array is not None:
            self.xmin = array[0][0]
            self.xmax = array[0][1]
            self.ymin = array[1][0]
            self.ymax = array[1][1]
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax
        # workaround pour les projections sur des vecteurs arbitraires
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin

    def get_array(self):
        """
        Accesseur qui retourne sous format matriciel
        """
        return [[self.xmin, self.xmax], [self.ymin, self.ymax]]


def calcModeleGaussien(data, message=''):
    """
    Calcule les stats de base de données
    :param data: les données à traiter, devrait contenir 1 point N-D par ligne
    :param message: si présent, génère un affichage des stats calculées
    :return: la moyenne, la matrice de covariance, les valeurs propres et les vecteurs propres de "data"
    """
    # TODO L1.E2.2 Remplacer les valeurs bidons avec les fonctions appropriées ici

    moyenne = np.round([np.mean(data[:,0]), np.mean(data[:,1])])
    matr_cov = np.round(np.cov(data[:,0], data[:,1]))
    val_propres, vect_propres = np.linalg.eigh(matr_cov)
    
    if message:
        print(message)
        print(f'Moy:\n {moyenne} \nCov:\n {matr_cov} \nVal prop:\n {val_propres} \nVect prop:\n {vect_propres}')
    return moyenne, matr_cov, val_propres, vect_propres


def viewEigen(center, values, vectors):
     # Largest Value/Vector
    largest_value = max(values)
    largest_vector = vectors[np.argmax(values)]
    # Smallest Vector
    smallest_value = min(values)
    smallest_vector = vectors[np.argmin(values)]
    quiveropts = dict(headaxislength=0, color='red', headlength=0, units='xy',angles='xy',scale=1)
    plt.quiver(center[0], center[1], largest_vector[0]*np.sqrt(largest_value), largest_vector[1]*np.sqrt(largest_value), **quiveropts)
    plt.quiver(center[0], center[1], smallest_vector[0]*np.sqrt(smallest_value), smallest_vector[1]*np.sqrt(smallest_value), **quiveropts)


def viewEllipse(data, ax, scale=1, facecolor='none', edgecolor='red', **kwargs):
    """
    ***Testé seulement sur les données du labo
    Ajoute une ellipse à distance 1 sigma du centre d'une classe
    Inspiration de la documentation de matplotlib 'Plot a confidence ellipse'

    data: données de la classe, les lignes sont des données 2D
    ax: axe des figures matplotlib où ajouter l'ellipse
    scale: Facteur d'échelle de l'ellipse, peut être utilisé comme paramètre pour tracer des ellipses à une
        équiprobabilité différente, 1 = 1 sigma
    facecolor, edgecolor, and kwargs: Arguments pour la fonction plot de matplotlib

    retourne l'objet Ellipse créé
    """
    moy, cov, lambdas, vectors = calcModeleGaussien(data)
    
    # center of ellipsis
    center = (moy[0], moy[1])
    
     # Largest Value/Vector
    largest_value = max(lambdas)
    largest_vector = vectors[:,np.argmax(lambdas)]
    # Smallest Vector
    smallest_value = min(lambdas)
    smallest_vector = vectors[:,np.argmin(lambdas)]
    quiveropts = dict(headaxislength=0, color=edgecolor, headlength=0, units='xy',angles='xy',scale=1)
    plt.quiver(center[0], center[1], largest_vector[0]*np.sqrt(largest_value), largest_vector[1]*np.sqrt(largest_value), **quiveropts)
    plt.quiver(center[0], center[1], smallest_vector[0]*np.sqrt(smallest_value), smallest_vector[1]*np.sqrt(smallest_value), **quiveropts)


    # scale 95%
    chisquare_val = 2.4477
    scale_x = np.sqrt(cov[0, 0] * chisquare_val)
    scale_y = np.sqrt(cov[1, 1] * chisquare_val)

    # size x - y
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson) #* scale_x
    ell_radius_y = np.sqrt(1 - pearson) #* scale_y

    # angle of largest vector
    rad = np.arctan(largest_vector[1]/largest_vector[0])
    angle = np.degrees(rad) 

    # TODO L2.E1.2 Remplacer les valeurs par les bons paramètres à partir des stats ici
    ellipse = Ellipse(center, width=ell_radius_x * 2, height=ell_radius_y * 2,
                      angle=angle, facecolor=facecolor,
                      edgecolor=edgecolor, linewidth=2, **kwargs)
    return ax.add_patch(ellipse)


def view_classes(data, extent):
    """
    Affichage des classes dans data
    *** Fonctionne pour des classes 2D

    data: tableau des classes à afficher. La première dimension devrait être égale au nombre de classes.
    extent: bornes du graphique
    """
    #  TODO: rendre général, seulement 2D pour l'instant
    dims = np.asarray(data).shape

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title(r'Visualisation des classes et des ellipses à distance 1$\sigma$')

    colorpoints = ['orange', 'purple', 'black']
    colorfeatures = ['red', 'green', 'blue']

    for i in range(dims[0]):
        tempdata = data[i]
        m, cov, valpr, vectprop = calcModeleGaussien(tempdata)
        ax1.scatter(tempdata[:, 0], tempdata[:, 1], s=5, c=colorpoints[i])
        ax1.scatter(m[0], m[1], c=colorfeatures[i])
        viewEllipse(tempdata, ax1, edgecolor=colorfeatures[i])

    ax1.set_xlim([extent.xmin, extent.xmax])
    ax1.set_ylim([extent.ymin, extent.ymax])

    ax1.axes.set_aspect('equal')


def decorrelate(data, basis):
    """
    Permet de projeter des données sur une base pour les décorréler
    :param data: classes à décorréler, la dimension 0 est le nombre de classes
    :param basis: les vecteurs propres sur lesquels projeter les données
    :return: les données projetées
    """
    dims = np.asarray(data).shape
    decorrelated = np.zeros(np.asarray(data).shape)
    for i in range(dims[0]):
        # TODO L1.E2.5 Remplacer l'opération bidon par la bonne projection ici
        tempdata = np.matmul(basis, data[i].T)
        decorrelated[i] = tempdata.T
    return decorrelated


def creer_hist2D(data, title, nbin=15):
    """
    Crée une densité de probabilité pour une classe 2D au moyen d'un histogramme
    data: liste des points de la classe, 1 point par ligne (dimension 0)

    retourne un array 2D correspondant à l'histogramme
    """

    x = np.array(data[:, 0])
    y = np.array(data[:, 1])

    # TODO L2.E1.1 Faire du pseudocode et implémenter une segmentation en bins...
    # pas des bins de l'histogramme
    delta_x = 1
    delta_y = 1
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    axe_x = np.linspace(min_x, max_x)
    axe_y = np.linspace(min_y, max_y)

    # TODO : remplacer les valeurs bidons par la bonne logique ici
    hist, xedges, yedges = np.histogram2d(axe_x, axe_y, bins=[delta_x, delta_y])
    # normalise par la somme (somme de densité de prob = 1)
    histsum = np.sum(hist)
    hist = hist / histsum

    # affichage, commenter l'entièreté de ce qui suit si non désiré
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(f'Densité de probabilité de {title}')

    # calcule des bords des bins
    xpos, ypos = np.meshgrid(xedges[:-1] + delta_x / 2, yedges[:-1] + delta_y / 2, indexing="ij")
    dz = hist.ravel()

    # list of colors
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos.ravel(), ypos.ravel(), 0, deltax * .9, deltay * .9, dz, color=rgba)
    # Fin "à commenter" si affichage non désiré

    return hist, xedges, yedges
