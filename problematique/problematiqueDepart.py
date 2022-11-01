"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
import random

from operator import contains

from ImageCollection import ImageCollection
import image_anal
import os
import cv2 as cv
import copy

import classifiers
import analysis as an

def start_main():
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E3 et problématique
    N = 5
    im_list = np.sort(random.sample(
        range(np.size(ImageCollection.image_list, 0)), N))
    print(im_list)
    ImageCollection.images_display(im_list)
    ImageCollection.view_histogrammes(im_list)
    plt.show()


#######################################
#   Image normalization
#######################################
#  This function normalizes the color intensities by dividing each color
#  value by the sum of all color values. In the case of an RGB image
#  R = R/(R+G+B), G = G/(R+G+B), B = B/(R+G+B)
def normalize_intensity(img: np.array):
    img = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R_G_B = np.sum(img[i][j])
            img[i][j] = img[i][j] / R_G_B if R_G_B > 0 else np.zeros(3)
    return img


# This function normalizes for the color spectrum of the illuminant of the
# scene. Suppose again that we have an RGB image. If (r1,g1,b1) and
# (r2,g2,b2) are the responses of two points in the image under one colour
# of light then (k*r1,l*g1,m*b1) and (k*r2,l*g2,m*b2) are the responses of
# these same point under a different color of light. It can be easily shown
# that we can cancel the factors (k, l and m):
#
# ( 2r1/(r1+r2), 2g1/(r1+r2), 2b1/(r1+r2) ) and
#
# ( 2r2/(r1+r2), 2g2/(r1+r2), 2b2/(r1+r2) )
def normalize_illumination(img: np.array):
    a = img
    img = copy.deepcopy(img)
    N = img.shape[0]
    for color in range(img.shape[2]):
        intensity = 3/(N*N) * np.sum(img[:, :, color])
        img[:, :, color] = img[:, :, color] / intensity
    

    return img

# This algorithm is based on the folowing paper:
# ------
# Finlayson, G., Schiele, B., & Crowley, J. (1998).
# Comprehensive Colour Image Normalization.
# Computer Vision�ECCV�98, 1406, 475�490.
# https://doi.org/10.1007/BFb0055655
# ------
#  Script written by (Niels) N.W. Schurink, 23-11-2016, master 3 student
#  Technical Medicine, University of Twente, the Netherlands, during
#  master thesis at Netherlands Cancer Institute - Antoni van Leeuwenhoek
#  Ported from MATLAB to PYTHON by Charles-Etienne Granger.


def comprehensive_color_normalize(img: np.array, display: bool = False):
    difference = 1
    threshold = 10**-19
    prev_img = copy.deepcopy(img)
    prev_img = prev_img.astype('float64')
    while difference > threshold:
        # normalize intensity
        next_img = normalize_intensity(prev_img)

        # normalize the illumination
        next_img = normalize_illumination(next_img)

        # Calcul des diff (# ça pue python)
        prev_square, next_square = np.square(prev_img), np.square(next_img)
        prev_sum, next_sum = np.sum(prev_square), np.sum(next_square)
        prev_sqrt, next_sqrt = np.sqrt(prev_sum), np.sqrt(next_sum)
        difference = prev_sqrt - next_sqrt

        prev_img = copy.deepcopy(next_img)

    if display:
        cv.imshow('img', img)
        cv.imshow('comprehensive normalized_img', prev_img)
        cv.waitKey()
        cv.destroyAllWindows()

    return prev_img

def normalize_image(img: np.array, display: bool = False):
    mu = np.mean(img)
    sigma = np.std(img)
    n_img = (img - mu) / sigma if sigma > 1 else img

    if display:
        cv.imshow('img', img)
        cv.imshow('normalized_img', n_img)
        cv.waitKey()
        cv.destroyAllWindows()

    return [n_img, mu, sigma]

#######################################
#   Image Loading
#######################################


def load_images(directory, size=256, normalize=False, random=False):
    filenames = os.listdir(directory)
    if random:
        classes = ['coast', 'forest', 'street']
        nb_images = len(classes)
        images = np.zeros((nb_images, size, size, 3))
        labels = np.zeros(nb_images)
        filesizes = np.zeros(nb_images)
        for idx in range(len(classes)):
            filename = np.random.choice(
                [file for file in filenames if classes[idx] in file])
            filesizes[idx] = (os.stat(os.path.join(
                directory, filename)).st_size)  # Bytes
            img = cv.imread(os.path.join(directory, filename))
            if img is not None:
                print('load img number:', idx)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if normalize:
                    # Normalize (img - mean / sigma)
                    # images[idx], mu, sigma = normalize_image(img, display=False)
                    # Comprehensive color normalize
                    images[idx] = comprehensive_color_normalize(img, display=False)
                else:
                    images[idx] = img

                # Labeling
                if contains(filename, 'coast'):
                    labels[idx] = 0
                elif contains(filename, 'forest'):
                    labels[idx] = 1
                elif contains(filename, 'street'):
                    labels[idx] = 2
    else:
        nb_images = len(filenames)
        images = np.zeros((nb_images, size, size, 3))
        labels = np.zeros(nb_images)
        filesizes = np.zeros(nb_images)
        for idx, filename in enumerate(filenames):
            filesizes[idx] = (os.stat(os.path.join(
                directory, filename)).st_size)  # Bytes
            img = cv.imread(os.path.join(directory, filename))
            if img is not None:
                print('load img number:', idx)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                #img = img.astype('uint8')
                if normalize:
                    # Normalize (img - mean / sigma)
                     images[idx], mu, sigma = normalize_image(img, display=False)
                    # Comprehensive color normalize
                    #images[idx] = comprehensive_color_normalize(img, display=False)
                else:
                    images[idx] = img

                # Smoll dataset.
                # p = os.path.join('./problematique/test/small', filename)
                # s = skimage.transform.resize(img, (32,32))
                # skimage.io.imsave(p, s)

                # Labeling
                if contains(filename, 'coast'):
                    labels[idx] = 0
                elif contains(filename, 'forest'):
                    labels[idx] = 1
                elif contains(filename, 'street'):
                    labels[idx] = 2

    return images, labels, filesizes


def main():
    # Data load
    dataset_path = './problematique/baseDeDonneesImages'
    dataset_short_path = "./problematique/test/"
    dataset_32x32_path = './problematique/test/small'

    images, labels, filesizes = load_images(
        dataset_32x32_path, size=32, normalize=False, random=False)

    features = image_anal.main(images, labels)[0]
    length = 290
    
    coasts_features = features[np.where(labels == 0)][:length]
    forests_features = features[np.where(labels == 1)][:length]
    streets_features = features[np.where(labels == 2)][:length]
    coasts_labels = labels[np.where(labels==0)][:length]
    forests_labels = labels[np.where(labels==1)][:length]
    streets_labels = labels[np.where(labels==2)][:length]

    features = np.zeros((len(coasts_features)+len(forests_features)+len(streets_features),features.shape[1]))
    
    i = 0
    for c in coasts_features:
        features[i] = c
        i += 1
    for f in forests_features:
        features[i] = f
        i+=1
    for s in streets_features:
        features[i] = s
        i+=1
        
    labels = np.array([coasts_labels, forests_labels, streets_labels]).reshape(-1)

    ndonnees = 15000
    min, max = np.min(features), np.max(features)
    donneesTest = an.genDonneesTest(ndonnees, an.Extent(xmin=min, xmax=max, ymin=min, ymax=max), ndim=features.shape[1])

    if False:
        x = [coasts_features, forests_features, streets_features] #/ np.max(features)
        
        # TODO Classifier Bayesien
        # classification
        # Bayes
        #                           (train_data, train_classes, donnee_test, title, extent, test_data, test_classes)
        classifiers.full_Bayes_risk(x, labels, donneesTest,
                                    'Bayes risque #1', an.Extent(xmin=min, xmax=max, ymin=min, ymax=max), features, labels)

    if True:
        ndonnees = 15000
        coasts_features = coasts_features / np.max(coasts_features, axis=0)
        forests_features = forests_features / np.max(forests_features, axis=0)
        streets_features = streets_features / np.max(streets_features, axis=0)
        features = features  / np.max(features, axis=0)
        
        min, max = np.min(features), np.max(features)
        donneesTest = an.genDonneesTest(ndonnees, an.Extent(xmin=min, xmax=max, ymin=min, ymax=max), ndim=features.shape[1])

        data = [coasts_features, forests_features, streets_features]

        # x-mean sur chacune des classes
        # suivi d'un y-PPV avec ces nouveaux représentants de classes
        cluster_centers, cluster_labels = classifiers.full_kmean(150, data, labels, 'kmean', an.Extent(xmin=min, xmax=max, ymin=min, ymax=max))
        classifiers.full_ppv(1, cluster_centers, cluster_labels, donneesTest, '5v5', an.Extent(xmin=min, xmax=max, ymin=min, ymax=max), features, labels)

    if False:
        features = np.zeros((len(coasts_features)+len(forests_features)+len(streets_features),features.shape[1]))
        i = 0
        for c in coasts_features:
            features[i] = c
            i += 1
        for f in forests_features:
            features[i] = f
            i+=1
        for s in streets_features:
            features[i] = s
            i+=1

        features = features / np.max(features)
        labels = np.array([coasts_labels, forests_labels, streets_labels]).reshape(-1)

        # TODO Classifier NN
        n_hidden_layers = 5
        n_neurons = 19
        classifiers.full_nn(n_hidden_layers, n_neurons, features, labels, donneesTest,
                f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche', an.Extent(xmin=min, xmax=max, ymin=min, ymax=max), features, labels)
    plt.show()


######################################
if __name__ == '__main__':
    main()
