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

import classifiers

#######################################


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
def normalize_image(img: np.array, display: bool = False):
    mu = np.mean(img)
    sigma = np.std(img)
    n_img = (img - mu) / sigma

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
                    images[idx], mu, sigma = normalize_image(
                        img, display=False)
                    # Comprehensive color normalize
                    #images[idx] = comprehensive_color_normalize(img, display=False)
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
                    images[idx], mu, sigma = normalize_image(
                        img, display=False)
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
        dataset_32x32_path, size=32, normalize=False, random=True)

    features = image_anal.main(images, labels)[0]

    length = 50
    
    coasts_features = features[np.where(labels == 0)][:length]
    forests_features = features[np.where(labels == 1)][:length]
    streets_features = features[np.where(labels == 2)][:length]
    coasts_labels = labels[np.where(labels==0)][:length]
    forests_labels = labels[np.where(labels==1)][:length]
    streets_labels = labels[np.where(labels==2)][:length]
    
    features = np.array([coasts_features, forests_features, streets_features])
    labels = np.array([coasts_labels, forests_labels, streets_labels])
    donneesTest = []
    labels = []

    if False:
        # TODO Classifier Bayesien
        ret = 0
        # classification
        # Bayes
        #                           (train_data, train_classes, donnee_test, title, extent, test_data, test_classes)
        classifiers.full_Bayes_risk(features, labels, donneesTest,
                                    'Bayes risque #1', [np.min(features), np.max(features)], donneesTest, labels)

    if False:
        # TODO Classifier K-Mean
        ret = 1

    if True:
        # TODO Classifier NN
        ret = 2
        n_hidden_layers = 3
        n_neurons = 5
        classifiers.full_nn(n_hidden_layers, n_neurons, features, labels, donneesTest,
                f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche', TroisClasses.extent, features, labels)



######################################
if __name__ == '__main__':
    main()
