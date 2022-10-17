from operator import contains, itemgetter
from statistics import covariance

# user imports
import cv2 as cv
from cv2 import mean
from matplotlib import image
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skimage.measure     
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Feature list ###
#  - Entropy == BAD
#  - Entropy per color == BAD
#  - Entropy grayscale == BAD
#  - Color histogram == Meh
#  - Color histogram conversion (HSV, Lab, XYZ, CMYK, etc.) == TO DO
#  - Color histogram localized == TO DO
#  - Shape recognition == TO DO


#######################################
#   Entropy
#######################################
def extract_entropy(images, labels, display=False): 
    entropy_list = [[], [], []] # Coast, Forest, Street
    for idx, img in enumerate(images):
        entropy_img = entropy(img, disk(1))
        entropy_nb = skimage.measure.shannon_entropy(img)

        if display:
            cv.imshow('img', img)
            cv.imshow('entropy_img', entropy_img)
            cv.waitKey()
            cv.destroyAllWindows()

        # Labeling
        if labels[idx] == 0:
            entropy_list[0].append(entropy_nb)
        if labels[idx] == 1:
            entropy_list[1].append(entropy_nb)
        if labels[idx] == 2:
            entropy_list[2].append(entropy_nb)
    
    view_entropy(entropy_list)

def view_entropy(entropy_list):
    # Entropy ça suce...
    print('ENTROPY ANALYSIS...')
    print("COAST \n MEAN: ", np.sum(entropy_list[0])/len(entropy_list[0]), "\n STD: ", np.std(entropy_list[0]))
    print("FOREST \n MEAN: ", np.sum(entropy_list[1])/len(entropy_list[1]), "\n STD: ", np.std(entropy_list[1]))
    print("STREET \n MEAN: ", np.sum(entropy_list[2])/len(entropy_list[2]), "\n STD: ", np.std(entropy_list[2]))

    plt.hist(entropy_list[0], bins=40)
    plt.hist(entropy_list[1], bins=40)
    plt.hist(entropy_list[2], bins=40)
    labels = ["coast", "forest", "street"]
    plt.legend(labels)

    plt.show()


#######################################
#   Color histograms
#######################################
def extract_color_histogram(images, labels):
    color_list = [[], [], []]   # Coast, Forest, Street | intensity bins
    n_bins = 256
    for idx,img in enumerate(images):
        # Color bining
        color = np.zeros((3, n_bins))
        for i in range(n_bins):
            for j in range(3):
                color[j, i] = np.count_nonzero(img[:, :, j] == i)

        # Labeling
        if labels[idx] == 0:
            color_list[0].append(color)
        if labels[idx] == 1:
            color_list[1].append(color)
        if labels[idx] == 2:
            color_list[2].append(color)
    
    view_colors(color_list, n_bins)

def view_colors(color_list, n_bins):
    for i in range(3):
        counter = 0
        color = np.zeros((3, n_bins))
        for img in color_list[i]:
            counter += 1
            color += img
        color = color / counter
        plt.figure(i)
        plt.scatter(range(n_bins), color[0]/max(color[0]), c='blue')
        plt.scatter(range(n_bins), color[1]/max(color[1]), c='green')
        plt.scatter(range(n_bins), color[2]/max(color[2]), c='red')
       # plt.set(xlabel='pixels', ylabel='compte par valeur d\'intensité')

    plt.show()

#######################################
#   simple stats
#######################################
def extract_simple_stats(images, labels):
    sum_pixel_gray = 0
    max_pixel_gray = 0
    min_pixel_gray = 0
    mean_pixel_gray = 0
    mean_pixel_blue = 0
    mean_pixel_green = 0
    mean_pixel_red = 0

    for idx, img in enumerate(images):
        print("awww")

#######################################
#   Image normalization
#######################################
def normalize_image(img):
    mu = np.mean(img)
    sigma = np.std(img)
    n_img = (img - mu) / sigma
    return [n_img, mu, sigma]

def denormalize_image(img, mu, sigma):
    return (img * sigma) + mu

#######################################
#   Image Loading
#######################################
def load_images(directory, normalize=False):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img = cv.imread(os.path.join(directory,filename))
        if img is not None:
            if normalize:
                images.append(normalize_image(img))
            else:
                images.append()
    
        # Labeling
        if contains(filename, 'coast'):
            labels.append(0)
        elif contains(filename, 'forest'):
            labels.append(1)
        elif contains(filename, 'street'):
            labels.append(2)
    
    return images, labels

#######################################
#   Main
#######################################
def main():
    # Data load
    normalized = False
    dataset_path = './problematique/baseDeDonneesImages'
    dataset_short_path = "./problematique/test/"
    images, labels = load_images(dataset_short_path, normalize=normalized)

    if normalized:
        images_mat = list(map(itemgetter(0), images))
    else:
        images_mat = images
   
    # Features
    if False:
        extract_entropy(images_mat, labels)
    if True:
        extract_color_histogram(images_mat, labels)
    if False:
        extract_simple_stats(images_mat, labels)


######################################
if __name__ == '__main__':
    main()



    

    

    
    

    