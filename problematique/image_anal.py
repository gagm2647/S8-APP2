from operator import contains
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skimage.measure     
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

def view_colors(color_list):
    # FUUUUUU

    n_bins = 256
    for i in range(3):
        counter = 0
        color = np.zeros((3, n_bins))
        for img in color_list[i]:
            counter += 1
            color += img
        color = color / counter
        plt.figure(i)
        plt.scatter(range(n_bins), color[0], c='blue')
        plt.scatter(range(n_bins), color[1], c='green')
        plt.scatter(range(n_bins), color[2], c='red')
       # plt.set(xlabel='pixels', ylabel='compte par valeur d\'intensité')

    plt.show()



    print('ENTROPY ANALYSIS...')
    print("COAST \n MEAN: ", np.sum(color_list[0])/len(color_list[0]), "\n STD: ", np.std(color_list[0]))
    print("FOREST \n MEAN: ", np.sum(color_list[1])/len(color_list[1]), "\n STD: ", np.std(color_list[1]))
    print("STREET \n MEAN: ", np.sum(color_list[2])/len(color_list[2]), "\n STD: ", np.std(color_list[2]))

    plt.hist(np.sum(color_list[0])/len(color_list[0]), bins=256)
    plt.hist(np.sum(color_list[1])/len(color_list[1]), bins=256)
    plt.hist(np.sum(color_list[2])/len(color_list[2]), bins=256)
    labels = ["coast", "forest", "street"]
    plt.legend(labels)

    plt.show()
    
def pisse(list, n_bins):
    print('Fuck u')


dataset_path = './problematique/baseDeDonneesImages'
dataset_short_path = "./problematique/test/"

images = []
entropy_list = [[], [], []] # Coast, Forest, Street
color_list = [[], [], []]   # Coast, Forest, Street | [BGR] = 0 1 2

for filename in os.listdir(dataset_short_path):
    img = cv.imread(os.path.join(dataset_short_path,filename))
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img = img[:, :, 2]
    if img is not None:
        images.append(img)

    

    # # Entropy
    # entropy_img = entropy(img, disk(1))
    # entropy_nb = skimage.measure.shannon_entropy(img)
    # # cv.imshow('img', img)
    # # cv.imshow('entropy_img', entropy_img)
    # # cv.waitKey()
    # # cv.destroyAllWindows()

    # if contains(filename, 'coast'):
    #     entropy_list[0].append(entropy_nb)
    # elif contains(filename, 'forest'):
    #     entropy_list[1].append(entropy_nb)
    # if contains(filename, 'street'):
    #     entropy_list[2].append(entropy_nb)

    # Color Analysis
    n_bins = 256
    color = np.zeros((3, n_bins))
    for i in range(n_bins):
        for j in range(3):
            color[j, i] = np.count_nonzero(img[:, :, j] == i)

    

    if contains(filename, 'coast'):
        color_list[0].append(color)
    elif contains(filename, 'forest'):
        color_list[1].append(color)
    elif contains(filename, 'street'):
        color_list[2].append(color)
    
print("fuk u")

    
view_colors(color_list)
#view_entropy(entropy_list)