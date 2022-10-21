from contextlib import nullcontext
from operator import contains, itemgetter
from statistics import covariance

# user imports
import cv2 as cv
from cv2 import mean
from matplotlib import image
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk    
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Feature list ###
#  - Entropy == BAD
#  - Entropy per color == BAD
#  - Entropy grayscale == BAD
#  - Color histogram == Meh / TO DO plus fif  que normalized
#  - Color histogram Normalized (cool) == Meh
#  - Color histogram conversion (HSV, Lab, XYZ, CMYK, etc.) == TO DO
#  - Color histogram localized == TO DO
#  - Shape recognition == TO DO


#######################################
#   Entropy
#######################################
def extract_entropy(images, labels, display=False): 
    entropy_list = [[], [], []] # Coast, Forest, Street
    for idx, img in enumerate(images):
        
        entropy_nb = skimage.measure.shannon_entropy(img)

        if display:
            entropy_img = entropy(img, disk(1))
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
#   Skimage Feature Exploration
#######################################
def extract_skimage_features(images:np.array, labels:np.array, display:bool=False):
    for idx, img in enumerate(images):
        
        gray_img = skimage.color.rgb2gray(img)
        # edge filter
        canny_img = skimage.feature.canny(gray_img)
        canny_img_sig3 = skimage.feature.canny(gray_img, sigma=2)

        # corner SHI-TOMASI BAD
        # corner_img = skimage.feature.corner_shi_tomasi(gray_img) SHIT
        
        # HOG
        fd, hog_img = skimage.feature.hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

        if display:
            # display results
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

            ax[0].imshow(img)
            ax[0].set_title('original image', fontsize=20)

            ax[1].imshow(hog_img)
            ax[1].set_title(r'Hog filter, $\sigma=1$', fontsize=20)

            ax[2].imshow(canny_img, cmap='gray')
            ax[2].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

            for a in ax:
                a.axis('off')

            fig.tight_layout()
            plt.show()

def extract_high_freq_entropy(images:np.array, labels:np.array, sigma:int=1, display:bool=False):
    high_freq_entropy_values = np.zeros(len(labels))
    for idx, img in enumerate(images):
        # edge filter at sigma #1
        gray_img = skimage.color.rgb2gray(img)
        canny_img = skimage.feature.canny(gray_img, sigma=sigma)
        # entropy
        entropy = skimage.measure.shannon_entropy(canny_img)
        high_freq_entropy_values[idx] = entropy

    
    if display:
        fig, ax = plt.subplots()
        coasts_entropy_values  = high_freq_entropy_values[np.where(labels == 0)]
        forests_entropy_values = high_freq_entropy_values[np.where(labels == 1)]
        streets_entropy_values = high_freq_entropy_values[np.where(labels == 2)]
        
        coasts_mu = np.mean(coasts_entropy_values)
        forests_mu = np.mean(forests_entropy_values)
        streets_mu = np.mean(streets_entropy_values)

        coasts_sigma = np.std(coasts_entropy_values) * 2
        forests_sigma = np.std(forests_entropy_values) * 2
        streets_sigma = np.std(streets_entropy_values) * 2

        ax.scatter(high_freq_entropy_values[np.where(labels == 0)], labels[labels == 0], label='Coasts')
        ax.scatter(high_freq_entropy_values[np.where(labels == 1)], labels[labels == 1], label='Forests')
        ax.scatter(high_freq_entropy_values[np.where(labels == 2)], labels[labels == 2], label='Streets')
        
        ax.plot(coasts_mu,  0,'kx', markersize=10) 
        ax.plot(forests_mu, 1,'kx', markersize=10)
        ax.plot(streets_mu, 2,'kx', markersize=10) 

        ax.plot(coasts_mu - coasts_sigma,  0,'k|', markersize=20) 
        ax.plot(coasts_mu + coasts_sigma,  0,'k|', markersize=20) 
        ax.plot(forests_mu - forests_sigma, 1,'k|', markersize=20)
        ax.plot(forests_mu + forests_sigma, 1,'k|', markersize=20)
        ax.plot(streets_mu - streets_sigma, 2,'k|', markersize=20) 
        ax.plot(streets_mu + streets_sigma, 2,'k|', markersize=20) 
        
        ax.legend()
        ax.set_xlabel('Entropy of whole images')
        ax.set_ylabel('Classes')
        ax.set_title(r'Entropy of high frequencies for each classes avec moy et 2x std, $\sigma=$' + str(sigma))

        plt.show()

    return high_freq_entropy_values


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
#   Image normalization
#######################################
def correlate2d(data:np.array):
    for dim in data.shape()[0]: # It should be an array/list of ndim x ndata
        print(dim)

    # vectors = (data1, data2)
    # sigma = np.std(img)
    # n_img = (img - mu) / sigma
    # return [n_img, mu, sigma]


#######################################
#   Image Loading
#######################################
def load_images(directory, normalize=False):
    images = np.zeros((len(os.listdir(directory)), 256, 256, 3))
    labels = np.zeros(len(os.listdir(directory)))
    for idx, filename in enumerate(os.listdir(directory)):
        img = cv.imread(os.path.join(directory,filename))
        if img is not None:
            if normalize:
                images[idx] = normalize_image(img)
            else:
                images[idx] = img
    
        # Labeling
        if contains(filename, 'coast'):
            labels[idx] = 0
        elif contains(filename, 'forest'):
            labels[idx] = 1
        elif contains(filename, 'street'):
            labels[idx] = 2
    
    return images, labels

#######################################
#   mean_hsv_features
#######################################
def mean_hsv_features(images: np.array):
    mean_h = np.mean(images[:,:,0])
    mean_s = np.mean(images[:,:,1])
    mean_v = np.mean(images[:,:,2])
    return mean_h, mean_s, mean_v

#######################################
#   std_hsv_features
#######################################
def std_hsv_features(images: np.array):
    std_h = np.std(images[:,:,0])
    std_s = np.std(images[:,:,1])
    std_v = np.std(images[:,:,2])
    return std_h, std_s, std_v
    
#######################################
#   hsv_channel_value_of_images
#######################################
def hsv_channel_value_of_images(images:np.array, channel: int):
    out = np.zeros(len(images))
    for idx, img in enumerate(images):
        out[idx] = np.mean(img[:,channel])
    return out

#######################################
#   convert_to_hsv
#######################################
def convert_to_hsv(images:np.array, labels:np.array, display:bool=False):
    hsv_imgs = np.zeros(images.shape)
    for idx, img in enumerate(images):
        img_32 = np.float32(img)
        hsv_imgs[idx] = cv.cvtColor(img_32, cv.COLOR_RGB2HSV)
    
    if display:
        fig, ax = plt.subplots()
        coasts_hsv_imgs  = hsv_imgs[np.where(labels==0)]
        forests_hsv_imgs = hsv_imgs[np.where(labels == 1)]
        streets_hsv_imgs = hsv_imgs[np.where(labels == 2)]
        
        coasts_mean_hsv =  mean_hsv_features(coasts_hsv_imgs)
        coasts_std_hsv =  std_hsv_features(coasts_hsv_imgs)*2
    
        forests_mean_hsv =  mean_hsv_features(forests_hsv_imgs)
        forests_std_hsv =  std_hsv_features(forests_hsv_imgs)*2

        streets_mean_hsv =  mean_hsv_features(streets_hsv_imgs)
        streets_std_hsv =  std_hsv_features(streets_hsv_imgs)*2
        
        analyzed_channel = 2
        coasts_h = hsv_channel_value_of_images(coasts_hsv_imgs, analyzed_channel)
        forests_h =  hsv_channel_value_of_images(forests_hsv_imgs, analyzed_channel)
        streets_h =  hsv_channel_value_of_images(streets_hsv_imgs, analyzed_channel)
            
        ax.scatter(coasts_h, labels[labels == 0], label='Coasts')
        ax.scatter(forests_h, labels[labels == 1], label='Forests')
        ax.scatter(streets_h, labels[labels == 2], label='Streets')
        
        ax.plot(coasts_mean_hsv[analyzed_channel],  0,'kx', markersize=10) 
        ax.plot(forests_mean_hsv[analyzed_channel], 1,'kx', markersize=10)
        ax.plot(streets_mean_hsv[analyzed_channel], 2,'kx', markersize=10) 
        ax.plot(coasts_mean_hsv[analyzed_channel] - coasts_std_hsv[analyzed_channel] ,  0,'k|', markersize=20) 
        ax.plot(coasts_mean_hsv[analyzed_channel] + coasts_std_hsv[analyzed_channel] ,  0,'k|', markersize=20) 
        ax.plot(forests_mean_hsv[analyzed_channel] - forests_std_hsv[analyzed_channel] , 1,'k|', markersize=20)
        ax.plot(forests_mean_hsv[analyzed_channel] + forests_std_hsv[analyzed_channel] , 1,'k|', markersize=20)
        ax.plot(streets_mean_hsv[analyzed_channel] - streets_std_hsv[analyzed_channel] , 2,'k|', markersize=20) 
        ax.plot(streets_mean_hsv[analyzed_channel] + streets_std_hsv[analyzed_channel] , 2,'k|', markersize=20) 
        
        ax.legend()
        ax.set_xlabel('Saturation of whole images')
        ax.set_ylabel('Classes')
        ax.set_title(r'Saturation of images')

        plt.show()
    
    return hsv_imgs

#######################################
#   mean_lab_features
#######################################
def mean_lab_features(images: np.array):
    mean_l = np.mean(images[:,:,0])
    mean_a = np.mean(images[:,:,1])
    mean_b = np.mean(images[:,:,2])
    return mean_l, mean_a, mean_b

#######################################
#   std_lab_features
#######################################
def std_lab_features(images: np.array):
    std_l = np.std(images[:,:,0])
    std_a = np.std(images[:,:,1])
    std_b = np.std(images[:,:,2])
    return std_l, std_a, std_b
    
#######################################
#   lab_channel_value_of_images
#######################################
def lab_channel_value_of_images(images:np.array, channel: int):
    out = np.zeros(len(images))
    for idx, img in enumerate(images):
        out[idx] = np.mean(img[:,channel])
    return out

#######################################
#   convert_to_lab
#######################################
def convert_to_lab(images:np.array, labels:np.array, display:bool = False):
    lab_img = np.zeros(images.shape)
    for idx, img in enumerate(images):
        img_32 = np.float32(img)
        lab_img[idx] = cv.cvtColor(img_32, cv.COLOR_RGB2LAB)
    
    if display:
        fig, ax = plt.subplots()
        coasts_lab_img  = lab_img[np.where(labels==0)]
        forests_lab_img = lab_img[np.where(labels == 1)]
        streets_lab_img = lab_img[np.where(labels == 2)]
        
        coasts_mean_lab =  mean_lab_features(coasts_lab_img)
        coasts_std_lab =  std_lab_features(coasts_lab_img)*2
    
        forests_mean_lab =  mean_lab_features(forests_lab_img)
        forests_std_lab =  std_lab_features(forests_lab_img)*2

        streets_mean_lab =  mean_lab_features(streets_lab_img)
        streets_std_lab =  std_lab_features(streets_lab_img)*2
        
        analyzed_channel = 0
        
        coasts_l = lab_channel_value_of_images(coasts_lab_img, analyzed_channel)
        forests_l =  lab_channel_value_of_images(forests_lab_img, analyzed_channel)
        streets_l =  lab_channel_value_of_images(streets_lab_img, analyzed_channel)
            
        coasts_a = lab_channel_value_of_images(coasts_lab_img, 1)
        forests_a =  lab_channel_value_of_images(forests_lab_img, 1)
        streets_a =  lab_channel_value_of_images(streets_lab_img, 1)
        
        coasts_b = lab_channel_value_of_images(coasts_lab_img, 2)
        forests_b =  lab_channel_value_of_images(forests_lab_img, 2)
        streets_b =  lab_channel_value_of_images(streets_lab_img, 2)
        ax.scatter(coasts_a, coasts_l, label='Coasts')
        ax.scatter( forests_a, forests_l, label='Forests')
        ax.scatter( streets_a, streets_l, label='Streets')
        ax.legend()
        ax.set_xlabel(f'AB of whole images')
        ax.set_ylabel('L of whole images')
        ax.set_title(f'LAB of images')
        # ax.scatter(coasts_l, labels[labels == 0], label='Coasts')
        # ax.scatter(forests_l, labels[labels == 1], label='Forests')
        # ax.scatter(streets_l, labels[labels == 2], label='Streets')
        
        # ax.plot(coasts_mean_lab[analyzed_channel],  0,'kx', markersize=10) 
        # ax.plot(forests_mean_lab[analyzed_channel], 1,'kx', markersize=10)
        # ax.plot(streets_mean_lab[analyzed_channel], 2,'kx', markersize=10) 
        # ax.plot(coasts_mean_lab[analyzed_channel] - coasts_std_lab[analyzed_channel] ,  0,'k|', markersize=20) 
        # ax.plot(coasts_mean_lab[analyzed_channel] + coasts_std_lab[analyzed_channel] ,  0,'k|', markersize=20) 
        # ax.plot(forests_mean_lab[analyzed_channel] - forests_std_lab[analyzed_channel] , 1,'k|', markersize=20)
        # ax.plot(forests_mean_lab[analyzed_channel] + forests_std_lab[analyzed_channel] , 1,'k|', markersize=20)
        # ax.plot(streets_mean_lab[analyzed_channel] - streets_std_lab[analyzed_channel] , 2,'k|', markersize=20) 
        # ax.plot(streets_mean_lab[analyzed_channel] + streets_std_lab[analyzed_channel] , 2,'k|', markersize=20) 
        
        # ax.legend()
        # ax.set_xlabel(f'{analyzed_channel} of LAB of whole images')
        # ax.set_ylabel('Classes')
        # ax.set_title(f'{analyzed_channel} of LAB of images')

        plt.show()
    return images

#######################################
#   Main
#######################################
def main():
    # Data load
    normalized = False
    dataset_path = './problematique/baseDeDonneesImages'
    dataset_short_path = "./problematique/test/"
    images, labels = load_images(dataset_path, normalize=normalized)

    images_mat = np.zeros(len(labels))
    if normalized:
        images_mat = list(map(itemgetter(0), images))
    else:
        images_mat = images
   
   
    features = []
    # Features
    if False:
        canny = extract_high_freq_entropy(images_mat, labels, sigma=1, display=True)
        features.append(canny)
    if True:
        color_hist = extract_color_histogram(images_mat, labels)
        features.append(color_hist)
    if False:
        simple_stats = extract_simple_stats(images_mat, labels)
        features.append(simple_stats)
    if False:
        skimage_features = extract_skimage_features(images_mat, labels, display=True)
        features.append(skimage_features)
    if False:
        entropy = extract_entropy(images_mat, labels)
        features.append(entropy)
    if False:
        hsv = convert_to_hsv(images_mat, labels, display=True)
        features.append(hsv)
    if True:
        lab = convert_to_lab(images_mat, labels, display=True)
        features.append(lab)

    # Correlation
    if True:
        correlate2d(np.array(features))
    


######################################
if __name__ == '__main__':
    main()



    

    

    
    

    