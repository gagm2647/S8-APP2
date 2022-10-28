import enum
from turtle import color

# user imports
import cv2 as cv
from cv2 import mean
from cv2 import integral
from matplotlib import image
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import copy
from plot import *


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
    entropy_list = [[], [], []]  # Coast, Forest, Street
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
    print("COAST \n MEAN: ", np.sum(
        entropy_list[0])/len(entropy_list[0]), "\n STD: ", np.std(entropy_list[0]))
    print("FOREST \n MEAN: ", np.sum(
        entropy_list[1])/len(entropy_list[1]), "\n STD: ", np.std(entropy_list[1]))
    print("STREET \n MEAN: ", np.sum(
        entropy_list[2])/len(entropy_list[2]), "\n STD: ", np.std(entropy_list[2]))

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
    color_list = [[], [], []]   # Coast, Forest, Street | RGB intensity bins
    n_bins = 256
    for idx, img in enumerate(images):
        img = np.round(img * 256)
        # Color bining
        color = np.zeros((3, n_bins))
        for i in range(n_bins):
            for j in range(3):
                color[j, i] = np.count_nonzero(img[:, :, j] == i)

        # Labeling Coast, Forest, Street
        if labels[idx] == 0:
            color_list[0].append(color)
        if labels[idx] == 1:
            color_list[1].append(color)
        if labels[idx] == 2:
            color_list[2].append(color)

    view_colors(color_list, n_bins, labels)


def extract_integral_fft(img: np.array, skip: int, nbins: int):
    ffts = img[:, skip:-skip]
    integrals = np.zeros(3)
    fftxaxis = range(nbins)[skip:-skip]
    for i in range(3):
        #ffts[i] = np.abs(np.fft.fftshift(np.fft.fft(img[i])[skip:-skip]))
        integrals[i] = np.sum(img[i, :])
    return ffts, integrals


def view_colors(color_list, n_bins, labels: np.array):
    coast_integrals  = np.zeros((len(color_list[0]), 3))
    forest_integrals = np.zeros((len(color_list[1]), 3))
    street_integrals = np.zeros((len(color_list[2]), 3))
    integrals = np.array([coast_integrals, forest_integrals, street_integrals])
    
    for i, classe in enumerate(color_list):
        for j, img in enumerate(classe):
            integral_r = np.sum(img[0, :] / np.max(img[0, :]))
            integral_g = np.sum(img[1, :] / np.max(img[1, :]))
            integral_b = np.sum(img[2, :] / np.max(img[2, :]))
            integrals[i][j] = np.array([integral_r, integral_g, integral_b])


    print('pouet')

    # integrals = np.zeros((3, 3))
    # stds = np.zeros((3, 3))
    # skip = 15
    # ffts = []
    # integrals = []
    # for i in range(3):
    #     color = np.zeros((3, n_bins))
    #     fftxaxis = range(n_bins)[skip:-skip]

    #     for k, img in enumerate(color_list[i]):
    #         color = np.add(color, img)
            
    #         fft, integral = extract_integral_fft(img, skip, n_bins)
    #         ffts.append(fft)
            # integrals.append(integral)
            # plt.figure(k)
            # plt.scatter(fftxaxis, ffts[0], c='blue')
            # plt.scatter(fftxaxis, ffts[1], c='green')
            # plt.scatter(fftxaxis, ffts[2], c='red')
            # plt.title(f"{k}th img fft")
        # plt.show()
        # color = color / len(color_list[i])
        # color[0] = color[0]/max(color[0])
        # color[1] = color[1]/max(color[1])
        # color[2] = color[2]/max(color[2])
        # colors = color[:, skip:-skip]
        # ffts = np.zeros(colors.shape)
        # for j in range(len(ffts)):
        #     ffts[j] = np.abs(np.fft.fftshift(np.fft.fft(color[j])[skip:-skip]))
        #     integrals[i][j] = np.trapz(ffts[j], x=[fftxaxis])
        #     stds[i][j] = np.std(integrals[i][j])

        # compute integral of blue channel

        # plt.figure(i)
        # plt.scatter(range(n_bins), color[0], c='blue')
        # plt.scatter(range(n_bins), color[1], c='green')
        # plt.scatter(range(n_bins), color[2], c='red')
        # plt.title('Color Range')
        # plt.figure(i+3)
        # plt.scatter(fftxaxis, ffts[0] / max(ffts[0]), c='blue')
        # plt.scatter(fftxaxis, ffts[1] / max(ffts[1]), c='green')
        # plt.scatter(fftxaxis, ffts[2] / max(ffts[2]), c='red')
        # plt.title('FFTs')
    integrals_stats = np.zeros((3, 3, 2))
    for idx, classe in enumerate(integrals):
        for c_channel in range(3):
            mean = np.mean(classe[:, c_channel])
            std  = np.std(classe[:, c_channel])
            integrals_stats[idx, c_channel] = np.array(mean, std)

    coasts_integrals_mean = integrals_stats[0, :, 0]
    coasts_integrals_std = integrals_stats[0, :, 1]
    forests_integrals_mean = integrals_stats[1, :, 0]
    forests_integrals_std = integrals_stats[1, :, 1]
    streets_integrals_mean = integrals_stats[2, :, 0]
    streets_integrals_std = integrals_stats[2, :, 1]

    data = {
        "Coasts": coasts_integrals_mean,
        "Forests": forests_integrals_mean,
        "Streets": streets_integrals_mean
    }
    print(coasts_integrals_mean, forests_integrals_mean, streets_integrals_mean)
    print(coasts_integrals_std, forests_integrals_std, streets_integrals_std)
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=0.8, single_width=0.9,
             xbar=['R', 'G', 'B'], stache=[coasts_integrals_std, forests_integrals_std, streets_integrals_std])

    plt.show()
    print('pouet')

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
def extract_skimage_features(images: np.array, labels: np.array, display: bool = False):
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


def extract_daisy_features(images: np.array, labels: np.array, display: bool = False):
    daisy_values = np.zeros(len(labels))
    for idx, img in enumerate(images):
        # edge filter at sigma #1
        gray_img = skimage.color.rgb2gray(img)
        canny_img = skimage.feature.canny(gray_img, sigma=1)
        # entropy
        entropy = skimage.measure.shannon_entropy(canny_img)
        daisy_values[idx] = entropy

    if display:
        fig, ax = plt.subplots()
        coasts_entropy_values = daisy_values[np.where(labels == 0)]
        forests_entropy_values = daisy_values[np.where(
            labels == 1)]
        streets_entropy_values = daisy_values[np.where(
            labels == 2)]

        coasts_mu = np.mean(coasts_entropy_values)
        forests_mu = np.mean(forests_entropy_values)
        streets_mu = np.mean(streets_entropy_values)

        coasts_sigma = np.std(coasts_entropy_values) * 2
        forests_sigma = np.std(forests_entropy_values) * 2
        streets_sigma = np.std(streets_entropy_values) * 2

        ax.scatter(daisy_values[np.where(
            labels == 0)], labels[labels == 0], label='Coasts')
        ax.scatter(daisy_values[np.where(
            labels == 1)], labels[labels == 1], label='Forests')
        ax.scatter(daisy_values[np.where(
            labels == 2)], labels[labels == 2], label='Streets')

        ax.plot(coasts_mu,  0, 'kx', markersize=10)
        ax.plot(forests_mu, 1, 'kx', markersize=10)
        ax.plot(streets_mu, 2, 'kx', markersize=10)

        ax.plot(coasts_mu - coasts_sigma,  0, 'k|', markersize=20)
        ax.plot(coasts_mu + coasts_sigma,  0, 'k|', markersize=20)
        ax.plot(forests_mu - forests_sigma, 1, 'k|', markersize=20)
        ax.plot(forests_mu + forests_sigma, 1, 'k|', markersize=20)
        ax.plot(streets_mu - streets_sigma, 2, 'k|', markersize=20)
        ax.plot(streets_mu + streets_sigma, 2, 'k|', markersize=20)

        ax.legend()
        ax.set_xlabel('Entropy of whole images')
        ax.set_ylabel('Classes')
        ax.set_title(
            r'Entropy of high frequencies for each classes avec moy et 2x std')

        plt.show()

    return daisy_values


def extract_high_freq_entropy(images: np.array, labels: np.array, sigma: int = 1, display: bool = False):
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
        coasts_entropy_values = high_freq_entropy_values[np.where(labels == 0)]
        forests_entropy_values = high_freq_entropy_values[np.where(
            labels == 1)]
        streets_entropy_values = high_freq_entropy_values[np.where(
            labels == 2)]

        coasts_mu = np.mean(coasts_entropy_values)
        forests_mu = np.mean(forests_entropy_values)
        streets_mu = np.mean(streets_entropy_values)

        coasts_sigma = np.std(coasts_entropy_values) * 2
        forests_sigma = np.std(forests_entropy_values) * 2
        streets_sigma = np.std(streets_entropy_values) * 2

        ax.scatter(high_freq_entropy_values[np.where(
            labels == 0)], labels[labels == 0], label='Coasts')
        ax.scatter(high_freq_entropy_values[np.where(
            labels == 1)], labels[labels == 1], label='Forests')
        ax.scatter(high_freq_entropy_values[np.where(
            labels == 2)], labels[labels == 2], label='Streets')

        ax.plot(coasts_mu,  0, 'kx', markersize=10)
        ax.plot(forests_mu, 1, 'kx', markersize=10)
        ax.plot(streets_mu, 2, 'kx', markersize=10)

        ax.plot(coasts_mu - coasts_sigma,  0, 'k|', markersize=20)
        ax.plot(coasts_mu + coasts_sigma,  0, 'k|', markersize=20)
        ax.plot(forests_mu - forests_sigma, 1, 'k|', markersize=20)
        ax.plot(forests_mu + forests_sigma, 1, 'k|', markersize=20)
        ax.plot(streets_mu - streets_sigma, 2, 'k|', markersize=20)
        ax.plot(streets_mu + streets_sigma, 2, 'k|', markersize=20)

        ax.legend()
        ax.set_xlabel('Entropy of whole images')
        ax.set_ylabel('Classes')
        ax.set_title(
            r'Entropy of high frequencies for each classes avec moy et 2x std, $\sigma=$' + str(sigma))

        plt.show()

    return high_freq_entropy_values


def denormalize_image(img, mu, sigma):
    return (img * sigma) + mu

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

#######################################
#   Correlation des données ?
#######################################


def correlate2d(data: np.array):
    for dim in data.shape()[0]:  # It should be an array/list of ndim x ndata
        print(dim)

    # vectors = (data1, data2)
    # sigma = np.std(img)
    # n_img = (img - mu) / sigma
    # return [n_img, mu, sigma]


#######################################
#   mean_hsv_features
#######################################


def mean_hsv_features(images: np.array):
    mean_h = np.mean(images[:, :, 0])
    mean_s = np.mean(images[:, :, 1])
    mean_v = np.mean(images[:, :, 2])
    return mean_h, mean_s, mean_v

#######################################
#   std_hsv_features
#######################################


def std_hsv_features(images: np.array):
    std_h = np.std(images[:, :, 0])
    std_s = np.std(images[:, :, 1])
    std_v = np.std(images[:, :, 2])
    return std_h, std_s, std_v

#######################################
#   hsv_channel_value_of_images
#######################################


def hsv_channel_value_of_images(images: np.array, channel: int):
    out = np.zeros(len(images))
    for idx, img in enumerate(images):
        out[idx] = np.mean(img[:, channel])
    return out

#######################################
#   convert_to_hsv
#######################################


def convert_to_hsv(images: np.array, labels: np.array, display: bool = False):
    hsv_imgs = np.zeros(images.shape)
    for idx, img in enumerate(images):
        img_32 = np.float32(img)
        hsv_imgs[idx] = cv.cvtColor(img_32, cv.COLOR_RGB2HSV)

    if display:
        fig, ax = plt.subplots()
        coasts_hsv_imgs = hsv_imgs[np.where(labels == 0)]
        forests_hsv_imgs = hsv_imgs[np.where(labels == 1)]
        streets_hsv_imgs = hsv_imgs[np.where(labels == 2)]

        coasts_mean_hsv = mean_hsv_features(coasts_hsv_imgs)
        coasts_std_hsv = std_hsv_features(coasts_hsv_imgs)*2

        forests_mean_hsv = mean_hsv_features(forests_hsv_imgs)
        forests_std_hsv = std_hsv_features(forests_hsv_imgs)*2

        streets_mean_hsv = mean_hsv_features(streets_hsv_imgs)
        streets_std_hsv = std_hsv_features(streets_hsv_imgs)*2

        analyzed_channel = 2
        coasts_h = hsv_channel_value_of_images(
            coasts_hsv_imgs, analyzed_channel)
        forests_h = hsv_channel_value_of_images(
            forests_hsv_imgs, analyzed_channel)
        streets_h = hsv_channel_value_of_images(
            streets_hsv_imgs, analyzed_channel)

        ax.scatter(coasts_h, labels[labels == 0], label='Coasts')
        ax.scatter(forests_h, labels[labels == 1], label='Forests')
        ax.scatter(streets_h, labels[labels == 2], label='Streets')

        ax.plot(coasts_mean_hsv[analyzed_channel],  0, 'kx', markersize=10)
        ax.plot(forests_mean_hsv[analyzed_channel], 1, 'kx', markersize=10)
        ax.plot(streets_mean_hsv[analyzed_channel], 2, 'kx', markersize=10)
        ax.plot(coasts_mean_hsv[analyzed_channel] -
                coasts_std_hsv[analyzed_channel],  0, 'k|', markersize=20)
        ax.plot(coasts_mean_hsv[analyzed_channel] +
                coasts_std_hsv[analyzed_channel],  0, 'k|', markersize=20)
        ax.plot(forests_mean_hsv[analyzed_channel] -
                forests_std_hsv[analyzed_channel], 1, 'k|', markersize=20)
        ax.plot(forests_mean_hsv[analyzed_channel] +
                forests_std_hsv[analyzed_channel], 1, 'k|', markersize=20)
        ax.plot(streets_mean_hsv[analyzed_channel] -
                streets_std_hsv[analyzed_channel], 2, 'k|', markersize=20)
        ax.plot(streets_mean_hsv[analyzed_channel] +
                streets_std_hsv[analyzed_channel], 2, 'k|', markersize=20)

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
    mean_l = np.mean(images[:, :, 0])
    mean_a = np.mean(images[:, :, 1])
    mean_b = np.mean(images[:, :, 2])
    return mean_l, mean_a, mean_b

#######################################
#   std_lab_features
#######################################


def std_lab_features(images: np.array):
    std_l = np.std(images[:, :, 0])
    std_a = np.std(images[:, :, 1])
    std_b = np.std(images[:, :, 2])
    return std_l, std_a, std_b

#######################################
#   lab_channel_value_of_images
#######################################


def lab_channel_value_of_images(images: np.array, channel: int):
    out = np.zeros(len(images))
    for idx, img in enumerate(images):
        out[idx] = np.mean(img[:, channel])
    return out

#######################################
#   convert_to_lab
#######################################


def convert_to_lab(images: np.array, labels: np.array, display: bool = False):
    lab_img = np.zeros(images.shape)
    for idx, img in enumerate(images):
        img_32 = np.float32(img)
        lab_img[idx] = cv.cvtColor(img_32, cv.COLOR_RGB2LAB)

    if display:
        fig, ax = plt.subplots()
        coasts_lab_img = lab_img[np.where(labels == 0)]
        forests_lab_img = lab_img[np.where(labels == 1)]
        streets_lab_img = lab_img[np.where(labels == 2)]

        coasts_mean_lab = mean_lab_features(coasts_lab_img)
        coasts_std_lab = std_lab_features(coasts_lab_img)*2

        forests_mean_lab = mean_lab_features(forests_lab_img)
        forests_std_lab = std_lab_features(forests_lab_img)*2

        streets_mean_lab = mean_lab_features(streets_lab_img)
        streets_std_lab = std_lab_features(streets_lab_img)*2

        analyzed_channel = 0

        coasts_l = lab_channel_value_of_images(
            coasts_lab_img, analyzed_channel)
        forests_l = lab_channel_value_of_images(
            forests_lab_img, analyzed_channel)
        streets_l = lab_channel_value_of_images(
            streets_lab_img, analyzed_channel)

        coasts_a = lab_channel_value_of_images(coasts_lab_img, 1)
        forests_a = lab_channel_value_of_images(forests_lab_img, 1)
        streets_a = lab_channel_value_of_images(streets_lab_img, 1)

        coasts_b = lab_channel_value_of_images(coasts_lab_img, 2)
        forests_b = lab_channel_value_of_images(forests_lab_img, 2)
        streets_b = lab_channel_value_of_images(streets_lab_img, 2)
        ax.scatter(coasts_a, coasts_l, label='Coasts')
        ax.scatter(forests_a, forests_l, label='Forests')
        ax.scatter(streets_a, streets_l, label='Streets')
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


def extract_noise_level(images: np.array, labels: np.array, display: bool = False):
    noise = np.zeros(len(labels))
    gray_imgs = np.zeros(len(labels))
    mean_img = np.zeros(images.shape)
    mean_img = np.sum(images, 1)
    # for idx, img in enumerate(images):
    #     mean_img = np.add(mean_img, img)
    mean_img = np.uint8(mean_img // len(labels))

    for idx, img in enumerate(images):
        gray_img = cv.cvtColor(np.uint8(img), cv.COLOR_RGB2GRAY)
        gray_imgs[idx] = np.mean(gray_img)

        noise[idx] = cv.PSNR(np.uint8(gray_img), np.uint8(mean_img))

    if display:
        fig, ax = plt.subplots()
        coasts_noise = noise[np.where(labels == 0)]
        forests_noise = noise[np.where(labels == 1)]
        streets_noise = noise[np.where(labels == 2)]

        coasts_gray = gray_imgs[np.where(labels == 0)]
        forests_gray = gray_imgs[np.where(labels == 1)]
        streets_gray = gray_imgs[np.where(labels == 2)]

        ax.scatter(coasts_noise, coasts_gray, label='Coasts')
        ax.scatter(forests_noise, forests_gray, label='Forests')
        ax.scatter(streets_noise, streets_gray, label='Streets')
        ax.legend()
        ax.set_xlabel(f'Noise level')
        ax.set_ylabel('Grayscale')
        ax.set_title(f'Noise vs. Gray')
        plt.show()


def get_fft(images: np.array, labels: np.array, display: bool = False):
    ffts = np.zeros(images.shape[:-1])

    for idx, img in enumerate(images):
        gray_img = cv.cvtColor(np.uint8(img), cv.COLOR_RGB2GRAY)
        ffts[idx] = np.fft.fft(gray_img)

    if display:
        fig, ax = plt.subplots()
        coasts_ffts = ffts[np.where(labels == 0)]
        forests_ffts = ffts[np.where(labels == 1)]
        streets_ffts = ffts[np.where(labels == 2)]

        ax.plot(coasts_ffts[0], label='Coasts')
        ax.plot(forests_ffts[0], label='Forests')
        ax.plot(streets_ffts[0], label='Streets')
        ax.legend()
        ax.set_xlabel(f'img')
        ax.set_ylabel('ffts')
        ax.set_title(f'Noise vs. Gray')
        plt.show()


def view_filesize(filesizes: np.array, labels: np.array):
    fig, ax = plt.subplots()
    coasts_filesizes = filesizes[np.where(labels == 0)]
    forests_filesizes = filesizes[np.where(labels == 1)]
    streets_filesizes = filesizes[np.where(labels == 2)]

    coasts_mu = np.mean(coasts_filesizes)
    forests_mu = np.mean(forests_filesizes)
    streets_mu = np.mean(streets_filesizes)

    coasts_sigma = np.std(coasts_filesizes) * 2
    forests_sigma = np.std(forests_filesizes) * 2
    streets_sigma = np.std(streets_filesizes) * 2

    ax.scatter(filesizes[np.where(labels == 0)],
               labels[labels == 0], label='Coasts')
    ax.scatter(filesizes[np.where(labels == 1)],
               labels[labels == 1], label='Forests')
    ax.scatter(filesizes[np.where(labels == 2)],
               labels[labels == 2], label='Streets')

    ax.plot(coasts_mu,  0, 'kx', markersize=10)
    ax.plot(forests_mu, 1, 'kx', markersize=10)
    ax.plot(streets_mu, 2, 'kx', markersize=10)

    ax.plot(coasts_mu - coasts_sigma,  0, 'k|', markersize=20)
    ax.plot(coasts_mu + coasts_sigma,  0, 'k|', markersize=20)
    ax.plot(forests_mu - forests_sigma, 1, 'k|', markersize=20)
    ax.plot(forests_mu + forests_sigma, 1, 'k|', markersize=20)
    ax.plot(streets_mu - streets_sigma, 2, 'k|', markersize=20)
    ax.plot(streets_mu + streets_sigma, 2, 'k|', markersize=20)

    ax.legend()
    ax.set_xlabel('Entropy of whole images')
    ax.set_ylabel('Classes')
    ax.set_title(r'Filesizes for each classes avec moy et 2x std')

    plt.show()

# from skimage import data, io, segmentation, color
# from skimage.future import graph


def rag_merging(images: np.array):
    outs = np.zeros(images.shape)
    for idx, img in enumerate(images):
        # img[:, :, 0] = img[:, :, 0] / (np.max(img[:, :, 0]) * 256 + 10**-15)
        # img[:, :, 1] = img[:, :, 1] / (np.max(img[:, :, 1]) * 256 + 10**-15)
        # img[:, :, 2] = img[:, :, 2] / (np.max(img[:, :, 2]) * 256 + 10**-15)
        print('rag img number:', idx)
        rag_labels = skimage.segmentation.slic(
            img, compactness=30, n_segments=400, start_label=1)
        g = skimage.future.graph.rag_mean_color(img, rag_labels)

        rag_labels2 = skimage.future.graph.merge_hierarchical(rag_labels, g, thresh=35, rag_copy=False,
                                                              in_place_merge=True,
                                                              merge_func=merge_mean_color,
                                                              weight_func=weight_mean_color)

        out = skimage.color.label2rgb(rag_labels2, img, kind='avg', bg_label=0)
        out = skimage.segmentation.mark_boundaries(out, rag_labels2, (0, 0, 0))
        outs[idx] = out
        # cv.imshow('original img', img.astype(np.uint8))
        # cv.imshow('rag merged img', out.astype(np.uint8))
        # cv.waitKey()
        # cv.destroyAllWindows()
    return outs


def weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


#######################################
#   Main
#######################################


def main(images: np.array, labels: np.array):
    features = []
    # features
    if False:
        x = rag_merging(images)
        canny = extract_high_freq_entropy(
            x, labels, sigma=1, display=True)
        features.append(canny)
    if False:
        view_filesize(filesizes, labels)
    if False:
        canny = extract_daisy_features(
            images, labels, sigma=1, display=True)
        features.append(canny)
    if False:
        canny = extract_high_freq_entropy(
            images, labels, sigma=1, display=True)
        features.append(canny)
    if True:
        color_hist = extract_color_histogram(images, labels)
        features.append(color_hist)
    if False:
        simple_stats = extract_simple_stats(images, labels)
        features.append(simple_stats)
    if False:
        skimage_features = extract_skimage_features(
            images, labels, display=True)
        features.append(skimage_features)
    if False:
        entropy = extract_entropy(images, labels)
        features.append(entropy)
    if False:
        hsv = convert_to_hsv(images, labels, display=True)
        features.append(hsv)
    if False:
        lab = convert_to_lab(images, labels, display=True)
        features.append(lab)
    if False:  # @TODO : make it work, with mean images I guess for the PSNR
        noise = extract_noise_level(images, labels, True)
    if False:
        fft = get_fft(images, labels, True)

    # Correlation
    if False:
        correlate2d(np.array(features))
    return np.array(features)


######################################
if __name__ == '__main__':
    main()
