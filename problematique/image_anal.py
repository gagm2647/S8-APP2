from contextlib import nullcontext
from operator import contains, itemgetter
from statistics import covariance

# user imports
import cv2 as cv
from cv2 import mean
from cv2 import integral
from matplotlib import image
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    color_list = [[], [], []]   # Coast, Forest, Street | intensity bins
    n_bins = 256
    for idx, img in enumerate(images):
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

    view_colors(color_list, n_bins, labels)


def extract_integral_fft(img: np.array, skip: int, nbins: int):
    ffts = img[:, skip:-skip]
    integrals = np.zeros(len(img))
    fftxaxis = range(nbins)[skip:-skip]
    for i in range(len(img)):
        ffts[i] = np.abs(np.fft.fftshift(np.fft.fft(img[i])[skip:-skip]))
        integrals[i] = np.trapz(ffts[i], x=[fftxaxis])
    return ffts, integrals


def view_colors(color_list, n_bins, labels: np.array):
    integrals = np.zeros((3, 3))
    stds = np.zeros((3, 3))
    skip = 15
    ffts = []
    integrals = []
    for i in range(3):
        color = np.zeros((3, n_bins))
        fftxaxis = range(n_bins)[skip:-skip]

        for k, img in enumerate(color_list[i]):
            color = np.add(color, img)
            fft, integral = extract_integral_fft(img, skip, n_bins)
            ffts.append(fft)
            integrals.append(integral)
            # plt.figure(k)
            # plt.scatter(fftxaxis, ffts[0], c='blue')
            # plt.scatter(fftxaxis, ffts[1], c='green')
            # plt.scatter(fftxaxis, ffts[2], c='red')
            # plt.title(f"{k}th img fft")
        # plt.show()
        color = color / len(color_list[i])

        # colors = color[:, skip:-skip]
        # ffts = np.zeros(colors.shape)
        # for j in range(len(ffts)):
        #     ffts[j] = np.abs(np.fft.fftshift(np.fft.fft(color[j])[skip:-skip]))
        #     integrals[i][j] = np.trapz(ffts[j], x=[fftxaxis])
        #     stds[i][j] = np.std(integrals[i][j])

        # compute integral of blue channel

        plt.figure(i)
        plt.scatter(range(n_bins), color[0]/max(color[0]), c='blue')
        plt.scatter(range(n_bins), color[1]/max(color[1]), c='green')
        plt.scatter(range(n_bins), color[2]/max(color[2]), c='red')
        # plt.title('Color Range')
        # plt.figure(i+3)
        # plt.scatter(fftxaxis, ffts[0] / max(ffts[0]), c='blue')
        # plt.scatter(fftxaxis, ffts[1] / max(ffts[1]), c='green')
        # plt.scatter(fftxaxis, ffts[2] / max(ffts[2]), c='red')
        # plt.title('FFTs')
    ffts = np.array(ffts)
    integrals = np.array(integrals)
    coasts_fft = ffts[np.where(labels == 0)]
    coasts_fft_mean = np.mean(coasts_fft, axis=0)
    coasts_fft_std = np.std(coasts_fft, axis=0)
    coasts_integrals = integrals[np.where(labels == 0)]
    coasts_integrals_mean = np.mean(coasts_integrals, axis=0)
    coasts_integrals_std = np.std(coasts_integrals, axis=0)
    forests_fft = ffts[np.where(labels == 1)]
    forests_fft_mean = np.mean(forests_fft, axis=0)
    forests_fft_std = np.std(forests_fft, axis=0)
    forests_integrals = integrals[np.where(labels == 1)]
    forests_integrals_mean = np.mean(forests_integrals, axis=0)
    forests_integrals_std = np.std(forests_integrals, axis=0)
    streets_fft = ffts[np.where(labels == 2)]
    streets_fft_mean = np.mean(streets_fft, axis=0)
    streets_fft_std = np.std(streets_fft, axis=0)
    streets_integrals = integrals[np.where(labels == 2)]
    streets_integrals_mean = np.mean(streets_integrals, axis=0)
    streets_integrals_std = np.std(streets_integrals, axis=0)

    data = {
        "Coasts": coasts_integrals_mean,
        "Forests": forests_integrals_mean,
        "Streets": streets_integrals_mean
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=0.8, single_width=0.9,
             xbar=['R', 'G', 'B'], stache=[coasts_integrals_std, forests_integrals_std, streets_integrals_std])

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
def correlate2d(data: np.array):
    for dim in data.shape()[0]:  # It should be an array/list of ndim x ndata
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
        img = cv.imread(os.path.join(directory, filename))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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

#######################################
#   Main
#######################################


def main():
    # Data load
    normalized = False
    dataset_path = './problematique/baseDeDonneesImages'
    dataset_short_path = "./problematique/test/"
    images, labels = load_images(dataset_short_path, normalize=normalized)

    images_mat = np.zeros(len(labels))
    if normalized:
        images_mat = list(map(itemgetter(0), images))
    else:
        images_mat = images

    features = []
    # Features
    if False:
        canny = extract_high_freq_entropy(
            images_mat, labels, sigma=1, display=True)
        features.append(canny)
    if True:
        color_hist = extract_color_histogram(images_mat, labels)
        features.append(color_hist)
    if False:
        simple_stats = extract_simple_stats(images_mat, labels)
        features.append(simple_stats)
    if False:
        skimage_features = extract_skimage_features(
            images_mat, labels, display=True)
        features.append(skimage_features)
    if False:
        entropy = extract_entropy(images_mat, labels)
        features.append(entropy)
    if False:
        hsv = convert_to_hsv(images_mat, labels, display=True)
        features.append(hsv)
    if False:
        lab = convert_to_lab(images_mat, labels, display=True)
        features.append(lab)
    if False:  # @TODO : make it work, with mean images I guess for the PSNR
        noise = extract_noise_level(images_mat, labels, True)
    if False:
        fft = get_fft(images, labels, True)

    # Correlation
    if False:
        correlate2d(np.array(features))


######################################
if __name__ == '__main__':
    main()
