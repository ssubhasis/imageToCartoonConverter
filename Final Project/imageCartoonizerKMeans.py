#!/usr/bin/env python
import time
import numpy as np
from collections import defaultdict
from scipy import stats
import cv2
from matplotlib import pyplot as plt


def k_histogram(hist):
    """
    choose the best K for k-means and get the centroids
    """
    alpha = 0.001  # p-value threshold
    N = 80  # minimum group size
    centroid = np.array([128])

    while True:
        centroid, groups = update_centroid(centroid, hist)

        # increase K
        new_centroid = set()  # use set to avoid duplicate in centroid
        for i, indice in groups.items():
            # if there are not enough values in the group, do not separate
            if len(indice) < N:
                new_centroid.add(centroid[i])
                continue

            # separate the centroid if the values of the group is under a normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                # not a normal dist, separate to different centroids
                left = 0 if i == 0 else centroid[i - 1]
                right = len(hist) - 1 if i == len(centroid) - 1 else centroid[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (centroid[i] + left) / 2
                    c2 = (centroid[i] + right) / 2
                    new_centroid.add(c1)
                    new_centroid.add(c2)
                else:
                    # not a normal dist, but no extra space to separate
                    new_centroid.add(centroid[i])
            else:
                # normal dist, no need to separate
                new_centroid.add(centroid[i])
        if len(new_centroid) == len(centroid):
            break
        else:
            centroid = np.array(sorted(new_centroid))
    return centroid


def update_centroid(centroid, hist):
    """
    update centroids until they don't change
    """
    while True:
        groups = defaultdict(list)
        # assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(centroid - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_centroid = np.array(centroid)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_centroid[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))
        if np.sum(new_centroid - centroid) == 0:
            break
        centroid = new_centroid
    return centroid, groups


def smooth_image(image):
    smoothed_image = cv2.bilateralFilter(image, 5, 50, 50)
    for i in range(1, 3):
        smoothed_image = cv2.bilateralFilter(smoothed_image, 5, 50, 50)
    cv2.imwrite("imageCartoonizerKMeans_smoothed.jpg", smoothed_image)
    return smoothed_image


def get_edges(image):
    edge_image = cv2.Canny(image, 100, 200)
    cv2.imwrite("imageCartoonizerKMeans_edges.jpg", edge_image)
    return edge_image


def get_histogram(hsv_image):
    hists = []
    # H channel
    hist, _ = np.histogram(hsv_image[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    # plt.hist(hsv_image[:, :, 0].ravel(), bins=np.arange(180 + 1))
    # plt.show()

    # S channel
    hist, _ = np.histogram(hsv_image[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    # plt.hist(hsv_image[:, :, 0].ravel(), bins=np.arange(256 + 1))
    # plt.show()

    # V channel
    hist, _ = np.histogram(hsv_image[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)
    # plt.hist(hsv_image[:, :, 0].ravel(), bins=np.arange(256 + 1))
    # plt.show()

    return hists


def get_centroids(hists):
    centroids = []
    for h in hists:
        centroids.append(k_histogram(h))
    print("centroids: {0}".format(centroids))
    return centroids


def combine_image(edge, image):
    contours, _ = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, 0, thickness=1)
    return image


def invert(image):
    inverted_image = abs(255 - image)
    # cv2.imwrite("imageCartoonizerKMeans_inverted.jpg", inverted_image)
    return inverted_image


def cartoonize(image):
    """
    convert image into cartoon-like image using K means

    image: input image to be cartoonize
    """

    res_image = np.array(image)
    x, y, c = res_image.shape

    res_image = smooth_image(res_image)
    edge = get_edges(res_image)

    res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2HSV)
    hists = get_histogram(res_image)
    centroids = get_centroids(hists)

    res_image = res_image.reshape((-1, c))
    for i in range(c):
        channel = res_image[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - centroids[i]), axis=1)
        res_image[:, i] = centroids[i][index]
    res_image = res_image.reshape((x, y, c))
    res_image = cv2.cvtColor(res_image, cv2.COLOR_HSV2RGB)
    cv2.imwrite("imageCartoonizerKMeans_intermediate.jpg", res_image)

    res_image = combine_image(edge, res_image)
    return res_image


if __name__ == '__main__':
    """ 
    run as >>python imageCartoonizerKMeans.py
    """
    image = cv2.imread("capturedRawImage.jpg")

    start_time = time.time()
    converted_toon_image = cartoonize(image)
    end_time = time.time()
    t = end_time - start_time
    print('Execution time for imageCartoonizerKMeans : {0}s'.format(t))

    cv2.imwrite("imageCartoonizerKMeans_final.jpg", converted_toon_image)

    inverted_toon_image = invert(converted_toon_image)
    cv2.imwrite("imageCartoonizerKMeans_final_inverted.jpg", inverted_toon_image)
