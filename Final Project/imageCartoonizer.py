#!/usr/bin/env python
import cv2
import numpy as np
import time


def invert(image):
    inverted_image = abs(255 - image)
    # cv2.imwrite("imageCartoonizer_inverted.jpg", inverted_image)
    return inverted_image


def median_filter(image):
    blurred_image = cv2.medianBlur(image, 5)
    cv2.imwrite("imageCartoonizer_median_blurred.jpg", blurred_image)
    return blurred_image


def get_edges(image):
    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gs_image, 70, 140)
    # cv2.imwrite("imageCartoonizer_edges.jpg", edges)
    kernel = np.ones((2, 2), np.int)
    edge_image = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite("imageCartoonizer_edges.jpg", edge_image)
    return edge_image


def smooth_image(image):
    smoothed_image = cv2.bilateralFilter(image, 20, 10, 20)
    for i in range(1, 5):
        smoothed_image = cv2.bilateralFilter(smoothed_image, 20, 10, 20)
    cv2.imwrite("imageCartoonizer_smoothed.jpg", smoothed_image)
    return smoothed_image


def quantize(image):
    quantized_image = np.copy(image)
    for i in range(1, len(quantized_image)):
        for j in range(1, len(quantized_image[0])):
            quantized_image[i][j] = quantized_image[i][j] / 10 * 10 + 10 / 2
    cv2.imwrite("imageCartoonizer_quantized.jpg", quantized_image)
    return quantized_image


def combine_edges(converted_image, edge_img):
    res_image = np.copy(converted_image)
    for i in range(1, len(res_image)):
        for j in range(1, len(res_image[0])):
            if edge_img[i][j] > 0:
                res_image[i][j] = [0, 0, 0]
    return res_image


def cartoonize(image):
    """
    convert image into cartoon-like image

    image: input image to be cartoonize
    """
    img = np.copy(image)
    res_image = median_filter(img)
    edges = get_edges(res_image)
    res_image = smooth_image(res_image)
    res_image = median_filter(res_image)
    res_image = quantize(res_image)
    res_image = combine_edges(res_image, edges)
    return res_image


if __name__ == '__main__':
    """ 
    run as >>python imageCartoonizer.py capturedRawImage.jpg
    """
    image = cv2.imread("capturedRawImage.jpg")

    start_time = time.time()
    converted_toon_image = cartoonize(image)
    end_time = time.time()
    t = end_time - start_time
    print('Execution time for imageCartoonizer : {0}s'.format(t))

    cv2.imwrite("imageCartoonizer_final.jpg", converted_toon_image)

    inverted_toon_image = invert(converted_toon_image)
    cv2.imwrite("imageCartoonizer_final_inverted.jpg", inverted_toon_image)
