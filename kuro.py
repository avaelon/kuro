#MIT License
#Project initiated by Avaelon Technologies Sdn. Bhd. & Ilangkhumanan Thirunavokkarasu

import pandas as pd
import numpy as np
import plotly as plt
import matplotlib as mpl
import tensorflow as tf

# -*- coding: utf-8 -*-
""" 
Created on 8/10/2021 9:51 PM
@author  : Kuro Kien
@File name    : utils.py 
"""

import random

import cv2
import splitfolders


def split_train_val_test(input_path, output_path, ratio):
    '''This function to split data from class folder, if train,
    val ratio just(x,y)'''
    splitfolders.ratio(input_path, output_path, seed=1337, ratio=ratio)


def fill(img, h, w):
    '''This function to resize image'''

    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def horizontal_shift(img, ratio=0.0):
    '''This function to apply horizontal shift method'''

    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = fill(img, h, w)
    return img


def vertical_shift(img, ratio=0.0):
    '''This function to apply vertical shift method'''

    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
    img = fill(img, h, w)
    return img


def brightness(img, low, high):
    '''This function to apply brightness from low to high method'''

    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def zoom(img, value):
    '''This function to apply zoom method'''

    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value * h)
    w_taken = int(value * w)
    h_start = random.randint(0, h - h_taken)
    w_start = random.randint(0, w - w_taken)
    img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    img = fill(img, h, w)
    return img


def channel_shift(img, value):
    '''This function to apply channel shift method'''

    value = int(random.uniform(-value, value))
    img = img + value
    img[:, :, :][img[:, :, :] > 255] = 255
    img[:, :, :][img[:, :, :] < 0] = 0
    img = img.astype(np.uint8)
    return img


def horizontal_flip(img):
    '''This function to apply horizontal flip method'''
    return cv2.flip(img, 1)


def vertical_flip(img):
    '''This function to apply vertical flip method'''
    return cv2.flip(img, 0)


def rotation(img, angle):
    '''This function to apply rotation from center point'''
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return img

def sharpen(img):
    sharpening = np.array([ [-1,-1,-1],
                            [-1,10,1],
                            [-1,-1,-1]])
    return cv2.filter2D(img, -1, sharpening)
    

