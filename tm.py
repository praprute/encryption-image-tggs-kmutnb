import sys
import random
import os
import matplotlib.image as mping
from PIL import Image
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat as stat
import numpy as np
import numpy.fft
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte
from skimage.transform import SimilarityTransform, warp, swirl, rescale, resize, downscale_local_mean,rotate
from skimage.util import invert, random_noise, montage
import matplotlib.image as mping
import matplotlib.pylab as plt
import torch
import torch_dct as dct
from scipy.ndimage import affine_transform, zoom
from scipy import misc
from scipy.fftpack import dct, idct
import math
import copy
import cv2
import time


img = imread('c.jpg')
img = resize(img, (1000,1000, 3))

offset = 10
maxlength = 100
x_div = len(img[0])/offset
y_div = len(img[1])/offset


i = 20
random.seed('Junior')
idx = list(range(maxlength))
random.shuffle(idx)
print("i = %s" %i)
print("x_div : %s" %(x_div))
print("y_div : %s" %(y_div))

print("x1  : %s" %(math.floor(x_div) * (math.floor(i%offset))))
print("x2  : %s" %(math.floor(x_div) * (math.floor(i%offset)+1)))
print("y1  : %s" %(math.floor(y_div) * (math.floor(i/offset))))
print("y2  : %s" %(math.floor(y_div) * (math.floor(i/offset)+1)))
print("x11 : %s" %(math.floor(x_div) * (math.floor(idx[i]%offset))))
print("x22 : %s" %(math.floor(x_div) * (math.floor(idx[i]%offset)+1)))
print("y11 : %s" %(math.floor(y_div) * (math.floor(idx[i]/offset))))
print("y22 : %s" %(math.floor(y_div) * (math.floor(idx[i]/offset)+1)))

print(idx)
# print((math.floor(i%offset)+1))
# print((math.floor(i/offset)+1))
