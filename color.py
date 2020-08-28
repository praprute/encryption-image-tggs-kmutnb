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
#np.set_printoptions(threshold=sys.maxsize)

img = imread('t1.jpg') #original image
#imgs = imread('negative.jpg') #scrambling image
# imgs2 = imread('t3.jpg')  #unscrambling image

org  = resize(img, (1000,1000, 3))
imgs = resize(img, (1000,1000, 3))
# imgs2 = resize(imgs2, (1000,1000, 3))

offset = 8
maxlength = 64

x_div = len(img[0])/offset;
y_div = len(img[1])/offset;

random.seed('Junior')
idx = list(range(maxlength))
random.shuffle(idx)

listNega = [random.randint(0,1) for i in range(maxlength)]
listRandom = [random.randint(0,3) for i in range(maxlength)]
listColor = [random.randint(0,5) for i in range(maxlength)]

#----------------------------------------Encrypt------------------------------------------------------------------------------------------------------------------------------

for i in idx:
    x0_en = (math.floor(x_div) * int(math.floor(idx[i]%offset)))
    x1_en = (math.floor(x_div) * int(math.floor(idx[i]%offset)+1))
    y0_en = (math.floor(y_div) * (math.floor(idx[i]/offset)))
    y1_en = (math.floor(y_div)* (math.floor(idx[i]/offset)+1))
    if(listColor[i] == 0):
        imgs[ x0_en : x1_en, y0_en : y1_en , 0] = org[ x0_en : x1_en, y0_en : y1_en , 1]
        imgs[ x0_en : x1_en, y0_en : y1_en , 1] = org[ x0_en : x1_en, y0_en : y1_en , 0]
    elif(listColor[i] == 1):
        imgs[ x0_en : x1_en, y0_en : y1_en , 0] = org[ x0_en : x1_en, y0_en : y1_en , 2]
        imgs[ x0_en : x1_en, y0_en : y1_en , 2] = org[ x0_en : x1_en, y0_en : y1_en , 0]
    elif(listColor[i] == 2):
        imgs[ x0_en : x1_en, y0_en : y1_en , 1] = org[ x0_en : x1_en, y0_en : y1_en , 2]
        imgs[ x0_en : x1_en, y0_en : y1_en , 2] = org[ x0_en : x1_en, y0_en : y1_en , 1]
    elif(listColor[i] == 3):
        imgs[ x0_en : x1_en, y0_en : y1_en , 0] = org[ x0_en : x1_en, y0_en : y1_en , 1]
        imgs[ x0_en : x1_en, y0_en : y1_en , 1] = org[ x0_en : x1_en, y0_en : y1_en , 2]
        imgs[ x0_en : x1_en, y0_en : y1_en , 2] = org[ x0_en : x1_en, y0_en : y1_en , 0]
    elif(listColor[i] == 4):
        imgs[ x0_en : x1_en, y0_en : y1_en , 0] = org[ x0_en : x1_en, y0_en : y1_en , 2]
        imgs[ x0_en : x1_en, y0_en : y1_en , 1] = org[ x0_en : x1_en, y0_en : y1_en , 0]
        imgs[ x0_en : x1_en, y0_en : y1_en , 2] = org[ x0_en : x1_en, y0_en : y1_en , 1]
    

# for j in idx:
#     x0_de = (math.floor(x_div) * int(math.floor(idx[j]%offset)))
#     x1_de = (math.floor(x_div) * int(math.floor(idx[j]%offset)+1))
#     y0_de = (math.floor(y_div) * (math.floor(idx[j]/offset)))
#     y1_de = (math.floor(y_div)* (math.floor(idx[j]/offset)+1))
#     if(listColor[j] == 0):
#         imgs[ x0_de : x1_de , y0_de : y1_de , 0] = org[ x0_de : x1_de , y0_de : y1_de , 1]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 1] = org[ x0_de : x1_de , y0_de : y1_de , 0]
#     elif(listColor[j] == 1):
#         imgs[ x0_de : x1_de , y0_de : y1_de , 0] = org[ x0_de : x1_de , y0_de : y1_de , 2]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 2] = org[ x0_de : x1_de , y0_de : y1_de , 0]
#     elif(listColor[j] == 2):
#         imgs[ x0_de : x1_de , y0_de : y1_de , 1] = org[ x0_de : x1_de , y0_de : y1_de , 2]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 2] = org[ x0_de : x1_de , y0_de : y1_de , 1]
#     elif(listColor[j] == 3):
#         imgs[ x0_de : x1_de , y0_de : y1_de , 0] = org[ x0_de : x1_de , y0_de : y1_de , 2]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 1] = org[ x0_de : x1_de , y0_de : y1_de , 0]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 2] = org[ x0_de : x1_de , y0_de : y1_de , 1]
#     elif(listColor[j] == 4):
#         imgs[ x0_de : x1_de , y0_de : y1_de , 0] = org[ x0_de : x1_de , y0_de : y1_de , 1]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 1] = org[ x0_de : x1_de , y0_de : y1_de , 2]
#         imgs[ x0_de : x1_de , y0_de : y1_de , 2] = org[ x0_de : x1_de , y0_de : y1_de , 0]
        
#----------------------------------------Encrypt------------------------------------------------------------------------------------------------------------------------------

plt.imshow(imgs)
plt.imsave("dc2.jpg", imgs)
print("---------------------- Finish ----------------------")





