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
from skimage import img_as_float, color, viewer, exposure, data
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



img = imread('t1.jpg') #original image
#imgs = imread('t1.jpg') #scrambling image
imgs2 = imread('t3.jpg')  #unscrambling image

img = resize(img, (1000,1000, 3))
imgs = resize(img, (1000,1000, 3))
imgs2 = resize(imgs2, (1000,1000, 3))

x_div = len(img[0])/10;
y_div = len(img[1])/10;

offset = 10

random.seed(1)
idx = list(range(100))
random.shuffle(idx)

listRandom = [random.randint(0,3) for i in range(100)]
listposi = [random.randint(0,1) for j in range(100)]

print(idx)
print('----------------------------------------------')
print(listRandom)
print('-------Posi---------------------------------------')
print(listposi)
#print(len(listRandom))


for i in idx:

    piece = img[(math.floor(x_div) * int(math.floor(i%offset))) : (math.floor(x_div) * int(math.floor(i%offset)+1)), 
    (math.floor(y_div) * (math.floor(i/offset))) : (math.floor(y_div)*(math.floor(i/offset)+1)),:]
    
    rotp = np.rot90(piece, listRandom[i]);

    imgs[(math.floor(x_div) * int(math.floor(idx[i]%offset))) : (math.floor(x_div) * int(math.floor(idx[i]%offset)+1)), 
    (math.floor(y_div) * (math.floor(idx[i]/offset))) : (math.floor(y_div)* (math.floor(idx[i]/offset)+1)),
    :] = rotp




# plt.imsave("tu5.jpg", imgs)

for j in idx:
    piece2 = imgs[(math.floor(x_div) * int(math.floor(idx[j]%offset))) : (math.floor(x_div) * int(math.floor(idx[j]%offset)+1)), 
    (math.floor(y_div) * (math.floor(idx[j]/offset))) : (math.floor(y_div)* (math.floor(idx[j]/offset)+1)),
    :]

    rotp2 = np.rot90(piece2, 4-listRandom[j]);

    imgs2[(math.floor(x_div) * int(math.floor(j%offset))) : (math.floor(x_div) * int(math.floor(j%offset)+1)), 
    (math.floor(y_div) * (math.floor(j/offset))) : (math.floor(y_div)* (math.floor(j/offset)+1)),:] = rotp2


print(imgs.dtype)

#imsave("rotateDecrypt.jpg", imgs2)
plt.imshow(imgs)



