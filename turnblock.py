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
imgs = imread('t1.jpg') #scrambling image
imgs2 = imread('t2.jpg')  #unscrambling image

img = resize(img, (1000,1000))
imgs = resize(img, (1000,1000))
imgs2 = resize(imgs2, (1000,1000))

x_div = len(img[0])/10;
y_div = len(img[1])/10;

random.seed(1)
idx = list(range(100))
random.shuffle(idx)

listRandom = [random.randint(0,3) for i in range(100)]

# print(idx)
# print('----------------------------------------------')
# print(listRandom)
# print(len(listRandom))


# for i in idx:

#     piece = img[(math.floor(x_div) * int(math.floor(i%10))) : (math.floor(x_div) * int(math.floor(i%10)+1)), 
#     (math.floor(y_div) * (math.floor(i/10))) : (math.floor(y_div)* (math.floor(i/10)+1)),:]
    
#     rotp = np.rot90(piece, listRandom[i]);

#     imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#     :] = rotp

# for i in idx:
#     piece = img[(math.floor(x_div) * int(math.floor(i%10))) : (math.floor(x_div) * int(math.floor(i%10)+1)), 
#     (math.floor(y_div) * (math.floor(i/10))) : (math.floor(y_div)* (math.floor(i/10)+1)),:]
    
#     if(listRandom[i] == 1 or listRandom[i] == 3):
#         turnp = np.flipud(piece);
#         imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#         (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#         :] = turnp
#     if(listRandom[i] > 1):
#         turnp = np.fliplr(piece);
#         imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#         (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#         :] = turnp

# for j in idx:
#     piece2 = imgs[(math.floor(x_div) * int(math.floor(idx[j]%10))) : (math.floor(x_div) * int(math.floor(idx[j]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[j]/10))) : (math.floor(y_div)* (math.floor(idx[j]/10)+1)),
#     :]

#     rotp2 = np.rot90(piece2, 4-listRandom[j]);

#     imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#     (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = rotp2


# for j in idx:
#     piece2 = imgs[(math.floor(x_div) * int(math.floor(idx[j]%10))) : (math.floor(x_div) * int(math.floor(idx[j]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[j]/10))) : (math.floor(y_div)* (math.floor(idx[j]/10)+1)),
#     :]

#     if(listRandom[j] == 1 or listRandom[j] == 3):
#         returnp = np.flipud(piece2);
#         imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#         (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = returnp
    
#     if(listRandom[j] > 1):
#         returnp = np.fliplr(piece2);
#         imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#         (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = returnp

plt.imshow(imgs2)


# for i in idx:

#     piece = img[(math.floor(x_div) * int(math.floor(i%10))) : (math.floor(x_div) * int(math.floor(i%10)+1)), 
#     (math.floor(y_div) * (math.floor(i/10))) : (math.floor(y_div)* (math.floor(i/10)+1)),:]
    
#     rotp = np.rot90(piece, listRandom[i]);

#     imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#     :] = rotp

# for i in idx:
#     piece = img[(math.floor(x_div) * int(math.floor(i%10))) : (math.floor(x_div) * int(math.floor(i%10)+1)), 
#     (math.floor(y_div) * (math.floor(i/10))) : (math.floor(y_div)* (math.floor(i/10)+1)),:]
    
#     if(listRandom[i] == 1 or listRandom[i] == 3):
#         turnp = np.flipud(piece);
#         imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#         (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#         :] = turnp
#     if(listRandom[i] > 1):
#         turnp = np.fliplr(piece);
#         imgs[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#         (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),
#         :] = turnp



# for j in idx:

#     piece2 = imgs[(math.floor(x_div) * int(math.floor(idx[j]%10))) : (math.floor(x_div) * int(math.floor(idx[j]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[j]/10))) : (math.floor(y_div)* (math.floor(idx[j]/10)+1)),
#     :]

#     rotp2 = np.rot90(piece2, 4-listRandom[j]);

#     imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#     (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = rotp2

# for j in idx:
#     piece2 = imgs[(math.floor(x_div) * int(math.floor(idx[j]%10))) : (math.floor(x_div) * int(math.floor(idx[j]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[j]/10))) : (math.floor(y_div)* (math.floor(idx[j]/10)+1)),
#     :]

#     if(listRandom[j] == 1 or listRandom[j] == 3):
#         returnp = np.flipud(piece2);
#         imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#         (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = returnp
    
#     if(listRandom[j] > 1):
#         returnp = np.fliplr(piece2);
#         imgs2[(math.floor(x_div) * int(math.floor(j%10))) : (math.floor(x_div) * int(math.floor(j%10)+1)), 
#         (math.floor(y_div) * (math.floor(j/10))) : (math.floor(y_div)* (math.floor(j/10)+1)),:] = returnp