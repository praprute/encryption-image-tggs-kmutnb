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
# np.set_printoptions(threshold=sys.maxsize)

img = 't1.jpg'
img1 = cv2.imread(img, 1)
imgYCC = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
h , w , b = np.array(imgYCC.shape[:])
print(imgYCC.shape)
print(h , w , b)
vis0    = np.zeros((h,w,3), np.double)
Trans   = np.zeros((h,w*3), np.double)
vis0[:h, :w, :b]  = imgYCC
Trans[:h, :w] = vis0[:h, :w, 0]
Trans[:h, w:w*2] = vis0[:h, :w, 1]
Trans[:h, w*2:w*3] = vis0[:h, :w, 2]
print('---------- Vis0 ----------------------------------')
print(vis0.shape)
print('---------- Trans ----------------------------------')
print(Trans.shape)

pixel = 8
bx = math.floor(Trans.shape[1]/pixel)
by = math.floor(Trans.shape[0]/pixel) 
allbloc = bx*by
random.seed('junior') 
rp = list(range(1,allbloc+1)) 
random.shuffle(rp) 
# rp = random.randint(1, allbloc)
# rp = np.random.permutation(allbloc)
print(allbloc)
print(bx)
print(by)
print(len(rp))
print(max(rp))
print(min(rp))
print("----------------------")
# print(rp)
scram = np.zeros((h,w*3), np.double)
global i 
i = 0

for r in range(1,by):
    r2 = (r-1)*pixel+1
    print(r2)
    for c in range(1,bx):
        c2 = (c-1)*pixel+1
        print(c2)
        index = rp[i]
        print(index)
        rr = math.floor((index-1)/bx)*pixel + 1
        print(rr)
        cc = ((index-1)%bx)*pixel + 1
        print(cc)
        scram[r2-1:r2+pixel-2, c2-1:c2+pixel-2] = Trans[rr-1:rr+pixel-2, cc-1:cc+pixel-2]
        i = i+1

cv2.imwrite('v0.jpg', vis0)
cv2.imwrite('Trans.jpg', Trans)
cv2.imwrite('scram.jpg', scram)
cv2.imwrite('YCBCR.jpg', imgYCC)
cv2.waitKey(10000)