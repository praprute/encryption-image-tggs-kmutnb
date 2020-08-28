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

B = 10
fn3 = 'e2.jpg'
img = cv2.imread(fn3, 1)
# img  = cv2.resize(img1, (100,100))
# cv2.imwrite('resizeT2.jpg', img)
print(np.array(img.shape[:]))
h , w, c = np.array(img.shape[:])/B * B
h = int(h)
w = int(w)
c = int(c)
print('--------size image------------------------------------')
print(h)
print(w)
# print(c)
print('--------------------------------------------')
img = img[:h,:w, :c]
# print(img1)

blocksV =   h/B
blocksV =   int(blocksV)
blocksH =   w/B
blocksH =   int(blocksH)
vis0    =   np.zeros((h,w,c), np.double)
Trans   =   np.zeros((h,w,c), np.double)
vis0[:h, :w, :c] = img
# Trans[:h, :w] = img
print('----------nums blok on image----------------------------------')
print(blocksV)
print(blocksH)
print('--------------------------------------------')

print('----------vis0 Trans image----------------------------------')
print(vis0)
print(Trans)
print('--------------------------------------------')

print('----------image----------------------------------')
# print(img1[:h,:w,:c])
print(img.dtype)
print('--------------------------------------------')

# offset = 10
# maxlength = 15625
# x_div = len(img[0])/offset #ขนาดของบล็อค 1 บล็อคในแกน x
# y_div = len(img[1])/offset #ขนาดของบล็อค 1 บล็อคในแกน y
maxlength = B*B
random.seed('junior') 
idx = list(range(maxlength)) 
random.shuffle(idx) 
listNega = [random.randint(0,1) for i in range(maxlength)] # key ของการทำ negative positive 
listRandom = [random.randint(0,3) for i in range(maxlength)] # key ของการทำ rotage
listColor = [random.randint(0,5) for i in range(maxlength)] # key ของการทำสลับที่ rgb

print('---------- idx ----------------------------------')
# print(idx)
print(range(len(idx)))
print('--------------------------------------------')

# global blockIndex 
# blockIndex = 0
print('------------encryption--------------------------------')
# for j in idx:
#     for row in blocksV:
#         x1 = blocksV*
# for i in range(0,11):
#     print('i : ', i)
#     # print((math.floor(10) * int(math.floor(i%B))))
#     # print((math.floor(10) * int(math.floor(i%B))+1))
#     # print((math.floor(10) * (math.floor(i/B))))
#     # print(math.floor(10) * (math.floor(i/B)+1))
#     print((math.floor(10) * int(math.floor(i%B))),':',(math.floor(10) * int(math.floor(i%B))+1), ',' ,(math.floor(10) * (math.floor(i/B))),':',(math.floor(10) * (math.floor(i/B)+1)))

# global poIndex
# poIndex = 0 

# for row in range(1,blocksH):
#     r2 = (row-1)*B+1
#     for col in range(1,blocksV):
#         c2    = (col-1)*blocksV+1
#         index = idx[poIndex-1]
#         rr    = math.floor((index)/blocksV)*B + 1
#         cc    = ((index)%blocksV)*B + 1
#         Trans[r2:r2+B-1, c2:c2+B-1, :] = vis0[rr:rr+B-1, cc:cc+B-1, :]
#         poIndex = poIndex+1
#     pass
# pass

# for j in idx:
#     x1 = (math.floor(blocksV) * int(math.floor(j%B)))
#     x2 = (math.floor(blocksV) * int(math.floor(j%B)+1))
#     y1 = (math.floor(blocksH) * (math.floor(j/B)))
#     y2 = (math.floor(blocksH) * (math.floor(j/B)+1))

#     x11 = (math.floor(blocksV) * int(math.floor(idx[j]%B)))
#     x22 = (math.floor(blocksV) * int(math.floor(idx[j]%B)+1))
#     y11 = (math.floor(blocksH) * (math.floor(idx[j]/B))) 
#     y22 = (math.floor(blocksH) * (math.floor(idx[j]/B)+1))
#     Trans[ x11 : x22 , y11 : y22 ] = vis0[ x1 : x2 , y1 : y2]
    # rot = np.rot90(org, 4-listRandom[j])
    # if(listNega[j] == 1):
    #     rot = 255 - rot
    # Trans[ x11 : x22 , y11 : y22 , :] = rot

# for j in idx:
    # for row in range(blocksV):
    #     r2 = row*B+1
    #     for col in range(blocksH):
    #         c2    = col*blocksH+1
    #         index = idx[j]
    #         rr    = math.floor((index)/ blocksH)*B + 1
    #         cc    = ((index)%blocksH)*B + 1
    #         Trans[r2:r2+B, c2:c2+B, :] = vis0[rr:rr+B, cc:cc+B, :]
    #         print(col)
print('------------------------------------------------------')
print('------------decryption--------------------------------')
for i in idx:
    x1 = (math.floor(blocksV) * int(math.floor(idx[i]%B)))
    x2 = (math.floor(blocksV) * int(math.floor(idx[i]%B)+1))
    y1 = (math.floor(blocksH) * (math.floor(idx[i]/B)))
    y2 = (math.floor(blocksH) * (math.floor(idx[i]/B)+1))

    x11 = (math.floor(blocksV) * int(math.floor(i%B)))
    x22 = (math.floor(blocksV) * int(math.floor(i%B)+1))
    y11 = (math.floor(blocksH) * (math.floor(i/B)))
    y22 = (math.floor(blocksH) * (math.floor(i/B)+1))
    Trans[ x11 : x22 , y11 : y22  ] = vis0[ x1 : x2 , y1 : y2 ]
#     rot = np.rot90(org, listRandom[i])
#     if(listNega[i] == 1):
#         rot = 255 - rot
#     Trans[ x11 : x22 , y11 : y22 , : ] = rot
print('-------Trans-------------------------------------')
print(Trans)
print('-------ORG image-------------------------------------')
print(img)
cv2.imwrite('d2.jpg', Trans)
cv2.imshow('RGB Image',Trans)
cv2.waitKey(10000)

#def dec():
#    for j in idx:
#        x1 = (math.floor(x_div) * int(math.floor(j%offset)))
#        x2 = (math.floor(x_div) * int(math.floor(j%offset)+1))
#        y1 = (math.floor(y_div) * (math.floor(j/offset)))
#        y2 = (math.floor(y_div) * (math.floor(j/offset)+1))

#        x11 = (math.floor(x_div) * int(math.floor(idx[j]%offset)))
#        x22 = (math.floor(x_div) * int(math.floor(idx[j]%offset)+1))
#        y11 = (math.floor(y_div) * (math.floor(idx[j]/offset))) 
#        y22 = (math.floor(y_div) * (math.floor(idx[j]/offset)+1))

#        piece2 = imgsC[ x1 : x2 , y1 : y2 , :]
#        rotp2 = np.rot90(piece2, 4-listRandom[j]);
#        if(listNega[j] == 1):
#            rotp2 = 1 - rotp2
#        imgs[ x11 : x22 , y11 : y22 , :] = rotp2


# for i in idx:
#     x1 = (math.floor(x_div) * int(math.floor(idx[i]%offset)))
#     x2 = (math.floor(x_div) * int(math.floor(idx[i]%offset)+1))
#     y1 = (math.floor(y_div) * (math.floor(idx[i]/offset)))
#     y2 = (math.floor(y_div)* (math.floor(idx[i]/offset)+1))

#     x11 = (math.floor(x_div) * int(math.floor(i%offset)))
#     x22 = (math.floor(x_div) * int(math.floor(i%offset)+1))
#     y11 = (math.floor(y_div) * (math.floor(i/offset)))
#     y22 = (math.floor(y_div)* (math.floor(i/offset)+1))

#     piece = img[ x1 : x2 , y1 : y2 , : ]
#     rotp = np.rot90(piece, listRandom[i])
#     if(listNega[i] == 1):
#         rotp = 1 - rotp
#     imgs[ x11 : x22 , y11 : y22 , : ] = rotp



# cv2.imshow('RGB Image',img1)
# cv2.waitKey(5000)
# plt.imsave("e3.jpg", img1)

# [ 31 197 255]
# [ 19 191 255]
# [ 36 194 255]

# [  0 172 235]
# [  0 173 236]
# [  1 170 235]

# [  0 172 235] #encrypt

# [ 31 197 255] #decrypt
