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
np.set_printoptions(threshold=sys.maxsize)

img = imread('e1.jpg') #original image
imgs2 = imread('r1.jpg')  

print(img.shape)
start_time = time.time()
img   = resize(img, (1000,1000, 3))
imgs  = resize(img, (1000,1000, 3))
imgs2 = resize(imgs2, (1000,1000, 3))

# img   = img
# imgs  = img
# imgs2 = imgs2
# out  = org
# imgs2 = imgs2

# org   = img_as_ubyte(org)
# out   = img_as_ubyte(out)
# imgs2 = img_as_ubyte(imgs2)
# bsy = 8
# bsx = 8
# [sy, sx, c] = org.shape
# syd = math.floor(sy / bsy)
# sxd = math.floor(sx / bsx)
# print(syd)
# print(sxd)
# print(c)
# print(imgs2[0:100, 0:100, :])

# offset = 125 #จำนวนบล็อคในแต่ละแนว 
# maxlength = 15625 #จำนวนบล็อคทั้งหมด
offset = 10
maxlength = 100
x_div = len(img[0])/offset #ขนาดของบล็อค 1 บล็อคในแกน x
y_div = len(img[1])/offset #ขนาดของบล็อค 1 บล็อคในแกน y

random.seed(1) #yoshi
idx = list(range(maxlength)) # key ของแต่ละบล็อค
random.shuffle(idx) 
# print(idx)
listNega = [random.randint(0,1) for i in range(maxlength)] # key ของการทำ negative positive 
listRandom = [random.randint(0,3) for i in range(maxlength)] # key ของการทำ rotage
listColor = [random.randint(0,5) for i in range(maxlength)] # key ของการทำสลับที่ rgb
def encrypt_rgb():
    for i in idx:
        x0_en = (math.floor(x_div) * int(math.floor(i%offset)))
        x1_en = (math.floor(x_div) * int(math.floor(i%offset)+1))
        y0_en = (math.floor(y_div) * (math.floor(i/offset)))
        y1_en = (math.floor(y_div) * (math.floor(i/offset)+1))
        if(listColor[i] == 0):
            imgsC[ x0_en : x1_en, y0_en : y1_en , 0] = orgC[ x0_en : x1_en, y0_en : y1_en , 1]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 1] = orgC[ x0_en : x1_en, y0_en : y1_en , 0]
        elif(listColor[i] == 1):
            imgsC[ x0_en : x1_en, y0_en : y1_en , 0] = orgC[ x0_en : x1_en, y0_en : y1_en , 2]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 2] = orgC[ x0_en : x1_en, y0_en : y1_en , 0]
        elif(listColor[i] == 2):
            imgsC[ x0_en : x1_en, y0_en : y1_en , 1] = orgC[ x0_en : x1_en, y0_en : y1_en , 2]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 2] = orgC[ x0_en : x1_en, y0_en : y1_en , 1]
        elif(listColor[i] == 3):
            imgsC[ x0_en : x1_en, y0_en : y1_en , 0] = orgC[ x0_en : x1_en, y0_en : y1_en , 1]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 1] = orgC[ x0_en : x1_en, y0_en : y1_en , 2]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 2] = orgC[ x0_en : x1_en, y0_en : y1_en , 0]
        elif(listColor[i] == 4):
            imgsC[ x0_en : x1_en, y0_en : y1_en , 0] = orgC[ x0_en : x1_en, y0_en : y1_en , 2]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 1] = orgC[ x0_en : x1_en, y0_en : y1_en , 0]
            imgsC[ x0_en : x1_en, y0_en : y1_en , 2] = orgC[ x0_en : x1_en, y0_en : y1_en , 1]
def decrypt_rgb():
    global orgC  
    global imgsC 
    orgC  = resize(img, (1000,1000, 3))
    imgsC = resize(img, (1000,1000, 3)) 
    for j in idx:
        x0_de = (math.floor(x_div) * int(math.floor(j%offset)))
        x1_de = (math.floor(x_div) * int(math.floor(j%offset)+1))
        y0_de = (math.floor(y_div) * (math.floor(j/offset)))
        y1_de = (math.floor(y_div) * (math.floor(j/offset)+1))
        if(listColor[j] == 0):
            imgsC[ x0_de : x1_de , y0_de : y1_de , 0] = orgC[ x0_de : x1_de , y0_de : y1_de , 1]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 1] = orgC[ x0_de : x1_de , y0_de : y1_de , 0]
        elif(listColor[j] == 1):
            imgsC[ x0_de : x1_de , y0_de : y1_de , 0] = orgC[ x0_de : x1_de , y0_de : y1_de , 2]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 2] = orgC[ x0_de : x1_de , y0_de : y1_de , 0]
        elif(listColor[j] == 2):
            imgsC[ x0_de : x1_de , y0_de : y1_de , 1] = orgC[ x0_de : x1_de , y0_de : y1_de , 2]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 2] = orgC[ x0_de : x1_de , y0_de : y1_de , 1]
        elif(listColor[j] == 3):
            imgsC[ x0_de : x1_de , y0_de : y1_de , 0] = orgC[ x0_de : x1_de , y0_de : y1_de , 2]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 1] = orgC[ x0_de : x1_de , y0_de : y1_de , 0]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 2] = orgC[ x0_de : x1_de , y0_de : y1_de , 1]
        elif(listColor[j] == 4):
            imgsC[ x0_de : x1_de , y0_de : y1_de , 0] = orgC[ x0_de : x1_de , y0_de : y1_de , 1]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 1] = orgC[ x0_de : x1_de , y0_de : y1_de , 2]
            imgsC[ x0_de : x1_de , y0_de : y1_de , 2] = orgC[ x0_de : x1_de , y0_de : y1_de , 0]
#----------------------------------------Encrypt------------------------------------------------------------------------------------------------------------------------------
for i in idx:
    x1 = (math.floor(x_div) * int(math.floor(idx[i]%offset)))
    x2 = (math.floor(x_div) * int(math.floor(idx[i]%offset)+1))
    y1 = (math.floor(y_div) * (math.floor(idx[i]/offset)))
    y2 = (math.floor(y_div)* (math.floor(idx[i]/offset)+1))

    x11 = (math.floor(x_div) * int(math.floor(i%offset)))
    x22 = (math.floor(x_div) * int(math.floor(i%offset)+1))
    y11 = (math.floor(y_div) * (math.floor(i/offset)))
    y22 = (math.floor(y_div)* (math.floor(i/offset)+1))

    piece = img[ x1 : x2 , y1 : y2 , : ]
    rotp = np.rot90(piece, listRandom[i])
    if(listNega[i] == 1):
        rotp = 1 - rotp
    imgs[ x11 : x22 , y11 : y22 , : ] = rotp

# global orgC  
# global imgsC 
# orgC  = resize(imgs, (1000,1000, 3))
# imgsC = resize(imgs, (1000,1000, 3)) 
# encrypt_rgb()
#----------------------------------------Encrypt------------------------------------------------------------------------------------------------------------------------------
    
# for i in idx:
#     if(listNega[i] == 1):      
#         for x in range((math.floor(x_div) * int(math.floor(i%offset))), (math.floor(x_div) * int(math.floor(i%offset)+1))):
#             for y in range((math.floor(y_div) * (math.floor(i/offset))), (math.floor(y_div)*(math.floor(i/offset)+1))):
#                 imgs[x,y] = [1 - imgs[x,y][0], 1 - imgs[x,y][1], 1 - imgs[x,y][2]]            

#----------------------------------------Decrypt------------------------------------------------------------------------------------------------------------------------------
# decrypt_rgb()
# for j in idx:
#     if(listNega[j] == 1):
#         for x in range((math.floor(x_div) * int(math.floor(j%offset))), (math.floor(x_div) * int(math.floor(j%offset)+1))):
#             for y in range((math.floor(y_div) * (math.floor(j/offset))), (math.floor(y_div)*(math.floor(j/offset)+1))):
#                 imgsC[x,y] = [1 - imgsC[x,y][0], 1 - imgsC[x,y][1], 1 - imgsC[x,y][2]]

# for j in idx:
#     x1 = (math.floor(x_div) * int(math.floor(j%offset)))
#     x2 = (math.floor(x_div) * int(math.floor(j%offset)+1))
#     y1 = (math.floor(y_div) * (math.floor(j/offset)))
#     y2 = (math.floor(y_div) * (math.floor(j/offset)+1))

#     x11 = (math.floor(x_div) * int(math.floor(idx[j]%offset)))
#     x22 = (math.floor(x_div) * int(math.floor(idx[j]%offset)+1))
#     y11 = (math.floor(y_div) * (math.floor(idx[j]/offset))) 
#     y22 = (math.floor(y_div) * (math.floor(idx[j]/offset)+1))

#     piece2 = imgsC[ x1 : x2 , y1 : y2 , :]
#     rotp2 = np.rot90(piece2, 4-listRandom[j]);
#     if(listNega[j] == 1):
#         rotp2 = 1 - rotp2
#     imgs[ x11 : x22 , y11 : y22 , :] = rotp2

    
print('--------------------------------------------')
print('--------------------------------------------')

# print(imgs)
#----------------------------------------Decrypt------------------------------------------------------------------------------------------------------------------------------
print(" time processing --- %s seconds ---" % (time.time() - start_time))
# plt.imshow(imgs)
plt.imsave("e2.jpg", imgs)



# i = 1
# for r in range(1,syd):
#     r2 = (r-1)*bsy+1
#     for c in range(1,sxd):
#         c2 = (c-1)*bsx+1
#         rr = math.floor((idx[i]-1) / bsy)*sxd + 1
#         cc = ((idx[i]-1)%bsx) * sxd + 1
#         out[r2:r2+bsy-1, c2:c2+bsx-1, :] = org[rr:rr+bsy-1, cc:cc+bsx-1, :]
#         i = i+1

# for r in range(1,5):
#     r2 = (r-1)*bsy+1
#     # print("r2: %s"%r2)
#     for c in range(1,5):
#         c2 = (c-1)*bsx+1
#         # print("c2: %s"%c2)
#         rr = math.floor((idx[i-1]) / sxd)*bsy + 1
#         print("rr: %s"%rr)
#         cc = ((idx[i-1])% sxd) * bsx + 1
#         print("cc: %s"%cc)
#         out[r2:r2+bsy-1, c2:c2+bsx-1, :] = org[rr:rr+bsy-1, cc:cc+bsx-1, :]
#         i = i+1







