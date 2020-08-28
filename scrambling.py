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
from skimage.transform import SimilarityTransform, warp, swirl, rescale, resize, downscale_local_mean
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

def operation():
    return sys.argv[1]

def openImage():
    img = imread('t1.jpg') #original image
    imgs = imread('t1.jpg') #scrambling image
    imgs2 = imread('t2.jpg')  #unscrambling image
    print("original image shape: ", img.shape)
    print("scrambling image shape: ", imgs.shape)
    print("unscrambling image shape: ", imgs2.shape)


def resize():
    openImage()
    img = resize(img, (1000,1000))
    imgs = resize(img, (1000,1000))
    imgs2 = resize(imgs2, (1000,1000))
    print("resize original image shape: ", img.shape)
    print("resize scrambling image shape: ", img.shape)
    print("resize unscrambling image shape: ", img.shape)

def compareBlock():
    openImage()
    x_div = len(imgs2[0])/100;
    y_div = len(imgs2[1])/100;

def seed():
    random.seed(1)

def scramblingIndex():
    idx = list(range(10000))
    random.shuffle(idx)
    return idx

def Blockscrambling():
    openImage()
    seed()
    idx = scramblingIndex()
    for i in idx:
        imgs[(math.floor(x_div) * int(math.floor(idx[i]%100))) : (math.floor(x_div) * int(math.floor(idx[i]%100)+1)), 
        (math.floor(y_div) * (math.floor(idx[i]/100))) : (math.floor(y_div)* (math.floor(idx[i]/100)+1)),
        :] = img[(math.floor(x_div) * int(math.floor(i%100))) : (math.floor(x_div) * int(math.floor(i%100)+1)), 
        (math.floor(y_div) * (math.floor(i/100))) : (math.floor(y_div)* (math.floor(i/100)+1)),:]

    imsave("blockScrambling.jpg", imgs)

def unBlockscrambling():
    openImage()
    seed()
    idx = scramblingIndex()
    for i in idx:
        imgs2[(math.floor(x_div) * int(math.floor(i%100))) : (math.floor(x_div) * int(math.floor(i%100)+1)), 
        (math.floor(y_div) * (math.floor(i/100))) : (math.floor(y_div)* (math.floor(i/100)+1)),:] = imgs[(math.floor(x_div) * int(math.floor(idx[i]%100))) : (math.floor(x_div) * int(math.floor(idx[i]%100)+1)), 
        (math.floor(y_div) * (math.floor(idx[i]/100))) : (math.floor(y_div)* (math.floor(idx[i]/100)+1)),
        :]
    imsave("unblockScrambling.jpg", imgs2)

def main():


if __name__ == "__main__":
    main()


# for i in range(len(imgs[0])):
#     for j in range(len(imgs[1])):
#         if((math.floor(i/x_div)%2)==1):
#             if((math.floor(j/y_div)%2)==0):
#                 imgs1[i, j, :] = 0
#         if((math.floor(i/x_div)%2)==0):
#             if((math.floor(j/y_div)%2)==1):
#                 imgs1[i, j, :] = 0

# for h in range(len(imgsB[0])):
#     for k in range(len(imgsB[1])):
#         if((math.floor(h/x_divB)%2)==1):
#             if((math.floor(k/y_divB)%2)==0):
#                 imgs2[h, k, :] = 0
#         if((math.floor(h/x_divB)%2)==0):
#             if((math.floor(k/y_divB)%2)==1):
#                 imgs2[h, k, :] = 0


# for i in range(100):
#     imgs2[(math.floor(x_div) * int(math.floor(idx[i]%10))) : (math.floor(x_div) * int(math.floor(idx[i]%10)+1)), 
#     (math.floor(y_div) * (math.floor(idx[i]/10))) : (math.floor(y_div)* (math.floor(idx[i]/10)+1)),:] = img[(math.floor(x_div) * int(math.floor(i%10))) : (math.floor(x_div) * int(math.floor(i%10)+1)), 
#     (math.floor(y_div) * (math.floor(i/10))) : (math.floor(y_div)* (math.floor(i/10)+1)),:]


#imgPIL = Image.open('t1.jpg')




# print('--------------------------------------------')
# random.seed(y_div*x_div)

# idx = list(range(len(imgs1)))

# random.shuffle(idx)
# print(imgs1)

# out = []
# for i in idx:
#         out.append(imgs1[i])
        

# outImg = Image.new("RGB", imgPIL.size)
# w, h = imgPIL.size
# pxIter = iter(out)
# for x in range(w):
#     for y in range(h):
#         outImg.putpixel((x, y), next(pxIter))
# outImg.save("new.png")


# print('-------------out-------------------------------')

# print(out)


# #print(imgs1)
# print(imgs1.shape)

# #plt.imshow(imgs1)
# plt.imshow(imgs1)

# plt.show()

# def seed(img):
#     random.seed(hash(img.size))

# def getPixels(img):
#     w, h = img.size
#     pxs = []
#     for x in range(w):
#         for y in range(h):
#             pxs.append(img.getpixel((x, y)))
#     return pxs

# def scrambledIndex(pxs):
#     idx = list(range(len(pxs)))
#     random.shuffle(idx)
#     return idx
    
# def scramblePixels(img):
#     seed(img)
#     pxs = getPixels(img)
#     idx = scrambledIndex(pxs)
#     out = []
#     for i in idx:
#         out.append(pxs[i])
#     return out

# def storePixels(name, size, pxs):
#     outImg = Image.new("RGB", size)
#     w, h = size
#     pxIter = iter(pxs)
#     for x in range(w):
#         for y in range(h):
#             outImg.putpixel((x, y), next(pxIter))
#     outImg.save(name)

# for j in range(imgs.shape[0]):
#    if(j%2 == 0):
#        for i in range(100):
#            if(i%2 == 0):
#                imgs[j:100, j:100, :] = [0 ,0 ,0]


# n = 500 
# x,y = np.random.randint(0, img.width, n),
# np.random.randint(0, img.height, n)

# for (x,y) in zip(x,y):
#     img.putpixel((x,y), ((0,0,0) if np.random.rand()<0.5 else (255,255,255)))

# img.show()

# im = mping.imread('t1.jpg')
# print(im.shape, im.dtype, type(im))
# plt.imshow(im)
# plt.axis('off')
# plt.show()

# img = Image.open('t2.png')
# n = 5000
# x, y = np.random.randint(0, img.width, n), np.random.randint(0, img.height, n)
# for (x,y) in zip(x,y):
#     img.putpixel((x,y), ((0:10,0:10,0) if np.random.rand()<0.5
#     else (255,255,255)))
# img.show()

#    xแนวตั้ง  yแนนอน  
#imgs[0:1000 , 0:100, :] = [0, 0, 0]

#img_read[1::2, 0::2] = 0
#img_read[0::2, 1::2] = 0

#for x in range(w):
#    for y in range(100): 
#        img_read[x:y, x:y] =  [0, 0, 0]
#        print(x, y)

#for i in range(w):
#    for j in imgs[i]:
#        print(i,j)
#        if(i%2 == 0):
#            imgs[i:i+100, i:i+100, :] = [0,0,0]


      
# def dct():
#     w, h = img.size
#     outFunction = []
#     for x in range(w):
#         for y in range(h):
#             if(x == 0):
#                 alphaU = 1/math.sqrt(w)
#             else:
#                 alphaU = math.sqrt(2/w)

#             if(y == 0):
#                 alphaV = 1/math.sqrt(h)
#             else:
#                 alphaV = math.sqrt(2/h)

#             fXY = (math.cos((((2*x)+1)*math.pi*x)/2*w))*(math.cos((((2*y)+1)*math.pi*y)/2*h))
            
#             outFunction.append(alphaU*alphaV*fXY) 
#     x, y = outFunction
#     print(x, y)
# dct()

# imgs[0:100, 0:100, :] = [0 ,0 ,0]
# imgs[0:100, 200:300, :] = [0 ,0 ,0]
# imgs[0:100, 400:500, :] = [0 ,0 ,0]
# imgs[0:100, 600:700, :] = [0 ,0 ,0]
# imgs[0:100, 800:900, :] = [0 ,0 ,0]

# for i in range(imgs.shape[1]):
#         j = (i+2)
#         k = (i+1)
#         print(j,k)
#         imgs[0:100, j*100:k*100, :] = [0 ,0 ,0]
