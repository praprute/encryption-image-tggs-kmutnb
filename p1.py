
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import numpy as np
from skimage import io
from skimage import img_as_float
import sys
import random
import os


img = Image.open('t1.jpg')
print(img.size)

w, h = img.size
pxs = []

for x in range(w):
    for y in range(h):
        pxs.append(img.getpixel((x,y)))


idx = list(range(len(pxs)))
random.shuffle(idx)
print(type(idx))

seed_img = random.seed(hash(img.size))
out = []
for i in idx:
    out.append(pxs[i])

out_img = Image.new("RGB", img.size)
wO, hO = img.size
pxIter = iter(out)
for x in range(wO):
    for y in range(hO):
        out_img.putpixel((x,y), pxIter)
out_img.show()
    




def getPixels(img):
    w, h = img.size
    pxs = []
    for x in range(w):
        for y in range(h):
            pxs.append(img.getpixel((x,y)))
    return pxs

def seed(img):
    return random.seed(hash(img.size))

def scrambleIndex(pxs):
    idx = range(len(pxs))
    random.shuffle(idx)
    return idx

def scramblepixel(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambleIndex(pxs)
    out = []
    for i in idx:
        out.append(pxs[i])
    return out

    

'''
img = io.imread('t1.jpg')
print(img.min(),img.max())

#ทำสีแดงที่รูป
img[0:200, 0:200, :] = [255, 0, 0]
img[:, :, 0:200] = [0, 0, 255]
plt.imshow(img)
print(img)
print(img[0][0])
'''



'''
img_float = img_as_float(img)
print(img_float.min(),img_float.max())\
'''

'''
random_image = np.random.random([500, 500])
plt.imshow(random_image)
print(random_image)
'''