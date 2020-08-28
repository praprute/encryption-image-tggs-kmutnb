import cv2
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte
# np.set_printoptions(threshold=sys.maxsize)
B=1 #blocksize
fn3= 't1.jpg'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
h , w = np.array(img1.shape[:2])/B * B
h = int(h)
w = int(w)
print(h)
print(w)
img1 = img1[:h,:w]
blocksV = h/B
blocksV = int(blocksV)
blocksH = w/B
blocksH = int(blocksH)
maxlength = h*w
random.seed('junior')
ChoiceKey = [1,-1]
keyIndex = [random.choice(ChoiceKey) for i in range(maxlength)]
print('--------keyIndex----------------')
print(keyIndex[0])
print('--------------------------------')
vis0    = np.zeros((h,w), np.double)
Trans   = np.zeros((h,w), np.double)
BTrans  = np.zeros((h,w), np.double)
vis0[:h, :w] = img1

for row in range(blocksV):
    currentblock = cv2.dct(vis0[row*B:(row+1)*B, 0:blocksH])
    Trans[row*B:(row+1)*B, 0:blocksH] = currentblock
global positionKeyIndex
positionKeyIndex = 0
for  row in range(blocksV):
    for col in range(blocksH):
        ck = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        ck = ck*keyIndex[positionKeyIndex]
        Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = ck
        positionKeyIndex = positionKeyIndex + 1
positionKeyIndex = 0
print('-------Original-------------------------------')
print(vis0)
print('-------Trans-------------------------------')
print(Trans)
cv2.imwrite('Transformed.jpg', Trans)
# cv2.imshow('YCBCR Image',vis0)
# cv2.waitKey(10000)
global DscPositionKeyIndex
DscPositionKeyIndex = 0

for  row in range(blocksV):
    for col in range(blocksH):
        Dsck = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        Dsck = Dsck*keyIndex[DscPositionKeyIndex]
        Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = Dsck
        DscPositionKeyIndex = DscPositionKeyIndex + 1

DscPositionKeyIndex = 0

for row in range(blocksV):
    cTrans = cv2.idct(Trans[row*B:(row+1)*B, 0:blocksH])
    BTrans[row*B:(row+1)*B, 0:blocksH] = cTrans
cv2.imwrite('BackTransformed.jpg', BTrans)
print('-------BTrans-------------------------------')
print(BTrans)




# for  row in range(blocksV):
#     for col in range(blocksV):
#         ck = Trans[row*B:(row+1)*B, col*B:(col+1)*B]
#         if(ck < 0):
#             key.append(-1)
#             Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = abs(Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0])
#         else:
#             key.append(1)

# global poIndex
# poIndex = 0
# for row in range(blocksV):
#     for col in range(blocksH):
#         Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]*key[poIndex]
#         poIndex = poIndex + 1


# for row in range(blocksV):
#         for col in range(blocksH):
#                 currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
#                 if currentblock < 0:
#                         print(currentblock)
#                 Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock

# indensity range lected Region")
# back0 = np.zeros((h,w), np.float32)
# for row in range(blocksV):
#         for col in range(blocksH):
#                 currentblock = cv2.idct(Trans[row*B:(row+1)*B,col*B:(col+1)*B])
#                 back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
# cv2.imwrite('BackTransformed.jpg', back0)
