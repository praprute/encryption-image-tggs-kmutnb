import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte

B=8 #blocksize
fn3= 't1.jpg'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
h , w = np.array(img1.shape[:2])/B * B

h = int(h)
w = int(w)
print(h)
print(w)

# img1 = img1[:h , :w]
img1 = img1[:h,:w]

blocksV = h/B
blocksV = int(blocksV)
blocksH=w/B
blocksH = int(blocksH)
vis0 = np.zeros((h,w), np.float32)
Trans = np.zeros((h,w), np.float32)
vis0[:h, :w] = img1
for row in range(blocksV):
        for col in range(blocksH):
                currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                if currentblock < 0:
                        print(currentblock)
                Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
cv2.imwrite('Transformed.jpg', Trans)

# indensity range 

plt.imshow(img1,cmap="gray")
point=plt.ginput(1)
block=np.floor(np.array(point)/B) #first component is col, second component is row
print(block)
col=block[0,0]
col = int(col)
row=block[0,1]
row = int(row)
plt.plot([B*col,B*col+B,B*col+B,B*col,B*col],[B*row,B*row,B*row+B,B*row+B,B*row])
plt.axis([0,w,h,0])
plt.title("Original Image")

plt.figure()
plt.subplot(1,2,1)
selectedImg=img1[row*B:(row+1)*B,col*B:(col+1)*B]
N255=Normalize(0,255) #Normalization object, used by imshow()
plt.imshow(selectedImg,cmap="gray",norm=N255,interpolation='nearest')
plt.title("Image in selected Region")

plt.subplot(1,2,2)
selectedTrans=Trans[row*B:(row+1)*B,col*B:(col+1)*B]
plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')
plt.colorbar(shrink=0.5)
plt.title("DCT transform of selected Region")

back0 = np.zeros((h,w), np.float32)
for row in range(blocksV):
        for col in range(blocksH):
                currentblock = cv2.idct(Trans[row*B:(row+1)*B,col*B:(col+1)*B])
                back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
cv2.imwrite('BackTransformed.jpg', back0)

diff=back0-img1
print(diff.max())
print(diff.min())
MAD=np.sum(np.abs(diff))/float(h*w)
print("Mean Absolute Difference: ",MAD)
plt.figure()
plt.imshow(back0,cmap="gray")
plt.title("Backtransformed Image")
plt.show()