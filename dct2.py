import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte


# from scipy.fftpack import dct, idct

# # implement 2D DCT
# def dct2(a):
#     return dct(dct(a.T, norm='ortho').T, norm='ortho')

# # implement 2D IDCT
# def idct2(a):
#     return idct(idct(a.T, norm='ortho').T, norm='ortho')    

# from skimage.io import imread
# from skimage.color import rgb2gray
# import numpy as np
# import matplotlib.pylab as plt

# # read lena RGB image and convert to grayscale
# im = rgb2gray(imread('t1.jpg')) 
# imF = dct2(im)
# im1 = idct2(imF)

# # check if the reconstructed image is nearly equal to the original image
# np.allclose(im, im1)
# # True

# # plot original and reconstructed images with matplotlib.pylab
# plt.gray()
# plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
# plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
# plt.show()