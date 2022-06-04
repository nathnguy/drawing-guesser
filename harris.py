# harris.py
# Harris Corner Detection
# https://muthu.co/harris-corner-detector-implementation-in-python/

from scipy import signal as sig
from scipy import ndimage as ndi
import numpy as np

# k: sensitivity factor
#   - separates corners from edges
#   - smaller k for detecting sharper corners
k = 0.01

def gradient_x(imggray):
  ##Sobel operator kernels.
  kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
  return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
  kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  return sig.convolve2d(imggray, kernel_y, mode='same')

# img - grayscale 2D pixel matrix
# returns the number of detected corners in img
def num_corners(img):
  I_x = gradient_x(img)
  I_y = gradient_y(img)

  Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
  Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
  Iyy = ndi.gaussian_filter(I_y**2, sigma=1)

  detA = Ixx * Iyy - Ixy ** 2
  traceA = Ixx + Iyy
  
  harris_response = detA - k * traceA ** 2

  result = 0

  for rowindex, response in enumerate(harris_response):
      for colindex, r in enumerate(response):
          # Edge : r < 0
          # Corner : r > 0
          # Flat: r = 0
          if r > 0:
              result += 1  

  return result
