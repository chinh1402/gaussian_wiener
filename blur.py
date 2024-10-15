# Mờ

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load the image and convert to grayscale
img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

blurred_img = cv2.GaussianBlur(img, (9, 9), 10) 

# Apply Wiener filter
restored_img = wiener(blurred_img, (5, 5))
normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Visualize the results
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(blurred_img, cmap='gray'), plt.title('Blurred')
plt.subplot(1, 3, 3), plt.imshow(normalized_img, cmap='gray'), plt.title('Restored Image')
plt.show()