# Mờ nhiễu

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load the image and convert to grayscale
img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

blurred_img = cv2.GaussianBlur(img, (9, 9), 2) 

noise = np.random.normal(0, 10, img.shape)
noisy_img = blurred_img + noise

# Apply Wiener filter
restored_img = wiener(noisy_img, (5, 5))

# Visualize the results
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(noisy_img, cmap='gray'), plt.title('Blurred + Noisy Image')
plt.subplot(1, 3, 3), plt.imshow(restored_img, cmap='gray'), plt.title('Restored Image')
plt.show()

# This code works, i dunno why