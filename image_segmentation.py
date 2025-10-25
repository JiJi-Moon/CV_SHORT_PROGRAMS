import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('download.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 1. Thresholding
_, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
_, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# 2. Adaptive Thresholding
adap_thresh_mean = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
adap_thresh_gauss = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

# 3. Otsu’s Thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. K-means Clustering
Z = img.reshape((-1, 3))
Z = np.float32(Z)
K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
kmeans = res.reshape((img.shape))

# Display results
titles = ['Original Image', 'Binary', 'Binary Inv', 'Trunc', 'ToZero', 'ToZero Inv',
          'Adaptive Mean', 'Adaptive Gaussian', 'Otsu', 'K-means']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5,
          adap_thresh_mean, adap_thresh_gauss, otsu, kmeans]

plt.figure(figsize=(12, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
