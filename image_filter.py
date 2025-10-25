import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('download.jpeg')   # Replace with your image file
img = cv2.resize(img, (400, 400))  # Resize for convenience
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply different filters
# 1. Average filter
avg_filter = cv2.blur(img, (5, 5))

# 2. Gaussian filter
gaussian_filter = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Median filter
median_filter = cv2.medianBlur(img, 5)

# 4. Bilateral filter (edge-preserving)
bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

# Convert filtered images to RGB for matplotlib
avg_rgb = cv2.cvtColor(avg_filter, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_filter, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB)
bilateral_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

# Titles and images
titles = ['Original Image', 'Average Filter', 'Gaussian Filter', 'Median Filter', 'Bilateral Filter']
images = [img_rgb, avg_rgb, gaussian_rgb, median_rgb, bilateral_rgb]

# Plot all images together
plt.figure(figsize=(12, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
