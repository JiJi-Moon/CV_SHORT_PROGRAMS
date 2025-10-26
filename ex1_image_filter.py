import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('download.jpeg') 
img = cv2.resize(img, (400, 400))  
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

avg_filter = cv2.blur(img, (5, 5))

gaussian_filter = cv2.GaussianBlur(img, (5, 5), 0)

median_filter = cv2.medianBlur(img, 5)

bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

avg_rgb = cv2.cvtColor(avg_filter, cv2.COLOR_BGR2RGB)
gaussian_rgb = cv2.cvtColor(gaussian_filter, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB)
bilateral_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

titles = ['Original Image', 'Average Filter', 'Gaussian Filter', 'Median Filter', 'Bilateral Filter']
images = [img_rgb, avg_rgb, gaussian_rgb, median_rgb, bilateral_rgb]

plt.figure(figsize=(12, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()


