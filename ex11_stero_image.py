import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

left_path = r"known images\1.jpeg"
right_path = r"known images\2.jpeg"

left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

if left is None or right is None:
    raise FileNotFoundError("❌ One or both stereo images not found. Check your paths!")
if left.shape != right.shape:
    right = cv2.resize(right, (left.shape[1], left.shape[0]))
    
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5, 
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32
)

disparity = stereo.compute(left, right).astype(np.float32) / 16.0
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Left Image")
plt.imshow(left, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Right Image")
plt.imshow(right, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Disparity Map")
plt.imshow(disp_vis, cmap='plasma')
plt.axis('off')

plt.tight_layout()
plt.show()

