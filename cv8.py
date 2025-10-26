import cv2
import numpy as np

image_path = r'C:\Users\Pictures\starfish.png'
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image_gray is None:
    print(f"Error: Could not load image from {image_path}. Please update the 'image_path' variable.")
    exit()

image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

edges = cv2.Canny(image_blurred, 70, 200)
cv2.namedWindow('1. Canny Edges (After Blur)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('1. Canny Edges (After Blur)', 800, 600)
cv2.imshow('1. Canny Edges (After Blur)', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

corners = cv2.goodFeaturesToTrack(image_blurred, 25, 0.01, 10)
if corners is not None:
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)

cv2.namedWindow('2. Image with Corners (Green)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('2. Image with Corners (Green)', 800, 600)
cv2.imshow('2. Image with Corners (Green)', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

edges_for_hough = cv2.Canny(image_blurred, 70, 200, apertureSize=3)
lines = cv2.HoughLines(edges_for_hough, 1, np.pi / 180, 200)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.namedWindow('3. Corners and Hough Lines (Red)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('3. Corners and Hough Lines (Red)', 800, 600)
cv2.imshow('3. Corners and Hough Lines (Red)', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

