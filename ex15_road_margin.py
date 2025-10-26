import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("road.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)

h, w = edges.shape
mask = np.zeros_like(edges)
roi = np.array([[(0,h), (0.1*w,0.5*h), (0.9*w,0.5*h), (w,h)]], np.int32)
cv2.fillPoly(mask, roi, 255)
edges = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)
line_img = img.copy()
if lines is not None:
    for x1,y1,x2,y2 in lines[:,0]:
        slope = (y2-y1)/(x2-x1+1e-6)
        color = (255,0,0) if slope<0 else (0,255,0)
        if abs(slope)>0.2:
            cv2.line(line_img, (x1,y1), (x2,y2), color, 2)

result = cv2.addWeighted(img,0.8,line_img,1,0)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off'); plt.show()
cv2.imwrite('road_with_margins.jpg', result)

