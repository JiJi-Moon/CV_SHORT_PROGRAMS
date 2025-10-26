import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/content/pepper.png") 
if img is None:
    print("Error: Image not found!")
else:
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.title("Grayscale Histogram")
    plt.hist(gray.ravel(), 256, [0, 256], color='k')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("No. of Pixels")
    plt.show()

    colors = ('b', 'g', 'r')
    plt.title("Color Histogram")
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("No. of Pixels")
    plt.show()

    eq_gray = cv2.equalizeHist(gray)
    print("Original and Equalized Grayscale Images:")
    cv2_imshow(gray)
    cv2_imshow(eq_gray)

    plt.title("Equalized Grayscale Histogram")
    plt.hist(eq_gray.ravel(), 256, [0, 256], color='k')
    plt.show()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_2d = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

    plt.title("2D Hue-Saturation Histogram")
    plt.xlabel("Hue")
    plt.ylabel("Saturation")
    plt.imshow(hist_2d, interpolation='nearest')
    plt.colorbar()
    plt.show()
