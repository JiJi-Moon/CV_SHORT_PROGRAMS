#ex2 
import cv2
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("/content/pepper.png")  # Change the path if needed
if img is None:
    print("Error: Image not found!")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1️⃣ Grayscale Histogram ---
    plt.title("Grayscale Histogram")
    plt.hist(gray.ravel(), 256, [0, 256], color='k')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("No. of Pixels")
    plt.show()

    # --- 2️⃣ Color Histogram ---
    colors = ('b', 'g', 'r')
    plt.title("Color Histogram")
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("No. of Pixels")
    plt.show()

    # --- 3️⃣ Histogram Equalization ---
    eq_gray = cv2.equalizeHist(gray)
    print("Original and Equalized Grayscale Images:")
    cv2_imshow(gray)
    cv2_imshow(eq_gray)

    plt.title("Equalized Grayscale Histogram")
    plt.hist(eq_gray.ravel(), 256, [0, 256], color='k')
    plt.show()

    # --- 4️⃣ 2D Histogram (Hue vs Saturation) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_2d = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

    plt.title("2D Hue-Saturation Histogram")
    plt.xlabel("Hue")
    plt.ylabel("Saturation")
    plt.imshow(hist_2d, interpolation='nearest')
    plt.colorbar()
    plt.show()
