#ex4
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def detect_and_label_objects(image_path):
    # Read and convert image to HSV
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges and labels (in HSV)
    colors = {
        "Red": [(0, 120, 70), (10, 255, 255)],
        "Yellow": [(25, 100, 100), (35, 255, 255)],
        "Blue": [(100, 150, 0), (140, 255, 255)]
    }

    # Loop through each color
    for label, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes and labels
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Create and show binary label for each detected object
            binary_label = np.zeros_like(mask)
            cv2.rectangle(binary_label, (x, y), (x + w, y + h), 255, -1)
            print(f"Binary label for {label}:")
            cv2_imshow(binary_label)

    # Show final labeled image
    print("Detected and Labeled Objects:")
    cv2_imshow(img)

# Example usage
detect_and_label_objects("/content/apple.jpg")

