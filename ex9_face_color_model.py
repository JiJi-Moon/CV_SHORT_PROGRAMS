import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

test_path = r"test.webp"        
known_folder = r"known images"  

def color_hist(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

test_img = cv2.imread(test_path)
if test_img is None:
    raise ValueError("❌ Test image not found! Check the path.")
test_hist = color_hist(test_img)

best_match = None
min_distance = 1.0
known_paths = [os.path.join(known_folder, f) for f in os.listdir(known_folder)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for i, path in enumerate(known_paths):
    known_img = cv2.imread(path)
    known_hist = color_hist(known_img)
    distance = cv2.compareHist(test_hist, known_hist, cv2.HISTCMP_BHATTACHARYYA)
    if distance < min_distance:
        min_distance = distance
        best_match = i

threshold = 0.5

if min_distance < threshold:
    print(f"✅ Face recognized as: {os.path.basename(known_paths[best_match])} "
          f"(Distance: {min_distance:.3f})")
else:
    print("❌ Face not recognized (too different)")

plt.figure(figsize=(12, 4))
plt.subplot(1, len(known_paths) + 1, 1)
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title("Test Image")
plt.axis('off')

for i, path in enumerate(known_paths):
    img = cv2.imread(path)
    plt.subplot(1, len(known_paths) + 1, i + 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Known {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

