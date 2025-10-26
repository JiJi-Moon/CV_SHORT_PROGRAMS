import cv2, numpy as np, matplotlib.pyplot as plt, requests
from io import BytesIO

def cv2_imshow(t, i):
    if i is None: print(f"Error: Image for '{t}' is None."); return
    if len(i.shape) == 3 and i.shape[2] == 3: i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    plt.imshow(i, cmap='gray' if len(i.shape) == 2 else None); plt.title(t); plt.axis('off'); plt.show()

def detect_and_label_objects(p, cr, l):
    try:
        i = cv2.imread(p) if isinstance(p, str) else p if isinstance(p, np.ndarray) else cv2.imdecode(np.frombuffer(p.read(), np.uint8), cv2.IMREAD_COLOR)
        if i is None: print("Error: Could not load image."); return
    except Exception as e: print(f"Error loading image: {e}"); return

    li, hsv = i.copy(), cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    print("--- Starting Object Detection and Labeling ---")

    for idx, (lh, uh) in enumerate(zip(cr[::2], cr[1::2])):
        lower, upper, label = np.array(lh), np.array(uh), l[idx]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours: print(f"No objects found for color: {label}"); continue
        print(f"Found {len(contours)} object(s) for color: {label}")

        for j, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(li, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dl = f"{label}-{j+1}"
            cv2.putText(li, dl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            som = np.zeros_like(mask)
            cv2.drawContours(som, [c], -1, 255, thickness=cv2.FILLED)
            cv2_imshow(f"Binary Label: {dl} (Mask)", som[y:y+h, x:x+w])

    cv2_imshow("Original Image with Labeled Objects", li)

def get_sample_image():
    url = "https://picsum.photos/800/600/?random=1" 
    try:
        r = requests.get(url, stream=True, timeout=10); r.raise_for_status()
        print(f"Successfully downloaded a sample image."); return BytesIO(r.content)
    except requests.exceptions.RequestException as e:
        print(f"Could not download sample image: {e}")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 255), -1)
        cv2.rectangle(img, (120, 20), (200, 100), (0, 255, 255), -1)
        cv2.rectangle(img, (220, 20), (280, 280), (255, 0, 0), -1)
        print("Using generated fallback image."); return img

cr = [(0, 100, 100), (10, 255, 255), (25, 100, 100), (35, 255, 255), (100, 100, 100), (120, 255, 255)] 
l = ["Red", "Yellow", "Blue"] 
detect_and_label_objects(get_sample_image(), cr, l)
