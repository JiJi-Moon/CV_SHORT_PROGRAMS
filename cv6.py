import cv2 
import numpy as np 
import pytesseract 
import os
from PIL import Image

# --- CONFIGURATION & CONSTANTS ---
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

CASCADE_FILE = "haarcascade_russian_plate_number.xml"
cascade = cv2.CascadeClassifier(CASCADE_FILE) if os.path.exists(CASCADE_FILE) else None
if cascade is None:
    print(f"ERROR: Cascade file '{CASCADE_FILE}' not found. Please download it.")

STATES = {
    "AN":"Andaman and Nicobar", "AP":"Andhra Pradesh", "AR":"Arunachal Pradesh",
    "AS":"Assam", "BR":"Bihar", "CH":"Chandigarh", "CG":"Chhattisgarh",
    "DN":"Dadra and Nagar Haveli", "DD":"Daman and Diu", "DL":"Delhi", 
    "GA":"Goa", "GJ":"Gujarat", "HR":"Haryana", "HP":"Himachal Pradesh",
    "JK":"Jammu and Kashmir", "JH":"Jharkhand", "KA":"Karnataka", "KL":"Kerala",
    "LD":"Lakshadweep", "MP":"Madhya Pradesh", "MH":"Maharashtra", "MN":"Manipur", 
    "ML":"Meghalaya", "MZ":"Mizoram", "NL":"Nagaland", "OD":"Odisha", 
    "PY":"Pondicherry", "PN":"Punjab", "RJ":"Rajasthan", "SK":"Sikkim",
    "TN":"TamilNadu", "TR":"Tripura", "UP":"Uttar Pradesh", "UK":"Uttarakhand",
    "WB":"West Bengal", "TS":"Telangana"
}

def extract_num(img_filename): 
    if cascade is None:
        print("Cannot proceed with detection: Haar Cascade file is missing.")
        return
        
    img = cv2.imread(img_filename) 
    if img is None:
        print(f"Error: Could not load image file at {img_filename}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    nplate = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4) 

    recognized_plates = []
    output_filename = "Result_Annotated.png"

    for (x, y, w, h) in nplate: 
        a, b = (int(0.02 * img.shape[1]), int(0.02 * img.shape[0])) 
        plate = img[y + a:y + h - a, x + b:x + w - b, :] 
        
        kernel = np.ones((1, 1), np.uint8) 
        plate = cv2.dilate(plate, kernel, iterations=1) 
        plate = cv2.erode(plate, kernel, iterations=1) 
        
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) 
        _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY) 
        
        config = r'--oem 3 --psm 8'
        read = pytesseract.image_to_string(plate_thresh, config=config) 
        read = ''.join(e for e in read if e.isalnum()) 
        
        state_code = read[0:2].upper()
        state_name = STATES.get(state_code, "STATE UNKNOWN")
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2) 
        text_display = f"{read} ({state_name})"
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1) 
        
        cv2.putText(img, text_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
        recognized_plates.append((read, state_name))
        cv2.imshow(f"Detected Plate: {read}", plate_thresh) 

    print("-" * 30)
    if recognized_plates:
        print(f"Successfully processed {len(recognized_plates)} plate(s).")
        for num, state in recognized_plates:
            print(f"Recognized Number: {num}")
            print(f"Registered State: {state}")
    else:
        print("No license plate detected in the image.")
    print("-" * 30)
    
    cv2.imwrite(output_filename, img) 
    print(f"Annotated result saved to: {output_filename}")

    cv2.imshow("Annotated Car Image", img) 
    print("Press 'q' or ESC in the image window to exit.")
    key = cv2.waitKey(0) 
    if key == ord('q') or key == 27:
        cv2.destroyAllWindows() 
    
    cv2.destroyAllWindows()

# --- MAIN EXECUTION ---
extract_num("car_img.png")
