import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_path = r'C:\Users\Desktop\model.h5' 
img_path = r'C:\Users\Pictures\jump_4.jpg' 

action_model = load_model(model_path) 
action_classes = ['walking', 'running', 'jumping', 'standing', 'sitting', 'falling', 'other'] 

def predict_action(path, model, classes): 
    # Assuming model input shape is (Batch, 90, 90, 1) for grayscale
    img = image.load_img(path, target_size=(90, 90), color_mode='grayscale') 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0 
    
    prediction = model.predict(img_array, verbose=0)
    action_label = classes[np.argmax(prediction)] 
    
    return action_label 

action = predict_action(img_path, action_model, action_classes) 
print(f"Action: {action}")

