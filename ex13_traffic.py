from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  

video_path = "traffic_video.mp4"  
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame) 
    frame = results[0].plot()  

    out.write(frame) 
    cv2.imshow("Traffic Abnormal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved as", output_path)

