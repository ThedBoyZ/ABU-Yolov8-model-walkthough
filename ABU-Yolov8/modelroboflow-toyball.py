import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
skip_frames = 5

model = YOLO('D:/ThedboyCoding/ABU Yolov8/Models/roboflow_toyball.pt')

cap = cv2.VideoCapture(0)

while True:
 ret, frame = cap.read()
 if not ret:
   break

 if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames == 0:
   result = model(frame) 
   # Process and display result
   cv2.imshow('Frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()