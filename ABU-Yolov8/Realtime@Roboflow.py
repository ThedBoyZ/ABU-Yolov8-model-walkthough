import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor # Updated import

# Define how many frames to skip between processing
skip_frames = 5

model = YOLO('roboflow_toyball.pt')

cap = cv2.VideoCapture(0) # Replace with your video source

while True:
 # Read frame
 ret, frame = cap.read()
 if not ret:
   break

 # Process only if it's a frame to be analyzed
 if cap.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames == 0:
   result = model(frame) # Use model prediction
   # Process and display result (modify as needed)
   cv2.imshow('Frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()