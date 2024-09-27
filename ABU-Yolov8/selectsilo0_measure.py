from ultralytics import YOLO
import cv2
import os
import time

def calculate_distance(focal_length, actual_diameter, perceived_diameter):
    return (focal_length * actual_diameter) / perceived_diameter

def pure_math_distance(perceived_diameter):
    x = perceived_diameter
    result = (1.1583771347871026 * (10**-7) * (x**4) -
              0.00004630774584854284 * (x**3) +
              0.006964334626944125 * (x**2) -
              0.4961449554608948 * (x) +
              16.370014003126435)
    return result

# Load a model
model = YOLO('D:/ThedboyCoding/ABU Yolov8/Models/cyroV8-size-n.pt')

# Set ball diameter to 18.85 cm, cylo diameter to 27.94cm
actual_ball = 0.1910 
actual_cylo = 0.2794

# focal_length = (actual_size(mm) * distance_from_camera(mm)) / apparent_size_in_pixels
focal_ball = (85 * 800) / 102 # Old (EDIT: -10cm[start at griper], +30cm[end at 4.5m ball])
focal_cylo = (279 * 800) / 185 # +14cm[end at 4.3m cylo]

threshold = 0.65
distance = 0

video_path = 'D:/ThedboyCoding/ABU Yolov8/cyro.mp4'

# Timeout parameters
timeout_duration = 4  # seconds
last_detection_time = 0
last_detection_data = None

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
else:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file at {video_path}")
    else:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Unable to capture frame")
                break

            results = model(frame)[0]
            current_time = time.time()

            if current_time - last_detection_time > timeout_duration:
                last_detection_data = None

            # Initialize variables for closest detection
            closest_detection_x = float('inf')
            closest_detection_data = None

            # Iterate over detections
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)
                
                if score > threshold:
                    accuracy_score = score * 100
                    perceived_diameter = max(w, h)

                    if class_id == float("1.0"):  # Cylo detected
                        distance = calculate_distance(focal_cylo, actual_cylo, perceived_diameter)
                    else:  # Ball detected
                        distance = pure_math_distance(perceived_diameter) * (2 / 3)

                    # Check if the current detection is closer in x-coordinate
                    if x1 < closest_detection_x:
                        closest_detection_x = x1
                        closest_detection_data = {
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "class_id": class_id,
                            "accuracy_score": accuracy_score,
                            "distance": distance
                        }

            # Update last detection data with the closest detection in x-coordinate
            if closest_detection_data:
                last_detection_data = closest_detection_data
                last_detection_time = current_time

            if last_detection_data:
                x1, y1, x2, y2 = last_detection_data["box"]
                class_id = last_detection_data["class_id"]
                accuracy_score = last_detection_data["accuracy_score"]
                distance = last_detection_data["distance"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"{results.names[int(class_id)].upper()} {accuracy_score:.2f}%", 
                            (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{distance:.2f} m", (x1, y2 + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                # Return the position of the rectangle
                rectangle_position = (x1, y1)
                print(f"Rectangle Position: {rectangle_position}")
                
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()
