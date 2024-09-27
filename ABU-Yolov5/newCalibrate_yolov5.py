import cv2
from ultralytics import YOLO  # assuming YOLO is installed within ultralytics

def calculate_distance(focal_length, actual_diameter, perceived_diameter):
    return (focal_length * actual_diameter) / perceived_diameter

def pure_math_distance(perceived_diameter):
    x = perceived_diameter
    result = 1.1583771347871026*(10**-7)*(x**4)-0.00004630774584854284*(x**3)+0.006964334626944125*(x**2)-0.4961449554608948*(x)+16.370014003126435
    return result

# Load the model (assuming 'yolov5_BallCylo.pt' is in the same directory)
model = YOLO('yolov5_BallCylo.pt')  

# Set ball and cylinder diameters
actual_ball = 0.1910  # meters
actual_cylo = 0.2794  # meters

# Assuming focal lengths are pre-determined (adjust if needed)
focal_ball = (85 * 800) / 102  # millimeters to meters conversion
focal_cylo = (279 * 800) / 185  # millimeters to meters conversion

threshold = 0.70

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    # Process the frame with the YOLO model
    results = model(frame)[0]

    centers = []
    distances = []

    for result in results.pandas().xyxy[0]:  # Assuming pandas extension is installed
        x1, y1, x2, y2, conf, class_id = result.tolist()

        w = int(x2) - int(x1)
        h = int(y2) - int(y1)

        if conf > threshold:
            perceived_diameter = max(w, h)

            # Distance calculation based on class ID
            if class_id == 0.0:  # Assuming class 0 is 'blue' (ball)
                distance = calculate_distance(focal_ball, actual_ball, perceived_diameter)
            elif class_id == 1.0:  # Assuming class 1 is 'cylo' (cylinder)
                distance = calculate_distance(focal_cylo, actual_cylo, perceived_diameter)
            else:
                print(f"Warning: Unknown class ID {class_id}")
                continue

            centerX = (x1 + x2) / 2
            centerY = (y1 + y2) / 2

            centers.append((centerX, centerY))
            distances.append(distance)

            # Draw bounding box, class label, and distance
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, f"{result['name'].upper()} {conf:.2f}", (int(x1), int(y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{distance:.2f} m", (int(x1), int(y2 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
