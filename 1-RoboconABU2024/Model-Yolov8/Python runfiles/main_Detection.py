from ultralytics import YOLO
import cv2

def calculate_distance(focal_length, actual_diameter, perceived_diameter):
    return (focal_length * actual_diameter) / perceived_diameter

def pure_math_distance(perceived_diameter):
    x = perceived_diameter
    result = 1.1583771347871026*(10**-7)*(x**4)-0.00004630774584854284*(x**3)+0.006964334626944125*(x**2)-0.4961449554608948*(x)+16.370014003126435
    return(result)

# Load a model
model = YOLO('Yolov8_BallCyro.pt')  # load a custom model

# Set ball diameter to 18.85 cm, cylo diameter to 27.94cm
actual_ball = 0.1910 
actual_cylo = 0.2794

# focal_length = (actual_size(mm) * distance_from_camera(mm)) / apparent_size_in_pixels
focal_ball = (85 * 800) / 102 # Old (EDIT: -10cm[start at griper], +30cm[end at 4.5m ball])
focal_cylo = (279 * 800) / 185 # +14cm[end at 4.3m cylo]

#TODO: Need intrinsic calibration (Now this is calibrate just linear)  

threshold = 0.70
distance = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture frame")
        break

    frame = cv2.flip(frame, -1)
    
    results = model(frame)[0]
    
    centers = []
    dists = []

    for result in results.boxes.data.tolist():
        print(result)
        x1, y1, x2, y2, score, class_id = result
        
        w = int(x2) - int(x1)
        h = int(y2) - int(y1)
        
        if score > threshold:
            # Display confidence score (accuracy)
            accuracy_score = score * 100
            
            perceived_diameter = max(w, h)
            #TODO perceived_cylo = ???
            
            #TODO cylo incorrect distance 
            
            # class_id -> 0.0, 1.0, 2.0, 3.0 = blue, cylo, purple, red
            if class_id == float("1.0"):
                distance = calculate_distance(focal_cylo, actual_cylo, perceived_diameter)
            else:
                # distance = calculate_distance(focal_ball, actual_ball, perceived_diameter)
                distance = pure_math_distance(perceived_diameter)*(2/3)
            
            centerX = (x1 + x2) / 2
            centerY = (y1 + y2) / 2
            
            centers.append((centerX, centerY))
            dists.append(distance)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, f"{results.names[int(class_id)].upper()} {accuracy_score:.2f}%", (int(x1), int(y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{distance:.2f} m", (int(x1), int(y2 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
