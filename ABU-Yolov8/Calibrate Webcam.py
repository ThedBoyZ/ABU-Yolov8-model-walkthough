import cv2
import numpy as np

def calculate_distance(focal_length, actual_diameter, perceived_diameter):
    return (focal_length * actual_diameter) / perceived_diameter

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Set ball diameter to 8.30 cm
    actual_diameter = 0.083 

    # focal_length = (actual_size(mm) * distance_from_camera(mm)) / apparent_size_in_pixels
    focal_length = (83 * 800) / 102

    # Need intrinsic calibration (Now this is calibrate just linear)    

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 100:  # Adjust this threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                perceived_diameter = max(w, h)

                distance = calculate_distance(focal_length, actual_diameter, perceived_diameter)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Yellow Ball Distance Measurement', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
