from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import util

results = {}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("license_plate_detector.pt")

# Access camera
cap = cv2.VideoCapture(0)

vehicles = [2, 3, 5, 7]

frame_nmr = 0
# ... [rest of the code before the loop]

while True:
    ret, frame = cap.read()
    if ret:
        current_results = {}  # Use this to store results for the current frame only

        # Detect vehicles
        detections = coco_model(frame)[0]
        bboxes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                bboxes.append([x1, y1, x2, y2, score])

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Highlight license plate on the main video frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Drawing a yellow rectangle

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            try:
                # Read license plate number using EasyOCR
                detections = reader.readtext(license_plate_crop)
                for detection in detections:
                    license_plate_text = detection[1]
                    conf = detection[2]

                    if conf >= 0.8:
                        print(f"License Plate: {license_plate_text}, Confidence: {conf}")
                        current_results['license_plate'] = {'text': license_plate_text, 'bbox': [x1, y1, x2, y2],
                                                            'conf': conf}

                        # Annotate the frame with license plate text:
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 3)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255),
                                      3)  # yellow rectangle

            except Exception as e:
                print(f"EasyOCR Error: {e}")

        if 'license_plate' in current_results:  # Only write to CSV if a license plate was detected
            results[frame_nmr] = current_results
            util.write_csv(results, 'results.csv')

        # Display the frame
        cv2.imshow('Live Feed', frame)

        frame_nmr += 1

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
