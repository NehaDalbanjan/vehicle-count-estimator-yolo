from ultralytics import YOLO
import cv2
import pytesseract

# Set path to Tesseract (if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load video
video = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detect vehicles
    results = model(frame)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ["car", "truck", "bus"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_roi = frame[y1:y2, x1:x2]

            # Extract text from vehicle region
            text = pytesseract.image_to_string(vehicle_roi)
            if "AMBULANCE" in text.upper():
                cv2.putText(frame, "🚨 AMBULANCE DETECTED 🚨", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Ambulance Detection via Text", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
