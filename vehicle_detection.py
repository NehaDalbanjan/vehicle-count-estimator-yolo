from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' = nano version (small and fast)

# Load your CCTV footage (or webcam)
video = cv2.VideoCapture("sample_video.mp4")  # or use 0 for webcam

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Draw detection boxes on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
