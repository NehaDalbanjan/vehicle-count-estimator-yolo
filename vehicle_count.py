from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start video
video = cv2.VideoCapture("sample_video.mp4")

# Counting setup
count = 0
line_y = 300
counted_ids = set()  # store unique object IDs

# Enable tracking
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Use YOLO with tracking mode
    results = model.track(frame, persist=True)

    # Draw line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = int(classes[i])
            obj_id = int(ids[i])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if label in [2, 3, 5, 7]:  # vehicle classes
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                if line_y - 5 < cy < line_y + 5:
                    if obj_id not in counted_ids:
                        count += 1
                        counted_ids.add(obj_id)
                        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

    # Display total count
    cv2.putText(frame, f"Vehicle Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Accurate Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
