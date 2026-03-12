from ultralytics import YOLO
import cv2
import winsound  # for playing a short sound on Windows

# Load your trained model
model = YOLO(r"D:\training\runs\detect\train\weights\best.pt")

# Path to your test video
video_path = r"D:\training\3360811-sd_426_240_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ambulance_seen = False  # to prevent multiple prints/sounds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes

    # Play sound + print only once when ambulance first appears
    if len(boxes) > 0 and not ambulance_seen:
        print("🚑 Ambulance detected ✅")
        winsound.Beep(1000, 500)  # (frequency, duration in ms)
        ambulance_seen = True  # stop repeating

    # Reset flag when ambulance disappears
    if len(boxes) == 0:
        ambulance_seen = False

    cv2.imshow("Ambulance Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video finished.")
