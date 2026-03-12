from ultralytics import YOLO
import cv2

# Load your trained YOLO model (update the path correctly)
model = YOLO(r"runs/detect/train/weights/best.pt")

# Load the test image
img_path = r"D:\training\test.jpg"   # path to your image
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found at", img_path)
    exit()

# Perform detection
results = model(img)

# Plot detections on the image
annotated_img = results[0].plot()

# Display result
cv2.imshow("Ambulance & Vehicle Detection", annotated_img)

# Press 'q' to close the window
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
