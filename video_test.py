import cv2

# Load the video (replace 'traffic.mp4' with your file name)
video = cv2.VideoCapture("sample_video.mp4")

while True:
    ret, frame = video.read()   # Read frame by frame
    if not ret:
        break                   # Stop when video ends

    cv2.imshow("CCTV Footage", frame)  # Show the video

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
