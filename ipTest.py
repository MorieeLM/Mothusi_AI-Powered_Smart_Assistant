import cv2
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO('yolo11n.pt')  # Adjust model path as necessary

# Replace with your IP camera URL
ip_camera_url = 'http://192.168.1.147:8080/video'

# Open video capture
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Unable to connect to the IP camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Check if results are not empty
    if results:
        # Render results on the frame
        annotated_frame = results[0].plot()  # Use plot() instead of render()

        # Display the frame
        cv2.imshow('YOLO Object Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
