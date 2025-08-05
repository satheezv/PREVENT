import cv2
import os
import time



# Set folder to save images
save_path = "W:\gasSensor_ws\GasSensor_ws\data\\rack_missing_vials_2"
os.makedirs(save_path, exist_ok=True)



# Change to external webcam index (0 = default, 1 = external webcam)
camera_index = 0  # Change to 2 or 3 if needed
cap = cv2.VideoCapture(camera_index)

# Set resolution (optional)
cap.set(3, 1920)  # Width
cap.set(4, 1080)  # Height

# Set image count
image_count = 0

print("Press SPACE to capture an image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Show the live video feed
    cv2.imshow("Webcam Capture", frame)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # If SPACE is pressed, capture image
    if key == ord(" "):  
        image_name = os.path.join(save_path, f"image_{image_count:04d}.jpg")
        cv2.imwrite(image_name, frame)
        print(f"Saved: {image_name}")

        image_count += 1
        time.sleep(2)  # 2-second delay before next capture

    # If ESC is pressed, exit
    elif key == 27:  
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
