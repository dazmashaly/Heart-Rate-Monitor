import cv2
import requests
from PIL import Image
import base64
import numpy as np
# Function to send image to API and print response
def send_frame_to_api(frame):
    # Convert frame to JPEG format
    temp_file_path = 'temp_frame.jpg'
    cv2.imwrite(temp_file_path, frame)
    
    # Prepare payload
    files = {'image': open(temp_file_path, 'rb')}
    
    # Prepare payload
    
    # Send POST request to API endpoint
    response = requests.post('https://heart-monitor-1vor.onrender.com/hr', files=files)
    
    # Print response
    
    return response.json()

# Open webcam
    
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 1.5)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * 1.5)  # Set frame height

i = 1
cur = f"HR : 0 bpm, L : 0"
rates = [0]
while True:
    # Capture frame-by-frame
    
    frame = cap.read()[1]
    to_send = cv2.resize(frame, (960, 720))
    print(frame.shape)
    # Display the resulting frame
    # Send frame to API and print respons
    if i < 100 or i % 40 ==0:
        res = send_frame_to_api(to_send)
        if res['HR'] != 0:
            rates.append(res['HR']) 
    if len(rates) > 15:
        rates = rates[-15:]
    cur = f"HR : {int(np.mean(rates))} bpm, L : {res['L']}"
    frame[:100, :, :] = 0


    cv2.putText(frame, cur, (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.imwrite("test_frame1.jpg", frame)
    print(cur)
    # Break the loop when 'q' is pressed
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
