import os
import cv2
from PIL import Image
import numpy as np
from halo_serving import FaceRecognition
from model import get_input_shape
import helper
import time

# n -> freeze frames for n seconds
# F -> camera fps
# f -> current frame
F = 30
n = F
f = 0
r = 0
isFreeze = False

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=F,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# detector = MTCNN()
FR = FaceRecognition()

# Using webcam
video = cv2.VideoCapture(0)
# Using gstreamer
# video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not video.isOpened():
    raise Exception('Error opening the camera')

while True:
    # Open the camera
    ret, frame = video.read()

    # if n == F:
    #     n = 0

    # n += 1
    # i += 1
    # if n in predFrame:
    # Resize the frame size and convert it to RGB Color
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_small_frame = frame[:, :, ::-1]

    # When freeze don't analyze faces for 11 frames
    if isFreeze == True and 0 <= r <= 10:
        faces = []
        r += 1
        print(f"r = {r}")
    else:
        faces = helper.detectFacesLive(frame)
        r = 0
        isFreeze = False

    if len(faces) != 0:
        x, y, w, h = faces[0]
        
        # Draw rectangle in face
        cv2.rectangle(frame,
                    (x, y),
                    (x+w, y+h),
                    (0,155,255),
                    2)
        
        # Predict
        img = helper.detectFace(frame[y:y+h, x:x+w], get_input_shape(), stream=True)
        if img.shape[1:3] == get_input_shape(): 
            pred, score = FR.predict(img)
        
        # Draw label class prediction
        cv2.rectangle(frame, (x, y+h + 65), (x+w, y+h), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"{pred}", (x + 3, y+h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, f"{score:.2f}", (x + 3, y+h + 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1) if score != "" else ""
        
        # Count until freeze
        f += 1
        print(f"frame {f}")
        # Freeze to show the predicted face
        if f == n or score != "":
            isFreeze = True
            f = 0
            freeze_img = frame.copy()
            cv2.putText(freeze_img, "Not Employee!", (x + 3, y+h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1) if pred == "" else ""
            cv2.imshow(f"Hello {pred}", freeze_img)
            cv2.waitKey(3000)
            cv2.destroyWindow(f"Hello {pred}")

    # Resize the display window
    # display_frame = cv2.resize(frame, (1280,720))
    cv2.imshow('Hello Welcome to iNews Tower', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
