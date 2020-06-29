import os
import cv2
from PIL import Image
import numpy as np
from halo import FaceRecognition
from model import get_input_shape
import helper

# n -> fps count
# F -> desired framerate
# i -> counter value for naming filename
# n = 0
F = 30
# i = 0
# required_size = (224,224)
# predFrame = [10,20,30]
# model_path = "./models/model4-best"
# model_name = "best_face_recognition.h5" 
# lite_model_name = "best_lite_face_recognition.tflite"
# class_name = "face_recognition_class_names.npy"

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
video = cv2.VideoCapture(2)
# Using gstreamer
# video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not video.isOpened():
    print('Error opening the camera')

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

    # Detect the face from rgb input
    faces = helper.detectFacesLive(frame)

    if len(faces) != 0 :
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
            pred = FR.predict(img)
        
        # Draw label class prediction
        cv2.rectangle(frame, (x, y+h + 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(pred), (x + 3, y+h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Resize the display window
    display_frame = cv2.resize(frame, (720,480))
    cv2.imshow('Hello Welcome to iNews Tower', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
