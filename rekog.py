import os
import cv2
from PIL import Image
import numpy as np
from halo_serving import FaceRecognition
from model import get_input_shape
import helper
import time
from register import register

FPS = 30

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=FPS,
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

def rekog():
    # n -> freeze frames for n seconds
    # f -> current frame
    # R -> wait time 
    # r -> radius
    n = 30
    f = 0
    R = 0
    r = 180
    isFreeze = False
    
    
    # detector = MTCNN()
    FR = FaceRecognition()

    # Using webcam
    video = cv2.VideoCapture(0)
    # Using gstreamer
    # video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not video.isOpened():
        raise Exception('Error opening the camera')

    # Open the camera and get the camera size
    ret, temp = video.read()

    # convert image to grayscale image
    gray_image = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)
    # calculate moments of binary image
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    CX = int(M["m10"] / M["m00"])
    CY = int(M["m01"] / M["m00"])

    while True:
        # Open camera for recognizing
        ret, frame = video.read()
        rekog_frame = frame[CY-r:CY+r, CX-r:CX+r]
        
        # Draw rekog circle
        cv2.circle(frame, center=(CX,CY), radius=r, color=(243,144,29), thickness=3)

        # When freeze don't analyze faces for 11 frames
        if isFreeze == True and 0 <= R <= 10:
            faces = []
            R += 1
            print(f"R = {R}")
        else:
            faces = helper.detectFacesLive(rekog_frame)
            R = 0
            isFreeze = False

        if len(faces) != 0:
            x, y, w, h = faces[0]
            
            # Draw rectangle in face
            cv2.rectangle(rekog_frame,
                        (x, y),
                        (x+w, y+h),
                        (0,155,255),
                        2)
            
            # Predict
            img = helper.detectFace(rekog_frame[y:y+h, x:x+w], get_input_shape(), stream=True)
            if img.shape[1:3] == get_input_shape(): 
                pred, score = FR.predict(img)
            
            # Count until freeze
            f += 1
            print(f"frame {f}")
            
            # Freeze to show the predicted face
            if f == n or score != "":
                isFreeze = True
                f = 0
                freeze_img = rekog_frame.copy()
                # Draw label class prediction
                cv2.rectangle(freeze_img, (x, y+h + 65), (x+w, y+h), (0, 0, 0), cv2.FILLED)
                if pred == "":
                    cv2.rectangle(freeze_img, (x, y), (x+w, y+h), (0,155,255), cv2.FILLED)
                    cv2.putText(freeze_img, "Not Employee!", (x + 3, y+h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                elif pred != "":
                    cv2.putText(freeze_img, f"{pred}", (x + 3, y+h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    cv2.putText(freeze_img, f"{score:.2f}", (x + 3, y+h + 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                else:
                    None
                cv2.imshow(f"Hello {pred}", freeze_img)
                cv2.waitKey(3000)
                cv2.destroyWindow(f"Hello {pred}")

        cv2.imshow('Hello Welcome to iNews Tower', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            video.release()
            cv2.destroyAllWindows()
            name = input("Enter your name: ")
            register(name=name)
            rekog()

    video.release()
    cv2.destroyAllWindows()
