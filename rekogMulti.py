import os
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from train import FaceRecognition as FR

# n -> fps count
# F -> desired framerate
# i -> counter value for naming filename
n = 0
F = 30
i = 0
required_size = (224,224)
model_name = "face_recognition.h5" 
class_name = "face_recognition_class_names.npy"

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

detector = MTCNN()

# Using webcam
video = cv2.VideoCapture(0)
# Using gstreamer
# video = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not video.isOpened():
    print('Error opening the camera')

while True:
    # Open the camera
    ret, frame = video.read()

    if n == F:
        n = 0

    n += 1
    i += 1
    if n in range(4):
        # Resize the frame size and convert it to RGB Color
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = frame[:, :, ::-1]

        # Detect the face from rgb input
        results = detector.detect_faces(rgb_small_frame)

        if results:
            for face in results:
                bounding_box = face['box']
                keypoints = face['keypoints']
                x, y, width, height = bounding_box
                
                # Get and Resize face into (224,224)
                face_frame = frame[y:y+height, x:x+width].copy()
                try:
                    face_image = Image.fromarray(face_frame)
                except ValueError:
                    continue
                face_image = face_image.resize(required_size)
                face_array = np.asarray(face_image)
                
                # Predict the frame
                pred = FR.model_prediction(frame, face_array, face_frame, os.path.join("model", model_name), os.path.join("model", class_name))

                # Draw rectangle in face
                cv2.rectangle(frame,
                            (x, y),
                            (x+width, y+height),
                            (0,155,255),
                            2)
                
                # Draw label class prediction
                cv2.rectangle(frame, (x, y+height - 35), (x+width, y+height), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, pred, (x + 6, y+height - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                # Draw pinpoint in eyes, node, and mouth
                cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)

    cv2.imshow('Video MTCNN', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
