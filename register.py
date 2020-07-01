import cv2
import helper
from model import get_input_shape
from PIL import Image

def register(file_path):
    r = 0
    R = 11
    isRegister = False

    video = cv2.VideoCapture(0)

    if not video.isOpened():
        raise Exception('Error opening the camera')

    while True:
        ret, frame = video.read()
        
        faces = helper.detectFacesLive(frame)
        
        if len(faces) != 0 and r != R:
            x, y, w, h = faces[0]
            
            # Draw rectangle in face
            cv2.rectangle(frame,
                        (x, y),
                        (x+w, y+h),
                        (0,155,255),
                        2)
            
            r += 1
        elif len(faces) != 0 and r == R:
            r = 0
            cv2.imwrite(file_path, frame)
            isRegister = True
        else:
            r = 0

        if isRegister:
            cv2.destroyWindow("Register Face")
            cv2.imshow("Register Success!", frame)
            break
        else:
            cv2.imshow("Register Face", frame)
        
    video.release()
    cv2.destroyAllWindows()
