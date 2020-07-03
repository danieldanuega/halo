import cv2
import helper
from model import get_input_shape
from PIL import Image
from datetime import datetime
import os

now = datetime.now()
this_time = now.strftime("%b-%d-%Y--%H:%M:%S")


def register(database='./database', name='temp'):
    file_path = os.path.join(database,name)
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
            if os.path.exists(file_path) == False:
                os.makedirs(file_path)
            cv2.imwrite(os.path.join(file_path, this_time), frame)
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

if __name__ == '__main__':
    register(name='daniel')