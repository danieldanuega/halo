import cv2
from mtcnn import MTCNN

detector = MTCNN()

video = cv2.VideoCapture(0)
if not video.isOpened():
    print('Error opening the camera')

while True:
    # Open the camera
    ret, frame = video.read()

    # Resize the frame size and convert it to RGB Color
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = frame[:, :, ::-1]

    # Detect the face from rgb input
    results = detector.detect_faces(rgb_small_frame)

    if results:
        for face in results:
            bounding_box = face['box']
            keypoints = face['keypoints']

            cv2.rectangle(frame,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,155,255),
                        2)

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