import cv2
import os


def take_image():
    cap = cv2.VideoCapture(0)
    frame = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    """name = 'pic'
    base = os.path.abspath(os.path.dirname(__file__))
    base = os.path.join(base, "images")
    base = os.path.join(base, name)
    cv2.imwrite(base + '.jpg', frame)"""
    cap.release()
    cv2.destroyAllWindows()
    return frame