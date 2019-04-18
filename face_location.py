import cv2
import os


def get_face_locations(image):
    detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # image = cv2.imread('images/pic.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    base = os.path.abspath(os.path.dirname(__file__))
    base = os.path.join(base, "face_detected")
    i = 0
    for(x,y,w,h) in faces:
        face_image = image[y:y+h,x:x+w]
        temp = os.path.join(base, "face-"+str(i))
        cv2.imwrite(temp + '.jpg', face_image)
        i = i + 1