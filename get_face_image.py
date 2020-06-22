import cv2
import numpy as np

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
IMG_SIZE = 224


def take_image(path, name):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, )
        xam = gray[y:y + h, x:x + h]
        xam = cv2.resize(xam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(name, xam)

take_image('phuc.jpg', 'p.jpg')