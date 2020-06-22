import cv2
import os
from PIL import Image
import numpy as np
import pickle


def traning():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    X_train = []
    Y_train = []
    current_id = 0
    label_id = {}

    BASE_DIS = os.path.dirname(os.path.abspath(__file__))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image_dir = os.path.join(BASE_DIS, "image")
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace("", "").lower()
                if not label in label_id:
                    label_id[label] = current_id
                    current_id += 1
                id_ = label_id[label]
                pil_image = Image.open(path).convert("L")
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    X_train.append(roi)
                    Y_train.append(id_)

    with open("label.pickle", 'wb') as f:
        pickle.dump(label_id, f)

    recognizer.train(X_train, np.array(Y_train))
    recognizer.save("trainner.yml")
