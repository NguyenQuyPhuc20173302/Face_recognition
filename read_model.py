import cv2
import numpy as np

import process_data

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
list_name = ['DAO MINH DUC',
             'MAI THU TRANG',
             'NGUYEN QUY PHUC']
model = process_data.Model_(len(list_name))
model.load_weights("model.h5")


def name_predict(image):
    pre = model.predict(image)
    pre = pre[0]
    max_ = max(pre)
    if max_ >= 0.75:
        for i in range(len(list_name)):
            if max_ == pre[i]:
                return list_name[i] + ':' + str(round(max_ * 100, 2)) + '%'
    else:
        return ''


IMG_SIZE = 224


def take_image():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, )
            xam = gray[y:y + h, x:x + h]
            xam = cv2.resize(xam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            xam = np.array(xam).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
            xam /= 255.0
            name = name_predict(xam)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def read_image(file_path):
    img = cv2.imread(file_path, 0)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


'''path = 'p.jpg'
img = read_image(path)
img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
img /= 225.0
a = model.predict(img)
a = a[0]
for i in range(len(list_name)):
    print(list_name[i] + "\t\t:" + str(a[i] * 100) + "%")
'''
take_image()
