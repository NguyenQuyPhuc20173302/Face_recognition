import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)


def take_image(name):
    i = 0
    while True:
        try:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, )
                xam = gray[y:y + h, x:x + h]
            name1 = 'image/' + name + '/' + str(i) + '.png'
            cv2.imwrite(name1, xam)
            i = i + 1
            if i == 100:
                break
        except:
            continue
    cap.release()
    cv2.destroyAllWindows()


