# sử dụng video
import cv2
import pickle

def face_R():
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")
    labels = {}
    with open("label.pickle", "rb") as f:
        labels = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, )
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + h]
            id_, conf = recognizer.predict(roi_gray)
            for label in labels:
                if labels[label] is id_:
                    name = label + ':' + str(round(conf, 2)) + '%'

                    color = (255, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (x, y), font, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
