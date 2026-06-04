import cv2
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

# Labels
label_map = {
    0: "deepak",
    1: "narendra"
}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)


def recognize_person(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return "Unknown", "0%"

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        confidence_score = round(
            100 - confidence
        )

        if confidence_score > 50:
            return (
                label_map[label],
                f"{confidence_score}%"
            )

    return "Unknown Person", "0%"