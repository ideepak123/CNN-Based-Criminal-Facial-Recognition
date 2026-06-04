import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}

current_label = 0

# Face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# Read dataset
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face = cv2.equalizeHist(face)

            faces.append(face)
            labels.append(current_label)

    current_label += 1


# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

# Save model
recognizer.save("trained_model.yml")

print("Model trained successfully!")
print(label_map)