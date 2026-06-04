import cv2
import numpy as np
from keras.models import load_model

# Load trained CNN model
model = load_model("cnn_model.h5")

# Labels
label_map = {
    0: "deepak",
    1: "narendra"
}


def recognize_person_cnn(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return "Unknown Person", "0%"

    # Resize image
    image = cv2.resize(image, (224, 224))

    # Normalize image
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)

    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)

    confidence_score = round(
        confidence * 100
    )

    if confidence_score > 60:

        return (
            label_map[predicted_class],
            f"{confidence_score}%"
        )

    return "Unknown Person", "0%"