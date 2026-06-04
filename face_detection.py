import cv2


def detect_face(image_path):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml'
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(100,100)
    )

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return image