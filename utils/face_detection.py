import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        return image

    x,y,w,h = faces[0]

    face = image[y:y+h, x:x+w]

    return face