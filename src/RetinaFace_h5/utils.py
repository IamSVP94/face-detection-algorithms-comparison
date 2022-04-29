import cv2
from retinaface import RetinaFace

detect_model = RetinaFace.build_model()


def find_face(filename, threshold=0.8, model=detect_model):
    image = cv2.imread(str(filename))
    face_landmarks = RetinaFace.detect_faces(str(filename), model=model, threshold=threshold)
    return image, face_landmarks
