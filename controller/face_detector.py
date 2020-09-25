import cv2
from model.face_model import Face
from skimage.metrics import structural_similarity as ssim


class FaceDetector:
    def __init__(self):
        self.face_det = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

    def detect_faces(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.face_det.detectMultiScale(grayscale, minNeighbors=8)
        faces = list()
        for (x, y, w, h) in dets:
            face_img = img[y:y + h, x:x + w].copy()
            faces.append(Face(face_img, (x, y, w, h)))
        return faces

    def compare_faces(self, face_a, face_b):
        a_gray = cv2.cvtColor(face_a.img, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(face_b.img, cv2.COLOR_BGR2GRAY)
        b_resized = cv2.resize(b_gray, a_gray.shape)
        a_blurred = cv2.bilateralFilter(cv2.medianBlur(a_gray, 5), 5, 75, 75)
        b_blurred = cv2.bilateralFilter(cv2.medianBlur(b_resized, 5), 5, 75, 75)
        return ssim(a_blurred, b_blurred)
