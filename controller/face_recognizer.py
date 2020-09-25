import dlib
import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
from cnn import cnn_model as cm
from model.db_handler import DBHandler
from controller.face_detector import FaceDetector


class FaceRecognizer:
    def __init__(self, cs_threshold=0.31):
        self.cs_threshold = cs_threshold
        self.model = cm.get_model(model_path='resources/model_v2.h5')
        self.face_det = FaceDetector()
        self.shape_pred = dlib.shape_predictor('resources/shape_predictor_5_face_landmarks.dat')
        self.handler = DBHandler.get_instance()

    def align_face(self, face):
        x, y, z = face.img.shape
        rect = dlib.rectangle(0, 0, x, y)
        landmarks = self.shape_pred(face.img, rect)
        return dlib.get_face_chip(face.img, landmarks, 128)

    def normalize_face(self, face):
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        normalized = face.astype(np.float32) / 255
        normalized = normalized.reshape(1, normalized.shape[0], normalized.shape[1], 1)
        return normalized

    def recognize_face(self, face):
        aligned = self.align_face(face)
        normalized = self.normalize_face(aligned)

        feature_vector = self.model.get_feature_vector(normalized)
        person, max_similarity = None, -1
        for identity in self.handler.get_identities().values():
            # Cosine similarity
            similarity = dot(identity.features, feature_vector) / (norm(identity.features) * norm(feature_vector))
            print('Similarity with {} {} -> {}'.format(identity.name, identity.surname, similarity))
            if similarity > self.cs_threshold and similarity > max_similarity:
                person = identity
                max_similarity = similarity

        return person

    def get_facial_features(self, img):
        faces = self.face_det.detect_faces(img)
        if len(faces) > 1:
            raise ValueError('Multiple faces detected! Please provide image containing only one face')
        elif len(faces) == 0:
            raise ValueError('No faces detected! Please provide image containing at least one face')

        aligned = self.align_face(faces[0])
        normalized = self.normalize_face(aligned)
        return self.model.get_feature_vector(normalized)
