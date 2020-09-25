from model.db_handler import DBHandler
from controller.face_recognizer import FaceRecognizer
import cv2


class IdentityController:
    def __init__(self):
        self.db_handler = DBHandler.get_instance()
        self.face_rec = FaceRecognizer()

    def get_identities(self):
        return self.db_handler.get_identities()

    def add_identity(self, identity, img_file_path):
        if not img_file_path:
            raise ValueError('Please specify path to image of person`s face.')
        if not identity.name or not identity.surname:
            raise ValueError('Please specify person`s name and surname')
        if not self.valid_img_file_ext(img_file_path):
            raise ValueError('Please specify path to valid image file')

        img = cv2.imread(img_file_path)
        facial_vector = self.face_rec.get_facial_features(img)
        identity.features = facial_vector
        self.db_handler.add_identity(identity)

    def edit_identity(self, identity, img_file_path):
        if not identity.name or not identity.surname:
            raise ValueError('Please specify person`s name and surname')

        if img_file_path:
            if not self.valid_img_file_ext(img_file_path):
                raise ValueError('Please specify path to valid image file')

            img = cv2.imread(img_file_path)
            facial_vector = self.face_rec.get_facial_features(img)
            identity.features = facial_vector
        self.db_handler.edit_identity(identity)

    def delete_identity(self, identity_id):
        self.db_handler.delete_identity(identity_id)

    def valid_img_file_ext(self, file_path):
        return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
