from controller.face_recognizer import FaceRecognizer
from threading import Thread


class Worker(Thread):
    def __init__(self, request_queue, response_queue):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.face_rec = FaceRecognizer()

    def run(self):
        while True:
            face = self.request_queue.get()
            if face is None:
                break
            identity = self.face_rec.recognize_face(face)
            face.identity = identity
            self.response_queue.put(face)
