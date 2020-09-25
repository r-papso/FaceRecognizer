import cv2
import uuid
from model import face_model as fm
from controller.face_detector import FaceDetector
from controller.worker import Worker
from queue import Queue, Empty


class FrameController:
    def __init__(self, sim_threshold=0.54, num_workers=2, max_queue_size=10, frame_proc_freq=3):
        self.frame_counter = 0
        self.frame_proc_freq = frame_proc_freq
        self.sim_threshold = sim_threshold
        self.present_faces = dict()
        self.face_det = FaceDetector()
        self.request_queue = Queue(maxsize=max_queue_size)
        self.response_queue = Queue(maxsize=max_queue_size)
        self.workers = list()
        for i in range(num_workers):
            worker = Worker(self.request_queue, self.response_queue)
            worker.start()
            self.workers.append(worker)

    def stop(self):
        for i in range(len(self.workers)):
            self.request_queue.put(None)
        for worker in self.workers:
            worker.join()

    def clear_face_list(self):
        self.present_faces.clear()

    def process_frame(self, frame):
        if self.frame_counter % self.frame_proc_freq == 0:
            faces = self.face_det.detect_faces(frame)
            self.face_correlation(faces)
            self.frame_counter = 0
        self.frame_counter += 1

        self.get_results()
        self.label_faces(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def face_correlation(self, new_faces):
        new_faces_dict = dict()
        for new_face in new_faces:
            max_similarity, joint_index = 0, ''
            for key, present_face in self.present_faces.items():
                similarity = self.face_det.compare_faces(new_face, present_face)
                if similarity > max_similarity:
                    max_similarity = similarity
                    joint_index = key
            if max_similarity < self.sim_threshold:
                new_face.face_id = uuid.uuid4().hex
                self.request_queue.put(new_face)
            else:
                joint_face = self.present_faces[joint_index]
                new_face.face_id = joint_face.face_id
                new_face.identity = joint_face.identity
                new_face.state = joint_face.state
            new_faces_dict[new_face.face_id] = new_face
        self.present_faces = new_faces_dict

    def label_faces(self, frame):
        for face in self.present_faces.values():
            (x, y, w, h) = face.coordinates
            if face.state == fm.UNKNOWN:
                color = (0, 0, 255)
                label = 'Unknown person'
            elif face.state == fm.RECOGNIZED:
                color = (0, 255, 0)
                label = '{} {}'.format(face.identity.name, face.identity.surname)
            else:
                color = (255, 0, 0)
                label = 'Processing...'
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)

    def get_results(self):
        while True:
            try:
                face = self.response_queue.get_nowait()
                if face.face_id in self.present_faces:
                    joint_face = self.present_faces[face.face_id]
                    joint_face.identity = face.identity
                    joint_face.state = fm.RECOGNIZED if face.identity is not None else fm.UNKNOWN
            except Empty:
                break
