PROCESSING = 0
UNKNOWN = 1
RECOGNIZED = 2


class Face:
    def __init__(self, img, coordinates, face_id='', identity=None, state=PROCESSING):
        self.img = img
        self.coordinates = coordinates
        self.face_id = face_id
        self.identity = identity
        self.state = state
