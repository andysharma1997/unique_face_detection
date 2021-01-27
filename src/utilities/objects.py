from src.utilities import sken_logger
from sklearn.cluster import DBSCAN
import numpy as np

logger = sken_logger.get_logger('object')


class DetectedFaces:
    def __init__(self, face_encodings=None):
        self.face_encodings = face_encodings

    def get_face_encoding(self):
        return self.face_encodings


class Video:
    def __init__(self, name, path, total_frames):
        self.name = name
        self.path = path
        self.total_frames = total_frames
        self.all_face_encodings = []

    def put_detected_face_encodings(self, face_encodings: DetectedFaces):
        if len(face_encodings.get_face_encoding()) > 0:
            self.all_face_encodings.extend(face_encodings.get_face_encoding())

    def get_unique_face(self):
        if len(self.all_face_encodings) > 0:
            encodings = [item for item in self.all_face_encodings]
            logger.info("Creating clusters for video = {} ".format(self.name))
            clt = DBSCAN(metric='euclidean', n_jobs=-1)
            clt.fit(encodings)
            label_ids = np.unique(clt.labels_)
            num_unique_faces = len(np.where(label_ids > -1)[0])
            logger.info("# unique faces = {}".format(num_unique_faces))
            return num_unique_faces
        else:
            logger.warning("No faces were detected in the video={}".format(self.name))
            return -1
