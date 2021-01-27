from keras_facenet import FaceNet
import cv2
from src.utilities import sken_logger, objects
import imutils
import time

logger = sken_logger.get_logger("keras_implementation")


class FaceDetector:

    def __init__(self):
        self.model = FaceNet()

    def _draw(self, frame, bounding_box_coordinates):
        try:
            persons = 1
            for x, y, w, h in bounding_box_coordinates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f'person {persons}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                persons += 1
            cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f'Total Persons : {persons - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow('output', frame)
        except Exception as e:
            pass

    def run(self, file_path):
        cap = cv2.VideoCapture(file_path)
        video = objects.Video(file_path.split("/")[-1], file_path, cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = cap.read()
            try:
                frame = imutils.resize(frame, width=min(800, frame.shape[1]))
                detections = self.model.extract(frame, threshold=0.95)
                boxes = []
                face_embeddings = []
                for detection in detections:
                    boxes.append(detection['box'])
                    face_embeddings.append(detection['embedding'])
                video.put_detected_face_encodings(objects.DetectedFaces(face_embeddings))
                self._draw(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), boxes)
            except Exception as e:
                pass
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cap.destroyAllWindows()
        return video


if __name__ == '__main__':
    # model_path = '/home/andy/Desktop/zoom/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
    # fcd = FaceDetector(model_path)
    fcd = FaceDetector()
    s = time.time()
    video = fcd.run("/home/andy/Downloads/7_tEmd7Zuw_EKqK.mp4")
    print("Time for face detection={}".format(time.time() - s))
    s = time.time()
    num_unique_faces = video.get_unique_face()
    print("Time for clustering = {}|| No.of Pepole={}".format(time.time() - s, num_unique_faces))


