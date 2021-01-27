import cv2
import imutils
import numpy as np
import face_recognition
from src.utilities import sken_logger, objects

logger = sken_logger.get_logger('unique_people_detection')


def detect_face(box_coordinates, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    person = 1
    for x, y, w, h in box_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    return frame


def detect_by_video_path(path):
    video_name = path.split("/")[-1]
    logger.info("Reading video={}".format(path))
    video = cv2.VideoCapture(path)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    check, frame = video.read()
    if not check:
        logger.error('Video Not Found. Please Enter a Valid Path {} is INCORRECT'.format(path))
        return -1
    logger.info("Creating video object for file: {}".format(video_name))
    video_obj = objects.Video(video_name, path, total_frames)
    logger.info("Detecting people...")
    while video.isOpened():
        # check is True if reading was successful
        frame_index = 1
        check, frame = video.read()
        if check:
            # Resize frame of video to 1/4 size for faster face recognition processing
            # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_boxes = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_boxes)
            all_face_encodings = []
            for encoding in face_encodings:
                if len(encoding) > 0:
                    all_face_encodings.append(objects.DetectedFaces(face_encodings))
            video_obj.put_detected_face_encodings(all_face_encodings)
            detect_face(face_boxes, rgb_small_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    logger.info("Finding all the unique faces......")
    unique_faces = video_obj.get_unique_face()
    if unique_faces != -1:
        return {"unique_faces": unique_faces}
    else:
        return {"unique_faces": 0}
