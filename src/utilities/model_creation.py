import cv2
import imutils
import numpy as np
import argparse


class HumanDetection:
    __instance__ = None
    HOGCV = None

    @staticmethod
    def get_instance():
        if HumanDetection.__instance__ is None:
            HumanDetection()
        return HumanDetection.__instance__

    def __init__(self):
        if HumanDetection.__instance__ is not None:
            raise Exception("This is a singleton class can't be initialised more thane once")
        else:
            # self.HOGCV = cv2.HOGDescriptor()
            # self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.HOGCV = cv2.CascadeClassifier(
                '/home/andy/Desktop/zoom/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
            HumanDetection.__instance__ = self

    def detect(self, frame):
        # bounding_box_cordinates, weights = self.HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8),
        #                                                                scale=1.03)
        bounding_box_cordinates = self.HOGCV.detectMultiScale(frame, 1.1, 4)
        person = 1
        print(bounding_box_cordinates)
        for x, y, w, h in bounding_box_cordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            person += 1

        cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('output', frame)
        return frame

    def detectByPathVideo(self, path, writer):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(writer)
        video = cv2.VideoCapture(path)
        check, frame = video.read()
        if check == False:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return

        print('Detecting people...')
        while video.isOpened():
            # check is True if reading was successful
            check, frame = video.read()

            if check:
                frame = imutils.resize(frame, width=min(800, frame.shape[1]))
                frame = self.detect(frame)

                if writer is not None:
                    writer.write(frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()

    def detectByPathImage(self, path, output_path):
        image = cv2.imread(path)

        image = imutils.resize(image, width=min(800, image.shape[1]))

        result_image = self.detect(image)

        if output_path is not None:
            cv2.imwrite(output_path, result_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def humanDetector(self, args):
        image_path = args["image"]
        video_path = args['video']

        writer = None
        if args['output'] is not None and image_path is None:
            writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
        if video_path is not None:
            print('[INFO] Opening Video from path.')
            self.detectByPathVideo(video_path, writer)
        elif image_path is not None:
            print('[INFO] Opening Image from path.')
            self.detectByPathImage(image_path, args['output'])
