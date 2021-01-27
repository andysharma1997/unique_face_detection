import argparse

from src.utilities import model_creation
from src.services import unique_people_detection

#
# def argsParse():
#     arg_parse = argparse.ArgumentParser()
#     arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
#     arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
#     arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
#     arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
#     args = vars(arg_parse.parse_args())
#
#     return args
#
#
# if __name__ == "__main__":
#     model_creation.HumanDetection.get_instance()
#     args = argsParse()
#     model_creation.HumanDetection.get_instance().humanDetector(args)

if __name__ == '__main__':
    path = "/home/andy/Downloads/test.mp4"
    result = unique_people_detection.detect_by_video_path(path)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(result)