import cv2
from recognition import recognize
from time import sleep
from utils import load_models

def webcam():

    models = load_models()

    iteration = 0

    cam = cv2.VideoCapture(0)

    while True:
        
        ret_val, image = cam.read()
        cv2.imshow('HackDrone', image)
        
        if iteration > 3:
            # RECOGNITION
            print('Recognition')
            letter, prob = recognize(image, models)

        sleep(2)

        iteration += 1

        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam()