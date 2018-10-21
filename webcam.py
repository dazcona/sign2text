import cv2
from recognition import recognize
from time import sleep
from utils import load_models

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (400, 700)
fontScale              = 1
fontColor              = (255, 255, 255)
lineType               = 2

def webcam():

    models = load_models()

    iteration = 0

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('HackDrone', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('HackDrone',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        
        ret_val, image = cam.read()
        
        if iteration > 5:
            # RECOGNITION
            print('Recognition')
            result = recognize(image, models)
            if result is not None:
                letter, prob = result
                text = 'Letter %s, confidence: %.2f' % (letter, prob)
                cv2.putText(image, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        
        cv2.imshow('HackDrone', image)

        sleep(1)

        iteration += 1

        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam()