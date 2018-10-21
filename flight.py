import sys
import traceback
import tellopy
import av
import pygame
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
from time import sleep
from recognition import recognize

def dronesign():
    drone = tellopy.Tello()
    try:
        drone.connect()
        drone.wait_for_connection(60.0)
        drone.takeoff()
        sleep(5)
        drone.down(50)
        sleep(5)
        drone.start_video()
        container = av.open(drone.get_video_stream())
        
        for frame in container.decode(video=0):
            
            image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            
            # RECOGNITION
            print('Recognition')
            result = recognize(image)
            if result is not None:
                letter, prob = result
                text = 'Letter %s, confidence: %.2f' % (letter, prob)
                print(text)

            sleep(5)
            stop = raw_input('Stop? [y]')
            if stop.lower() == 'y':
                break

        sleep(5)
        drone.land()
        sleep(5)
    except Exception as ex:
        # exc_type, exc_value, exc_traceback = sys.exc_info()
        # traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    dronesign()
