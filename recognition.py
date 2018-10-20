import os
import cv2
import numpy as np
from windows import pyramid, sliding_window
from utils import crop
import time
from classify import classify_gesture
from detect import hand_detection

SIDE = 100
winW, winH = 128, 128
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 
'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def recognize(image):

    print('Size: %s' % (image.size))
    print('Shape:', image.shape)
    print('Converting to gray:')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('Shape:', gray_image.shape)
    img_scaled = cv2.resize(gray_image, (200, 300), interpolation=cv2.INTER_AREA)
    print('Shape:', img_scaled.shape)

    # Create sliding window
    for resized in pyramid(img_scaled, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW
            window = crop(resized, x, x + winW, y, y + winH) # Side
            if True: # detection(window):
                
                print('Hand Detected!')
                window = cv2.resize(window, (28, 28), interpolation=cv2.INTER_AREA)
                window = window.reshape(window.shape[0], window.shape[1], 1)
                # Classification
                probs = classify_gesture(window)
                index = np.argmax(probs[0])
                letter = letters[index]
                print ('Letter: %s, Index: %d' % (letter, index))
                
                return letter
            
            print('No hand detected on this window')

            # Draw the window
            # clone = resized.copy()
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.imshow('Window', clone)
            # plt.imshow(clone)
            # filename = 'test.png'
            # cv2.imwrite(filename, clone)
            # cv2.waitKey(1)

            time.sleep(0.025)

        return None

if __name__ == '__main__':
    data_dir = 'data/sample/'
    for filename in os.listdir(data_dir):
        print('Analysing: %s' % (filename))
        filename = os.path.join(data_dir, filename)
        image = cv2.imread(filename)
        print('Recogize:')
        recognize(image)
        print('Recognition done!')
        time.sleep(2)
        break
