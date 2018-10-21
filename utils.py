from skimage.transform import resize
from keras.models import load_model


def crop(img, x1, x2, y1, y2):
    crp = img[y1:y2, x1:x2]
    crp = resize(crp, (128,128), mode='constant') # resize
    return crp


def load_models():
    print('Load models')
    return {
        'binary_hand_model': load_model('binary_hand_model.h5'),
        'recognition_sign_model': load_model('recognition_sign_model.h5'),
    }