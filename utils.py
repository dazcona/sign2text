from skimage.transform import resize

def crop(img, x1, x2, y1, y2):
    crp = img[y1:y2, x1:x2]
    crp = resize(crp, (128,128), mode='constant') # resize
    return crp