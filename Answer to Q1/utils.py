import cv2
import numpy as np

def saveImages(images, size, path):
    h, w = images.shape[1], images.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    result = merge_img * 255.
    result = np.clip(result, 0, 255).astype('uint8')
    return cv2.imwrite(path, result)