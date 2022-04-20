import numpy as np
import math
import cv2

def cylindrical_warping(img, f):

    h, w, c = img.shape
    x0, y0 = w / 2, h / 2

    img_warp, s = np.zeros((img.shape)), f
    x = np.array([i for i in range(w)])
    y = np.array([i for i in range(h)])

    for y in range(h):
      for x in range(w):
        h = (y - y0) / math.sqrt((x - x0) ** 2 + f ** 2) * s
        theta = np.arctan((x - x0) / f)
        img_warp[round(y0 + h), round(x0 + s * theta), :] = img[y, x, :]

    return img_warp.astype('uint8')