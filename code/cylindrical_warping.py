import numpy as np
import math
import cv2

def cylindrical_warping(img, f):

    h, w, c = img.shape
    x0, y0 = w // 2, h // 2 # center

    img_warp, s = np.zeros((img.shape)), f
    for y in range(h):
      for x in range(w):
        h = (y - y0) / math.sqrt((x - x0) ** 2 + f ** 2)
        theta = np.arctan((x - x0) / f)
        x_prime, y_prime = round(s * theta), round(s * h)
        img_warp[y0 + y_prime, x0 + x_prime] = img[y, x]
    
    pts = [[h/2, 0], [h/2, w-1]]
    pts_warp = cylindrical_warping_pts(pts, f, h, w)
    left, right = pts_warp[0][1], pts_warp[1][1]
    
    return img_warp.astype('uint8')[:, left:right+1]

def cylindrical_warping_pts(pts, f, h, w):

    x0, y0, s = w / 2, h / 2, f

    pts_warp = []
    for y, x in pts:
        h = (y - y0) / math.sqrt((x - x0) ** 2 + f ** 2) * s
        theta = np.arctan((x - x0) / f)
        pts_warp.append([round(y0 + h), round(x0 + s * theta)])
    
    return pts_warp