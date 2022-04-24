import numpy as np
import cv2

def stack_img(img1, img2, x, y):
    
    output = np.hstack((np.zeros((img1.shape[0], x, 3)), img1))

    if y < 0:
        output[:y, :x, :] = img2[abs(y):, :x, :]
    elif y > 0:
        output[y:, :x, :] = img2[:-y, :x, :]
    else:
         output[:, :x, :] = img2[:, :x, :]

    return output.astype('uint8')

def linear_blending(src, tar, x, y, percent = 0.2):
    th, tw, c = tar.shape
    sh, sw, c = src.shape

    overlap_x = tw - x
    blend_range_x = round(overlap_x * percent)
    tend_x, sstart_x = tw - round(overlap_x / 2), round(overlap_x / 2)

    tars = np.zeros((tar.shape))
    if y < 0:
        tars[:y] = tar[abs(y):]
    elif y > 0:
        tars[y:] = tar[:-y]
    else:
        tars = tar

    # shift
    blend_img = np.concatenate((tars[:, :tend_x], src[:, sstart_x:]), axis=1)

    for j in range(0, blend_range_x):
        tratio = 0.5 - j / blend_range_x
        if tratio < 0:
            tratio = 0
        sratio = 1 - tratio
        blend_img[:, tend_x+j] = src[:, sstart_x+j] * sratio + tars[:, tend_x+j] * tratio
        sratio, tratio = tratio, sratio
        blend_img[:, tend_x-j] = src[:, sstart_x-j] * sratio + tars[:, tend_x-j] * tratio
    
    blend_img = blend_img.astype(np.uint8)
    return blend_img
    
    
    



        