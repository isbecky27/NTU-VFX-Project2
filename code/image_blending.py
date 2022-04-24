import numpy as np
import cv2

def stack_img(img1, img2, x, y):
    
    output = np.hstack((np.zeros((img1.shape[0], x, 3)), img1))

    if y < 0:
        output[:y, :x, :] = img2[abs(y):, :x, :]
    else:
        output[y:, :x, :] = img2[:y*-1, :x, :]

    return output.astype('uint8')

def image_blending(src, tar, translation_x, translation_y, percent=0.2):
    th, tw, c = tar.shape
    sh, sw, c = src.shape

    overlap_x = tw - int(translation_x)
    blend_range_x = int(overlap_x * percent)
    tend_x, sstart_x = tw - int(overlap_x / 2), int(overlap_x / 2)
    # crop image
    crop_src = False 
    if th-abs(translation_y) < sh or th == sh:
        crop_src = True
    if translation_y < 0: # shift tar up
        if crop_src:
            src = src[:int(translation_y)]
        sh, sw, c = src.shape
        tar = tar[-int(translation_y):sh-int(translation_y)]
    else:
        if crop_src:
            src = src[int(translation_y):]
        sh, sw, c = src.shape
        tar = tar[sh-int(translation_y):-int(translation_y)]

    # shift
    blend_img = np.concatenate((tar[:, :tend_x], src[:, sstart_x:]), axis=1)

    for j in range(0, blend_range_x):
        tratio = 0.5 - j / blend_range_x
        if tratio < 0:
            tratio = 0
        sratio = 1-tratio
        blend_img[:, tend_x+j] = src[:, sstart_x+j] * sratio + tar[:, tend_x+j] * tratio
        sratio, tratio = tratio, sratio
        blend_img[:, tend_x-j] = src[:, sstart_x-j] * sratio + tar[:, tend_x-j] * tratio
    
    blend_img = blend_img.astype(np.uint8)
    return blend_img
    
    
    



        