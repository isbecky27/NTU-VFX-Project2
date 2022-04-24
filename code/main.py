from cylindrical_warping import *
from feature_description import SIFT_descriptor
from feature_matching import *
from image_blending import * 
from harris import *
import numpy as np
import argparse
import cv2
import os
import re
import math

def read_imgs_and_focals(path, filename):
    '''
    Input :
        path : the folder path to the .txt file
        filename : filename of .txt which stores the information of images
    Output:
        imgs : list of p images ( h x w x c )
        fs : list of focal length
    '''

    with open(os.path.join(path, filename)) as f:
        content = f.readlines()

    imgs, focals = [], []
    for line in content:
        if '.jpg' in line:
            imgfile = re.findall(r"[a-z0-9]*.jpg", line)
            img = cv2.imread(os.path.join(path, imgfile[0]))
            imgs.append(img)
        elif len(line.split()) == 1:
            focals.append(float(re.findall(r'\d+\.\d+', line)[0]))

    return imgs, focals

if __name__ == '__main__':

    ## add argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '../data/parrington/', help = 'Path to the directory that contains series of images.')
    parser.add_argument('--result_path', type = str, default = '../result/parrington/', help = 'Path to the directory that stores all of results.')
    # parser.add_argument('--series_of_images', type = str, default = 'parrington', help = 'The folder of a series of images that contains images and shutter time file.')
    parser.add_argument('--focal_length_filename', type = str, default = 'pano.txt', help = 'The name of the file where shutter time information is stored.')
    args = parser.parse_args()

    ## variables
    path = args.data_path
    save_path = args.result_path
    filename = args.focal_length_filename
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    ## read images and get the focal length of images
    print('Read images...')
    imgs, focals = read_imgs_and_focals(path, filename)

    ## cylindrical warping
    # print('Cylindrical Warping...')
    # imgs = [cylindrical_warping(img, focal) for img, focal in zip(imgs, focals)]

    keypts, descriptors = [], []
    for img in imgs:
        ## use Harris Corner Detector to get keypoints
        print('Feature detection...')
        keypoints, blur_img = harris_corner_detector(img)

        ## feature descriptor using SIFT
        print('Feature descriptor...')
        kp, des = SIFT_descriptor(blur_img, keypoints)
        keypts.append(kp)
        descriptors.append(des)
        # np.save(f'{i}_keypoint.npy', keypts)
        # np.save(f'{i}_descriptor.npy', descriptors)
    
    trans_x, trans_y = [], []
    for idx in range(len(imgs)-1):

        ## feature matching
        print('Feature matching...')
        matches = feature_matching(descriptors[idx], descriptors[idx+1])
        
        ## image matching
        print('Image matching...')
        translation_x, translation_y, good_matches = RANSAC_matches(keypts[idx], keypts[idx+1], matches)
        trans_x.append(translation_x)
        trans_y.append(translation_y)
        # draw_matches(imgs[idx], imgs[idx+1], keypts[idx], keypts[idx+1], good_matches)

    # align the first and the last image
    matches = feature_matching(descriptors[0], descriptors[-1])
    last_x, last_y, good_matches = RANSAC_matches(keypts[0], keypts[-1], matches)
    accu_y = np.cumsum(trans_y)
    dy = 0 #(translation_y - accu_y[-1]) / (len(imgs) - 1)

    output = None
    for idx in range(len(imgs)-1):
       
        ## image blending
        print('Image blending...')
        if idx == 0:
            output = linear_blending(imgs[idx], imgs[idx+1], trans_x[idx], round(accu_y[idx] + dy * idx))
        else:
            output = linear_blending(output, imgs[idx+1], trans_x[idx], round(accu_y[idx] + dy * idx))
      
        # cv2.imshow('show blending', output)
        # cv2.waitKey(0)

    output = output[:, round(last_x / 2):]
    # cv2.imwrite(save_path + 'linear_without_global.png', output)
    cv2.imshow('Result', output)
    cv2.waitKey(0)