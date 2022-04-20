from cv2 import COLOR_BGR2GRAY
import numpy as np
import argparse
import cv2
import os
import re
from feature_description import SIFT_descriptor
from feature_matching import *
from harris import *

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
    parser.add_argument('--result_path', type = str, default = '../result/', help = 'Path to the directory that stores all of results.')
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

    # cylindrical warping
    print('Cylindrical Warping...')
    '''
    TODO
    '''

    i = 0
    keypts, descriptors = [], []
    for img in imgs:

        if i > 4: break
        i += 1

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

    output = None
    for idx in range(4): # len(imgs)-1

        ## feature matching
        print('Feature matching...')
        matches = feature_matching(descriptors[idx], descriptors[idx+1])
        
        ## image matching
        print('Image matching...')
        print(imgs[idx].shape)
        translation, good_matches = RANSAC_matches(keypts[idx], keypts[idx+1], matches)
        # plot_matches(imgs[idx], imgs[idx+1], keypts[idx], keypts[idx+1], good_matches)

        ## image blending
        '''
        TODO
        '''
        if idx == 0:
            output = np.hstack((imgs[idx+1][:, :int(translation)], imgs[idx]))
        else:
            output = np.hstack((imgs[idx+1][:, :int(translation)], output))
        cv2.imshow('show blending', output)
        cv2.waitKey(0)
