from cv2 import COLOR_BGR2GRAY
import numpy as np
import argparse
import cv2
import os
import re
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
    
    ## use Harris Corner Detector to get keypoints
    keypoints, localmax_imgs = harris_corner_detector(imgs)
    print('Feature detection...')

    ## feature descriptor
    ## feature matching
    ## image matching
    ## image blending
