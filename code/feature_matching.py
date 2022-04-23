from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

def plot_matches(im1, im2, des1, des2, matches, i = 0):
    h1, w1, c = im1.shape
    h2, w2, c = im2.shape
    vis = np.zeros([max(h1, h2), w1 + w2, c], dtype=np.uint8) + 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:] = im2

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(vis)

    for (idx1, idx2) in matches:
        y1, x1 = des1[idx1]
        y2, x2 = des2[idx2]
        ax.plot([x1, w1 + x2], [y1, y2], marker = 'o', linewidth = 1, markersize = 4)
    
    # plt.savefig('matches-%d.png' % i)
    plt.show()

def feature_matching(des1, des2, threshold = 0.8):

    distances = cdist(des1, des2)
    sorted_index = np.argsort(distances, axis = 1)
    
    matches = []
    for i, idx in enumerate(sorted_index):
        first  = distances[i, idx[0]]
        second = distances[i, idx[1]]
        if first / second < threshold:
            matches.append([i, idx[0]])

    return matches

def RANSAC_matches(kp1, kp2, matches, iteration = 10):

    kp1, kp2, matches = np.array(kp1), np.array(kp2), np.array(matches) 
    idx1, idx2 = matches[:, 0], matches[:, 1]

    distances = np.linalg.norm(kp1[idx1] - kp2[idx2], axis = 1)

    good_matches_idx = []
    for iter in range(iteration):
        rand_dists = random.choices(distances, k = 5)
        mean_dists = np.mean(rand_dists)
        
        matches_idx = np.where(abs(distances - mean_dists) <= 3)[0]
        
        if len(matches_idx) > len(good_matches_idx):
            good_matches_idx = matches_idx
    
    good_matches = matches[good_matches_idx]

    translation = kp1[idx1] - kp2[idx2]
    mean_translation = np.mean(translation[good_matches_idx], axis = 0)
    translation_y, translation_x = mean_translation[0], mean_translation[1]

    return abs(round(translation_x)), round(translation_y), good_matches
  
# img0 = np.load('0_blur.npy')
# img1 = np.load('1_blur.npy')
# key0 = np.load('0_keypoint.npy')
# key1 = np.load('1_keypoint.npy')
# des0 = np.load('0_descriptor.npy')
# des1 = np.load('1_descriptor.npy')

# matches = find_matches(des0, des1)
# plot_matches(img0, img1, key0, key1, matches)