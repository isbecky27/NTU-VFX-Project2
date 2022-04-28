from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

def draw_matches(img1, img2, des1, des2, matches, i = 0):

    matching_img = np.hstack((img1, img2))
    
    plt.clf()
    for idx1, idx2 in matches:
        y1, x1 = des1[idx1]
        y2, x2 = des2[idx2]
        plt.plot([x1, img1.shape[1] + x2], [y1, y2], marker = 'o', linewidth = 1, markersize = 3)
    
    plt.imshow(cv2.cvtColor(matching_img, cv2.COLOR_BGR2RGB))
    # plt.savefig(f'matches-{i}.png')
    plt.show()

def feature_matching(des1, des2, threshold = 0.6):
    
    distances = distance.cdist(des1, des2)
    sorted_dist_idx = np.argsort(distances, axis = 1)
    
    matches = []
    for i, idx in enumerate(sorted_dist_idx):
        best_match, second_best  = distances[i, idx[0]], distances[i, idx[1]]

        # if ratio is high, it could be ambiguous match.
        ratio = best_match / second_best
        if ratio < threshold:
            matches.append([i, idx[0]])

    if len(matches) <= 0:
        matches = feature_matching(des1, des2, threshold + 0.05)

    return matches

def RANSAC_matches(kp1, kp2, matches, iteration = 10, threshold = 5):
    
    kp1, kp2, matches = np.array(kp1), np.array(kp2), np.array(matches) 
    idx1, idx2 = matches[:, 0], matches[:, 1]

    num = matches.shape[0] // 3 + 1

    distances = np.linalg.norm(kp1[idx1] - kp2[idx2], axis = 1)
    translation = kp1[idx1] - kp2[idx2]
    translation_x = translation[:, 1]

    good_matches_idx = []
    iter = 0
    while iter < iteration:

        rand_trans_x = random.choices(translation_x, k = 5)
        mean_trans_x = np.mean(rand_trans_x)
      
        matches_idx = np.where(abs(translation_x - mean_trans_x) <= threshold)[0]

        if len(matches_idx) > len(good_matches_idx):
            good_matches_idx = matches_idx
        
        iter += 1
        if  iter >= iteration and len(good_matches_idx) < num:
            threshold += 5
            iter = 0
    
    good_matches = matches[good_matches_idx]

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