from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
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

def feature_matching(v1, v2, thres=0.8):

    distances = cdist(v1, v2)
    sorted_index = np.argsort(distances, axis=1)
    
    matches = []
    for i, si in enumerate(sorted_index):
        first = distances[i, si[0]]
        second = distances[i, si[1]]
        if first / second < thres:
            matches.append([i, si[0]])
    
    print('found matches:', len(matches))
    return matches

# img0 = np.load('0_blur.npy')
# img1 = np.load('1_blur.npy')
# key0 = np.load('0_keypoint.npy')
# key1 = np.load('1_keypoint.npy')
# des0 = np.load('0_descriptor.npy')
# des1 = np.load('1_descriptor.npy')

# matches = find_matches(des0, des1)
# plot_matches(img0, img1, key0, key1, matches)