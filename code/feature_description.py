# feature descriptor using SIFT
import numpy as np
import cv2

def compute_orientation(img, bins = 8):
    Iy, Ix = np.gradient(img)

    # gradient magnitude
    m = np.sqrt(Ix ** 2 + Iy ** 2)

    # gradient orientation
    theta = np.rad2deg(np.arctan(Iy / (Ix + 1e-10))) % 360
    
    return m, theta

def compute_subpatch_descriptor(m, theta):
    histogram = [0 for _ in range(8)]

    m = m.flatten()
    theta = theta.flatten().astype('uint8')
    
    for vote, bin in zip(m, theta):
        histogram[bin] += vote
    
    return histogram

def compute_descriptor(m, theta, bins = 8):

    theta = theta // (360. / bins)
    
    ## calculate weighted gradient magnitude by 2D gaussian kernel
    gaussian_weigth = cv2.GaussianBlur(m, (9, 9), 0)
    
    histogram = []
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            histogram += compute_subpatch_descriptor(gaussian_weigth[i:i+4, j:j+4], theta[i:i+4, j:j+4])

    # normalize
    histogram_norm = [i / sum(histogram) for i in histogram]
    # clip values larger than 0.2
    histogram_norm = np.clip(histogram_norm, min(histogram_norm), 0.2)
    # renormalize
    histogram_norm = [i / sum(histogram) for i in histogram]

    return histogram_norm
    
def SIFT_descriptor(img, keypoints):

    m, theta = compute_orientation(img)

    keypts, descriptors= [], []
    for pts in keypoints:

        row, col = pts

        ## local patch
        sizeM = 16
        if row-sizeM < 0 or col-sizeM < 0 or row+sizeM >= img.shape[1] or col+sizeM >= img.shape[0]:
            continue

        R = cv2.getRotationMatrix2D((sizeM, sizeM), -theta[tuple(pts)], 1)
        m_rotated = cv2.warpAffine(m[row-sizeM:row+sizeM, col-sizeM:col+sizeM], R, (sizeM*2, sizeM*2))
        theta_rotated = cv2.warpAffine(theta[row-sizeM:row+sizeM, col-sizeM:col+sizeM], R, (sizeM*2, sizeM*2))

        gradient_length = m_rotated[8:24, 8:24]
        gradient_theta = theta_rotated[8:24, 8:24]

        if gradient_length.shape != (16, 16):
            continue

        keypts.append(pts)
        descriptors.append(compute_descriptor(gradient_length, gradient_theta))

    return keypts, descriptors