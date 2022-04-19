import numpy as np
import copy
import cv2
# from skimage.feature import peak_local_max
# from scipy import signal as sig
# from gc import get_threshold
# from sympy import det

def output_img(img, name):
    cv2.imwrite(f'../myresult/{name}.png', img)

def compute_R(img, ksize = 3, s = 3, k = 0.04):
    '''
        compute R for each image
    '''
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), s)
    # Ix, Iy =  cv2.Sobel(np.array(blur_img), cv2.CV_64F, 1, 0), cv2.Sobel(np.array(blur_img), cv2.CV_64F, 0, 1)
    # Ix, Iy =  cv2.Scharr(np.array(blur_img), cv2.CV_64F, 1, 0), cv2.Sobel(np.array(blur_img), cv2.CV_64F, 0, 1)
    Iy, Ix = np.gradient(blur_img)
    Ix2, Iy2, Ixy = Ix ** 2, Iy ** 2, Ix * Iy
    Sx2, Sy2, Sxy = cv2.GaussianBlur(Ix2, (ksize, ksize), s), cv2.GaussianBlur(Iy2, (ksize, ksize), s), cv2.GaussianBlur(Ixy, (ksize, ksize), s)
    '''
        M = [[Sx2, Sxy], 
            [S`xy, Sy2]]
    '''
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k * traceM ** 2
    
    return R
 
def find_local_maximum(R, k):
    '''
        k: window size
    '''
    h, w = R.shape
    localmax_img = np.zeros((h, w), dtype=int)
    localmax_pts = []
    
    for i in range(0, h-(k-1)):
        for j in range(0, w-(k-1)):
            arr = R[i:i+k]
            arr = np.array([arr[_][j:j+k] for _ in range(k)])
            localmax = np.amax(arr)
            if localmax < 10: # black
                continue
            x, y = np.where(arr == localmax)
            maxi, maxj = i + x[len(x)//2], j + y[len(y)//2]
            localmax_img[maxi][maxj] = 255
            localmax_pts.append([maxi, maxj])

    return localmax_img, localmax_pts

def harris_corner_detector(imgs):
    '''
        return: 
        1. keypoints: n x k x 2
            n: number of images
            k: number of keypoints for each image (i.e. k varies from images)
            2: [x, y]
        2. localmax_ims: n x h x w
            n: number of images
            h: image height
            w: image width
    '''
    ## convert img to gray
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    keypoints, localmax_imgs = [], []
    for i in range(len(imgs)):
        R = compute_R(imgs[i])
        k = 80 # window size
        localmax_img, localmax_points = find_local_maximum(R, k)
        keypoints.append(localmax_points)
        localmax_imgs.append(localmax_img)
    
    return keypoints, localmax_imgs