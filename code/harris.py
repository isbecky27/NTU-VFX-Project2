from feature_description import SIFT_descriptor
import numpy as np
import copy
import cv2

def compute_R(img, ksize = 7, s = 5, k = 0.04):
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
             [Sxy, Sy2]]
    '''
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k * traceM ** 2
    
    return R, blur_img
 
def threshold_R(R, threshold = 0.04):
    R[R <= threshold * R.max()] = 0
    return R

def find_local_maximum(R, k):
    '''
        k: window size
    '''
    h, w = R.shape
    
    localmax_img = np.zeros((h, w), dtype='uint8')
    localmax_pts = []
    
    k = k // 2
    for i in range(k, h-k):
        for j in range(k, w-k):
            center = R[i][j]
            ismax = True
            for di in range(-k, k+1):
                for dj in range(-k, k+1):
                    if di == 0 and dj == 0:
                        continue
                    if R[i+di][j+dj] >= center:
                        ismax = False
                        break 
            if ismax:
                localmax_pts.append([i, j])
                localmax_img[i][j] = 255

    return localmax_img, localmax_pts

def draw_red_points(img, points):

    kp_img = copy.deepcopy(img)
    for point in points:
        cv2.circle(kp_img, (point[1], point[0]), 1, (0, 0, 255), 3)
    cv2.imshow('show key points', kp_img)
    cv2.waitKey(0)

    return kp_img

def output_img(img, name):
    cv2.imwrite(f'../myresult/{name}.png', img)
   
def harris_corner_detector(img):
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
    ori = copy.deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## find corner response R
    R, blur_img = compute_R(img)
    R = threshold_R(R)
    # cv2.imshow('show R', R)
    # cv2.waitKey(0)

    ## find local maximum on R
    localmax_img, localmax_points = find_local_maximum(R, 3)
    # cv2.imshow('show local maximum', localmax_img)
    # cv2.waitKey(0)
    
    ## draw keypoint on the original image
    # kp_img = draw_red_points(ori, localmax_points)

    return localmax_points, blur_img

# img = cv2.imread('./cow.jpg')
# keypoints, localmax_imgs = harris_corner_detector([img])

