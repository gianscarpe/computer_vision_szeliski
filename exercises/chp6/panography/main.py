import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_panorama(img1, img2):
    h, w = img1.shape

    descriptor = cv2.ORB_create()
    panorama = np.zeros((h, w*2))
    
    kp1, des1 = descriptor.detectAndCompute(img1, None)
    kp2, des2  = descriptor.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher_create()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in raw_matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    
    if len(good)>10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        print("MATCH!")
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)        
        #        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        proj_img2 = cv2.warpPerspective(img2, M, dsize=(w*2, h))
        plt.imshow(proj_img2)
        plt.show()
        panorama[0:h, 0:w] = .5 * img1/ 255
        panorama += .5 * proj_img2 / 255
        panorama = (panorama * 255).astype('uint8')
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), 10) )
        raise Exception()

    return panorama

if __name__ == '__main__':
    img1 = cv2.imread('./img1.jpeg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./img2.jpeg', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('./img3.jpg', cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[2].imshow(img3)
    plt.show()
    
    panorama = get_panorama(img1, img2)
    panorama = get_panorama(img2, img3)
    plt.figure()
    plt.imshow(panorama)
    plt.show()
